use ndarray::Array2;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

use crate::error::FcesError;
use crate::math::l2_normalize;
use crate::types::KnnGraph;

/// 创建 USearch 索引配置。
///
/// # 参数
/// - `dim`: 向量维度。
/// - `metric`: 距离度量（IP / Cos / L2sq 等）。
fn make_index_options(dim: usize, metric: MetricKind) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
        metric,
        quantization: ScalarKind::F32,
        connectivity: 0,
        expansion_add: 0,
        expansion_search: 0,
        multi: false,
    }
}

/// 向索引中批量添加向量，key 使用行索引。
///
/// 上级流程：被 `build_knn_graph_ip_only` 和 `build_knn_graph` 调用。
/// 下级流程：逐行追加到 USearch HNSW 索引。
///
/// # 参数
/// - `index`: 已创建的 USearch 索引。
/// - `data`: 行优先的二维矩阵，每行为一个向量。
fn populate_index(index: &Index, data: &Array2<f32>) -> Result<(), FcesError> {
    let (n, _) = data.dim();
    index
        .reserve(n)
        .map_err(|e| FcesError::UsSearch(e.to_string()))?;
    for i in 0..n {
        let row = data.row(i);
        let slice = row.as_slice().ok_or_else(|| {
            FcesError::UsSearch(format!("第 {} 行内存不连续", i))
        })?;
        index
            .add(i as u64, slice)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;
    }
    Ok(())
}

/// 计算两个原始向量的余弦相似度。
///
/// `cos_sim = dot(a, b) / (|a| · |b|)`
///
/// # 参数
/// - `a`, `b`: 原始特征行视图。
/// - `norm_a`, `norm_b`: 预计算的 L2 范数。
fn cosine_similarity(
    a: ndarray::ArrayView1<f32>,
    b: ndarray::ArrayView1<f32>,
    norm_a: f32,
    norm_b: f32,
) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot / (norm_a * norm_b)
}

/// 构建 KNN 图（纯内积搜索路径）。
///
/// 用途：对 L2 归一化后的特征矩阵直接做 USearch IP 搜索，
///       为每个节点找到 k 个最近邻。不使用余弦预过滤。
///
/// 上级流程：由 `build_knn_graph` 在 `cosine_threshold ≤ 0` 时分派。
/// 下级流程：USearch HNSW 近似搜索 → `KnnGraph`。
///
/// # 参数
/// - `normalized`: L2 归一化后的特征矩阵 (N × ndim)。
/// - `k`: 每个节点的邻居数。
///
/// # 返回
/// - `KnnGraph`: 邻居索引矩阵和内积值矩阵。
fn build_knn_graph_ip_only(
    normalized: &Array2<f32>,
    k: usize,
) -> Result<KnnGraph, FcesError> {
    let (n, dim) = normalized.dim();
    let options = make_index_options(dim, MetricKind::IP);
    let index: Index =
        new_index(&options).map_err(|e| FcesError::UsSearch(e.to_string()))?;
    populate_index(&index, normalized)?;

    let mut nbrs = Array2::<u32>::zeros((n, k));
    let mut dists = Array2::<f32>::zeros((n, k));

    for i in 0..n {
        let query = normalized.row(i);
        let query_slice = query.as_slice().ok_or_else(|| {
            FcesError::UsSearch(format!("第 {} 行内存不连续", i))
        })?;
        let results = index
            .search(query_slice, k)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;

        let count = results.keys.len().min(k);
        for j in 0..count {
            nbrs[[i, j]] = results.keys[j] as u32;
            dists[[i, j]] = 1.0 - results.distances[j];
        }
    }

    Ok(KnnGraph { nbrs, dists })
}

/// 构建 KNN 图，支持可选的余弦预过滤 + 内积重排。
///
/// 若 `cosine_threshold` 为 `None` 或 `≤ 0.0`，使用原始内积搜索；
/// 否则使用两阶段搜索：先用余弦找到 2k 个候选，按原始特征余弦相似度过滤，
/// 再对过滤结果建立二次 IP 索引进行精确重排。
pub fn build_knn_graph(
    features: &Array2<f32>,
    k: usize,
    cosine_threshold: Option<f32>,
) -> Result<KnnGraph, FcesError> {
    let normalized = l2_normalize(features);
    let (n, dim) = normalized.dim();

    if n == 0 {
        return Err(FcesError::InvalidInput("特征矩阵为空".into()));
    }

    let effective_k = k.min(n);

    let threshold = cosine_threshold.unwrap_or(0.0);
    if threshold <= 0.0 {
        return build_knn_graph_ip_only(&normalized, effective_k);
    }

    let cos_k = (k * 2).min(n);

    // Stage 1: global cosine index
    let cos_options = make_index_options(dim, MetricKind::Cos);
    let cos_index: Index = new_index(&cos_options)
        .map_err(|e| FcesError::UsSearch(e.to_string()))?;
    populate_index(&cos_index, &normalized)?;

    // precompute original-feature norms for accurate cosine similarity
    let orig_norms: Vec<f32> = features
        .rows()
        .into_iter()
        .map(|row| {
            let sq_sum: f32 = row.iter().map(|&x| x * x).sum();
            sq_sum.sqrt().max(1e-12)
        })
        .collect();

    let ip_options = make_index_options(dim, MetricKind::IP);

    let mut nbrs = Array2::<u32>::zeros((n, effective_k));
    let mut dists = Array2::<f32>::zeros((n, effective_k));

    for i in 0..n {
        let query_row = normalized.row(i);
        let query_slice = query_row.as_slice().ok_or_else(|| {
            FcesError::UsSearch(format!("第 {} 行内存不连续", i))
        })?;
        let query_orig = features.row(i);
        let norm_i = orig_norms[i];

        // Stage 2a: cosine search for 2k candidates
        let cos_results = cos_index
            .search(query_slice, cos_k)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;

        // Stage 2b: filter by cosine similarity on original features
        let mut filtered: Vec<usize> = Vec::with_capacity(cos_results.keys.len());
        for &key in &cos_results.keys {
            let cand = key as usize;
            let cos_sim = cosine_similarity(
                query_orig,
                features.row(cand),
                norm_i,
                orig_norms[cand],
            );
            if cos_sim >= threshold {
                filtered.push(cand);
            }
        }

        if filtered.is_empty() {
            continue;
        }

        // Stage 3: build secondary IP index from filtered candidates
        let ip_index: Index = new_index(&ip_options)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;
        ip_index
            .reserve(filtered.len())
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;
        for (j, &orig_idx) in filtered.iter().enumerate() {
            let cand_row = normalized.row(orig_idx);
            let vec = cand_row.as_slice().ok_or_else(|| {
                FcesError::UsSearch(format!("第 {} 行内存不连续", orig_idx))
            })?;
            ip_index
                .add(j as u64, vec)
                .map_err(|e| FcesError::UsSearch(e.to_string()))?;
        }

        // Stage 4: IP re-rank on secondary index
        let ip_results = ip_index
            .search(query_slice, effective_k)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;

        let count = ip_results.keys.len().min(effective_k);
        for j in 0..count {
            nbrs[[i, j]] = filtered[ip_results.keys[j] as usize] as u32;
            dists[[i, j]] = 1.0 - ip_results.distances[j];
        }
    }

    Ok(KnnGraph { nbrs, dists })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_build_knn_graph_ip_only() {
        let features = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let graph = build_knn_graph(&features, 2, None).unwrap();
        assert_eq!(graph.nbrs.shape(), &[3, 2]);
        assert_eq!(graph.dists.shape(), &[3, 2]);
        // each node should find at least itself or neighbor
        for i in 0..3 {
            let row = graph.nbrs.row(i);
            let dist_row = graph.dists.row(i);
            for j in 0..2 {
                assert!(row[j] < 3, "neighbor index out of range");
                assert!(dist_row[j] >= 0.0 && dist_row[j] <= 1.0);
            }
        }
    }

    #[test]
    fn test_build_knn_graph_two_stage() {
        let features = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]]);
        let graph = build_knn_graph(&features, 2, Some(0.5)).unwrap();
        assert_eq!(graph.nbrs.shape(), &[4, 2]);
        assert_eq!(graph.dists.shape(), &[4, 2]);
    }

    #[test]
    fn test_build_knn_graph_threshold_zero() {
        let features = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let graph = build_knn_graph(&features, 1, Some(0.0)).unwrap();
        assert_eq!(graph.nbrs.shape(), &[2, 1]);
    }
}
