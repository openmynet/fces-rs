use ndarray::Array2;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

use crate::error::FcesError;
use crate::math::l2_normalize;
use crate::types::KnnGraph;

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

/// 构建 KNN 图。
///
/// 对 L2 归一化后的特征矩阵做 USearch IP 搜索，
/// 为每个节点找到 k 个最近邻（余弦预过滤已在 clustering 阶段处理）。
pub fn build_knn_graph(
    features: &Array2<f32>,
    k: usize,
    _cosine_threshold: Option<f32>,
) -> Result<KnnGraph, FcesError> {
    let normalized = l2_normalize(features);
    let (n, dim) = normalized.dim();

    if n == 0 {
        return Err(FcesError::InvalidInput("特征矩阵为空".into()));
    }

    let effective_k = k.min(n);

    let options = make_index_options(dim, MetricKind::IP);
    let index: Index =
        new_index(&options).map_err(|e| FcesError::UsSearch(e.to_string()))?;
    populate_index(&index, &normalized)?;

    let mut nbrs = Array2::<u32>::zeros((n, effective_k));
    let mut dists = Array2::<f32>::zeros((n, effective_k));

    for i in 0..n {
        let query = normalized.row(i);
        let query_slice = query.as_slice().ok_or_else(|| {
            FcesError::UsSearch(format!("第 {} 行内存不连续", i))
        })?;
        let results = index
            .search(query_slice, effective_k)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;

        let count = results.keys.len().min(effective_k);
        for j in 0..count {
            nbrs[[i, j]] = results.keys[j] as u32;
            dists[[i, j]] = 1.0 - results.distances[j];
        }
    }

    Ok(KnnGraph { nbrs, dists })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_build_knn_graph() {
        let features = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let graph = build_knn_graph(&features, 2, None).unwrap();
        assert_eq!(graph.nbrs.shape(), &[3, 2]);
        assert_eq!(graph.dists.shape(), &[3, 2]);
        for i in 0..3 {
            let row = graph.nbrs.row(i);
            let dist_row = graph.dists.row(i);
            for j in 0..2 {
                assert!(row[j] < 3, "neighbor index out of range");
                assert!(dist_row[j] >= 0.0 && dist_row[j] <= 1.0);
            }
        }
    }
}
