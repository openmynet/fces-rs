use ndarray::Array2;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

use crate::error::FcesError;
use crate::math::l2_normalize;
use crate::types::KnnGraph;

/// 使用 USearch 构建 KNN 图（内积搜索）。
///
/// 用途：对 L2 归一化后的特征矩阵做内积搜索，为每个节点找到 k 个最近邻。
///       USearch 使用 HNSW 近似搜索。
///
/// # 参数
/// - `features`: 特征矩阵 (N × ndim)。
/// - `k`: KNN 邻居数。
///
/// # 返回
/// - `Result<KnnGraph, FcesError>`: KNN 图或错误。
pub fn build_knn_graph(features: &Array2<f32>, k: usize) -> Result<KnnGraph, FcesError> {
    let normalized = l2_normalize(features);
    let (n, dim) = normalized.dim();

    if n == 0 {
        return Err(FcesError::InvalidInput("特征矩阵为空".into()));
    }

    let effective_k = k.min(n);

    let options = IndexOptions {
        dimensions: dim,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: 0,
        expansion_add: 0,
        expansion_search: 0,
        multi: false,
    };

    let index: Index = new_index(&options).map_err(|e| FcesError::UsSearch(e.to_string()))?;
    index.reserve(n).map_err(|e| FcesError::UsSearch(e.to_string()))?;

    for i in 0..n {
        let row = normalized.row(i);
        let vec = row
            .as_slice()
            .ok_or_else(|| FcesError::UsSearch(format!("第 {} 行内存不连续", i)))?;
        index
            .add(i as u64, vec)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;
    }

    let mut nbrs = Array2::<u32>::zeros((n, effective_k));
    let mut dists = Array2::<f32>::zeros((n, effective_k));

    for i in 0..n {
        let row = normalized.row(i);
        let query = row
            .as_slice()
            .ok_or_else(|| FcesError::UsSearch(format!("第 {} 行内存不连续", i)))?;
        let results = index
            .search(query, effective_k)
            .map_err(|e| FcesError::UsSearch(e.to_string()))?;

        let count = results.keys.len().min(effective_k);
        for j in 0..count {
            nbrs[[i, j]] = results.keys[j] as u32;
            dists[[i, j]] = 1.0 - results.distances[j];
        }
    }

    Ok(KnnGraph { nbrs, dists })
}
