use ndarray::Array2;

/// KNN 图：每个节点的 k 个最近邻及其内积值。
pub struct KnnGraph {
    /// 邻居索引，形状 (N, k)，按内积降序排列
    pub nbrs: Array2<u32>,
    /// 对应内积值，形状 (N, k)，值域约 [0, 1]
    pub dists: Array2<f32>,
}
