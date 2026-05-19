mod types;
mod math;
mod knn;
mod nep;
mod clustering;
pub mod error;
pub mod infomap;

use ndarray::Array2;

use crate::error::FcesError;

/// FC-ES 人脸聚类（自动 k）。
///
/// # 参数
/// - `features`: 特征矩阵 (N × ndim)。
/// - `theta`: 相似度门槛，选填，取值 0~1，默认 0.22。
/// - `drop_singletons`: 是否过滤掉单元素簇，选填，默认 false。
///
/// # 返回
/// - `Result<Vec<Vec<usize>>, FcesError>`: 每个聚类的成员索引列表。
pub fn cluster(
    features: &Array2<f32>,
    theta: Option<f32>,
    drop_singletons: Option<bool>,
) -> Result<Vec<Vec<usize>>, FcesError> {
    cluster_with_k(features, None, theta, drop_singletons)
}

/// FC-ES 人脸聚类（自定义 k）。
pub fn cluster_with_k(
    features: &Array2<f32>,
    k: Option<usize>,
    theta: Option<f32>,
    drop_singletons: Option<bool>,
) -> Result<Vec<Vec<usize>>, FcesError> {
    let (n, _) = features.dim();
    let k = k.unwrap_or_else(|| 80.min(n));
    let theta = theta.unwrap_or(0.22);
    let drop = drop_singletons.unwrap_or(false);
    run_pipeline(features, k, theta, drop)
}

fn run_pipeline(
    features: &Array2<f32>,
    k: usize,
    theta: f32,
    drop_singletons: bool,
) -> Result<Vec<Vec<usize>>, FcesError> {
    let knn = knn::build_knn_graph(features, k)?;
    let nep_dists = nep::compute_nep(&knn);
    let mut clusters = clustering::run(&knn, &nep_dists, theta)?;

    if drop_singletons {
        clusters.retain(|c| c.len() > 1);
    }

    Ok(clusters)
}
