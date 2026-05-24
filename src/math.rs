use ndarray::{Array2, Axis};

/// L2 归一化（按行）。
///
/// 用途：将特征矩阵每行除以其 L2 范数，使每行成为单位向量。
///
/// 上级流程：由 knn::build_knn_graph 在搜索前调用。
/// 下级流程：逐行计算范数 → 逐元素除以范数 → 返回归一化后的 Array2。
///
/// # 参数
/// - `features`: 原始特征矩阵 (N × ndim)。
///
/// # 返回
/// - `Array2<f32>`: L2 归一化后的特征矩阵，每行范数为 1。
pub fn l2_normalize(features: &Array2<f32>) -> Array2<f32> {
    let norms = features.map_axis(Axis(1), |row| {
        let sq_sum: f32 = row.iter().map(|x| x * x).sum();
        sq_sum.sqrt().max(1e-12)
    });
    features / norms.insert_axis(Axis(1))
}

/// Softmax 归一化（按行，带温度参数 sigma）。
///
/// 用途：将距离矩阵转换为概率分布，sigma 控制分布尖锐程度。
///
/// 上级流程：由 nep::compute_nep 调用，将 L2² 距离转为概率 P。
/// 下级流程：逐行 exp(-dist/sigma) → 逐行除以行和 → 返回概率矩阵 Array2。
///
/// # 参数
/// - `dists`: 距离矩阵 (N, k)，每个元素为距离值。
/// - `sigma`: 温度参数，默认 0.5，越小分布越尖锐。
///
/// # 返回
/// - `Array2<f32>`: 概率矩阵 (N, k)，每行和为 1。
pub fn softmax_rows(dists: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let (n, k) = dists.dim();
    let mut result = Array2::<f32>::zeros((n, k));

    for i in 0..n {
        let row = dists.row(i);
        let max_val: f32 = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;

        for j in 0..k {
            let val = ((row[j] - max_val) / (-sigma)).exp();
            result[[i, j]] = val;
            sum += val;
        }

        if sum > 0.0 {
            for j in 0..k {
                result[[i, j]] /= sum;
            }
        }
    }

    result
}

/// 内积值 → L2 平方距离。
///
/// 匹配 Python 参考实现：
///   1. clip(ip_dists, 0.0, 1.0)   — 忽略负内积（Python nep_distance2.py:69）
///   2. l2² = 2 - 2*ip             — 2-2*cos_sim，值域 [0, 2]
///
/// # 参数
/// - `ip_dists`: 内积值（余弦相似度）矩阵 (N, k)。
///
/// # 返回
/// - `Array2<f32>`: L2 平方距离矩阵 (N, k)，值域 [0, 2]。
pub fn ip_to_l2sq(ip_dists: &Array2<f32>) -> Array2<f32> {
    ip_dists.mapv(|ip| {
        let ip = ip.clamp(0.0, 1.0);
        2.0 - 2.0 * ip
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_l2_normalize() {
        let v = arr2(&[[3.0, 4.0], [0.0, 5.0]]);
        let normed = l2_normalize(&v);
        assert!((normed[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((normed[[0, 1]] - 0.8).abs() < 1e-6);
        assert!((normed[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_row() {
        let v = arr2(&[[0.0, 0.0]]);
        let normed = l2_normalize(&v);
        // Row should remain all zeros (no division by zero)
        assert_eq!(normed[[0, 0]], 0.0);
        assert_eq!(normed[[0, 1]], 0.0);
    }

    #[test]
    fn test_softmax_rows() {
        let v = arr2(&[[1.0, 2.0, 3.0]]);
        let p = softmax_rows(&v, 1.0);
        let row_sum: f32 = p.row(0).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
        // exp(-dist/sigma): 距离越大 → 概率越低
        assert!(p[[0, 2]] < p[[0, 1]]);
        assert!(p[[0, 1]] < p[[0, 0]]);
    }

    #[test]
    fn test_ip_to_l2sq() {
        let ip = arr2(&[[1.0, 0.5, 0.0, -0.5]]);
        let l2 = ip_to_l2sq(&ip);
        assert!((l2[[0, 0]] - 0.0).abs() < 1e-6); // 2 - 2*1.0 = 0
        assert!((l2[[0, 1]] - 1.0).abs() < 1e-6); // 2 - 2*0.5 = 1
        assert!((l2[[0, 2]] - 2.0).abs() < 1e-6); // 2 - 2*0.0 = 2
        assert!((l2[[0, 3]] - 2.0).abs() < 1e-6); // clip(-0.5,0,1)=0 → 2 - 0 = 2
    }
}
