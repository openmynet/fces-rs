use ndarray::Array2;

use crate::math::{ip_to_l2sq, softmax_rows};
use crate::types::KnnGraph;

/// NEP 二阶距离（Neighbor-based Edge Probability，基于邻居的边概率）。
///
/// ## 为什么需要 NEP
///
/// 一阶距离（如 L2、内积）衡量的是两个人脸向量"长得像不像"。
/// 但它有个致命缺陷：同一个人的不同照片，可能因为光照、角度、遮挡
/// 导致向量差别很大（一阶距离远），可它们共享大量共同邻居——
/// 这些邻居正是这个人的其他照片。
///
/// 举例：
/// ```text
///   张三_正面  ←→  张三_侧面    一阶距离很近？不一定，取决于光照
///   张三_正面  →  KNN邻居 = [张三_左脸, 张三_右脸, 张三_仰头, ...]
///   张三_侧面  →  KNN邻居 = [张三_左脸, 张三_右脸, 张三_低头, ...]
///   两者邻居列表高度重叠                            ↑↑↑↑↑↑↑↑↑↑↑↑
/// ```
///
/// NEP 捕捉的就是这种**结构相似性**——不看两个人脸向量本身有多近，
/// 而是看它们各自的一群邻居是否一致。邻居重合度越高，NEP 距离越小，
/// 越可能属于同一个人。
///
/// ## 计算过程（三步）
///
/// 1. **一阶距离 → 概率**：
///    把 KNN 返回的内积距离，用 softmax 转成概率分布 P[i]。
///    P[i][j] 可以理解为"节点 j 是节点 i 真正同类的概率"。
///
/// 2. **邻居投影（二阶交叉）**：
///    对于节点 i 和它的第 j 个邻居：
///    - 取出邻居 j 的概率分布 P[j]
///    - 把 P[j] 投射到 i 的邻居坐标系上（只看两者共有的邻居）
///    - 得到 y 矩阵：y[r][c] 表示"j 的第 r 个邻居，在 i 的列表中排第 c 位"
///
/// 3. **计算二阶概率 → NEP 距离**：
///    - 对每对 (r, c)，如果 y[r][c] 非零（说明该邻居同时被 i 和 j 关联）：
///      取 P[i][r] 和 y[r][c] 的平均值，累加到 tmp_dist[r]
///    - tmp_dist[j] 即 i 与 j 通过第 j 个共同邻居的"二阶关联概率"
///    - NEP 距离 = 1 - tmp_dist[j]，值越小越相似
///
/// ## 直观类比
///
/// 一阶距离 = 你俩长得像不像（看脸）
/// NEP 距离 = 你俩的朋友圈重合度（看社交关系）
///
/// ## 参数
/// - `knn`: KNN 图，包含邻居索引 nbrs (N, k) 和内积距离 dists (N, k)。
///
/// ## 返回
/// - `Array2<f32>`: NEP 距离矩阵 (N, k)，值域 [0, 1]，越小表示越相似。
///
/// 上级流程：由 lib::cluster 传入 KnnGraph。
/// 下级流程：ip_to_l2sq → softmax_rows → 二阶邻居交叉 → 1 - 二阶概率。
pub fn compute_nep(knn: &KnnGraph) -> Array2<f32> {
    const SIGMA: f32 = 0.5;

    let (n, k) = knn.nbrs.dim();

    // Step 1: 内积 → L2 平方距离
    let l2_dists = ip_to_l2sq(&knn.dists);

    // Step 2: Softmax 概率归一化
    let p = softmax_rows(&l2_dists, SIGMA);

    // Step 3: 二阶邻居距离
    let mut nep_dists = Array2::<f32>::zeros((n, k));

    // pos 数组复用：pos[node_id] = 该节点在 i 的邻居列表中的位置，-1 表示不在
    let mut pos = vec![-1i32; n];

    // 临时缓冲区复用
    let mut y = vec![0.0f32; k * k];
    let mut tmp_dist = vec![0.0f32; k];

    for i in 0..n {
        // 3a. 构建位置映射
        for j in 0..k {
            let nbr = knn.nbrs[[i, j]] as usize;
            pos[nbr] = j as i32;
        }

        // 3b. 填充 y 矩阵（一次性构建，对所有邻居 i 的所有邻居）
        // y[r][c] = sum_{where nbr_of_nbr∈i's list} P[nbr][idx_of_shared_nbr]
        // 对应参考代码中双层循环: x_ind(邻居维度) × y_ind(邻居的邻居维度)
        y.fill(0.0);
        for r in 0..k {
            let nbr_of_i = knn.nbrs[[i, r]] as usize; // i 的第 r 个邻居
            for c in 0..k {
                let shared = knn.nbrs[[nbr_of_i, c]] as usize; // 该邻居的第 c 个邻居
                let ppos = pos[shared];
                if ppos >= 0 {
                    // y[r][ppos] += ...  参考代码使用 = 赋值（最后一写有效）
                    y[r * k + ppos as usize] = p[[nbr_of_i, c]];
                }
            }
        }

        // 3c. 对每个邻居 j 计算 NEP 距离
        for j in 0..k {
            tmp_dist.fill(0.0);
            for l in 0..k {
                let y_val = y[j * k + l];
                if y_val != 0.0 {
                    tmp_dist[j] += (p[[i, l]] + y_val) / 2.0;
                }
            }
            nep_dists[[i, j]] = 1.0 - tmp_dist[j];
        }

        // 3d. 清理位置映射
        for j in 0..k {
            let nbr = knn.nbrs[[i, j]] as usize;
            pos[nbr] = -1;
        }
    }

    nep_dists
}
