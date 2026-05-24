use std::collections::HashMap;

use ndarray::Array2;

use crate::error::FcesError;
use crate::types::KnnGraph;

/// FC-ES 聚类核心。
///
/// # 参数
/// - `cosine_threshold`: 原始余弦相似度过滤阈值。`None` 或 `≤ 0` 跳过。
///
/// # 返回
/// - `Result<Vec<Vec<usize>>, FcesError>`: 每个聚类的成员索引列表。
pub fn run(
    knn: &KnnGraph,
    nep_dists: &Array2<f32>,
    theta: f32,
    cosine_threshold: Option<f32>,
) -> Result<Vec<Vec<usize>>, FcesError> {
    let n = knn.nbrs.dim().0;

    let (links, singles) = get_links(knn, nep_dists, theta, cosine_threshold);

    let infomap_results = crate::infomap::run_infomap(&links, n)?;

    // Step 3: 按模块分组（保留 .tree 文件行序）
    let mut module_order: Vec<u32> = Vec::new();
    let mut module_nodes: HashMap<u32, Vec<usize>> = HashMap::new();

    for &(node_id, module) in &infomap_results {
        if !module_nodes.contains_key(&module) {
            module_order.push(module);
        }
        module_nodes.entry(module).or_default().push(node_id);
    }

    // Step 4: 剔除哨兵节点（InfoMap 哨兵节点 ID 超出合法范围 [0, n)）
    for nodes in module_nodes.values_mut() {
        nodes.retain(|&node_id| node_id < n);
    }

    // Step 5: 构建有效簇列表
    let mut node_to_label: HashMap<usize, u32> = HashMap::new();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut next_label: u32 = 0;

    for module in &module_order {
        if let Some(nodes) = module_nodes.get(module) {
            if nodes.is_empty() {
                continue;
            }
            for &node_id in nodes {
                node_to_label.insert(node_id, next_label);
            }
            clusters.push(nodes.clone());
            next_label += 1;
        }
    }

    // Step 6: 合并孤立节点
    for &single in &singles {
        if !node_to_label.contains_key(&single) {
            node_to_label.insert(single, next_label);
            clusters.push(vec![single]);
            next_label += 1;
        }
    }

    Ok(clusters)
}

/// 构建连接图（Early Stopping）。
///
/// 用途：遍历每个节点 i 的 KNN 邻居（按内积降序），
///       若 sim = 1 - nep_dists[i][j] >= theta 则添加有向边，
///       一旦遇到第一个 sim < theta 立即停止（因为后续邻居只会更不相似）。
///
/// 上级流程：由 clustering::run 调用。
/// 下级流程：逐节点逐邻居判断 → 添加到 links HashMap → 收集孤立节点 singles → 返回 (links, singles)。
///
/// # 参数
/// - `knn`: KNN 图，提供每个节点的邻居索引。
/// - `nep_dists`: NEP 距离矩阵 (N, k)，用于计算相似度 sim = 1 - dist。
/// - `theta`: Early Stop 阈值。
///
/// # 返回
/// - `(HashMap<(usize, usize), f32>, Vec<usize>)`:
///   - links: 有向边集合，(src → dst, similarity)。
///   - singles: 没有建立任何连接的孤立节点列表。
pub fn get_links(
    knn: &KnnGraph,
    nep_dists: &Array2<f32>,
    theta: f32,
    cosine_threshold: Option<f32>,
) -> (HashMap<(usize, usize), f32>, Vec<usize>) {
    let (n, k) = knn.nbrs.dim();
    let cos_th = cosine_threshold.unwrap_or(0.0);
    let mut links: HashMap<(usize, usize), f32> = HashMap::new();
    let mut singles: Vec<usize> = Vec::new();

    for i in 0..n {
        let mut count = 0usize;
        let mut early_stop = false;

        for j in 0..k {
            let neighbor = knn.nbrs[[i, j]] as usize;

            // 跳过自环
            if i == neighbor {
                continue;
            }

            // 原始余弦相似度过滤
            if cos_th > 0.0 && knn.dists[[i, j]] < cos_th {
                early_stop = true;
                if early_stop {
                    break;
                }
                continue;
            }

            let dist = nep_dists[[i, j]];
            let sim = 1.0 - dist;

            if sim >= theta {
                // 相似度达标：添加有向边
                links.insert((i, neighbor), sim);
                count += 1;
            } else {
                // 相似度不达标：触发 Early Stop
                early_stop = true;
            }

            if early_stop {
                break;
            }
        }

        // 未建立任何连接的节点为孤立节点
        if count == 0 {
            singles.push(i);
        }
    }

    (links, singles)
}
