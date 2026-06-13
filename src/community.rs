use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use graphrs::{Edge, Graph, GraphSpecs};

use crate::error::FcesError;

static INFOMAP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Windows 平台下设置 CREATE_NO_WINDOW 标志，禁止控制台弹窗。
#[cfg(windows)]
fn suppress_no_window(cmd: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x08000000);
}

#[cfg(not(windows))]
fn suppress_no_window(_cmd: &mut std::process::Command) {}

/// 检查 Infomap 是否可用。
pub fn has_infomap() -> bool {
    find_infomap_path().is_some()
}

/// 查找 Infomap 可执行文件路径。
fn find_infomap_path() -> Option<String> {
    let search_dirs: Vec<PathBuf> = [
        std::env::current_dir().ok(),
        std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)),
        std::env::var("FCES_INFOMAP_DIR").ok().map(PathBuf::from),
    ]
    .into_iter()
    .flatten()
    .collect();

    for dir in &search_dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let name_lower = file_name.to_string_lossy().to_lowercase();
                let is_match = name_lower == "infomap" || name_lower == "infomap.exe";
                if is_match {
                    let full_path = dir.join(&file_name);
                    if full_path.is_file() {
                        return Some(full_path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    let mut probe_cmd = Command::new("Infomap");
    probe_cmd.arg("--version");
    suppress_no_window(&mut probe_cmd);
    let probe = probe_cmd.output();
    if probe.map_or(false, |o| o.status.success()) {
        return Some("Infomap".to_string());
    }

    None
}

fn find_infomap() -> String {
    find_infomap_path().unwrap_or_else(|| "Infomap".to_string())
}

/// InfoMap 社区发现。
///
/// # 返回
/// - `Result<Vec<(usize, u32)>, FcesError>`: (node_id, module_index) 列表。
pub fn run_infomap(
    links: &HashMap<(usize, usize), f32>,
    _num_nodes: usize,
) -> Result<Vec<(usize, u32)>, FcesError> {
    if links.is_empty() {
        return Ok(Vec::new());
    }

    let seq = INFOMAP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::var("FCES_TEMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join(format!("fces_infomap_{}_{}", std::process::id(), seq));
    fs::create_dir_all(&tmp_dir)?;

    let edge_path = tmp_dir.join("edges.txt");
    write_edge_list(links, &edge_path)?;

    let infomap_path = find_infomap();
    let mut infomap_cmd = Command::new(&infomap_path);
    infomap_cmd
        .current_dir(&tmp_dir)
        .arg("edges.txt")
        .arg(".")
        .arg("--two-level")
        .arg("--directed")
        .arg("--silent");
    suppress_no_window(&mut infomap_cmd);
    let output = infomap_cmd.output().map_err(|e| {
        FcesError::InfomapExecution(format!(
            "无法执行 Infomap: {}. 请确认已安装并在 PATH 或当前目录中",
            e
        ))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(FcesError::InfomapExecution(format!(
            "stdout: {}\nstderr: {}",
            stdout, stderr
        )));
    }

    let tree_path = find_tree_file(&tmp_dir)?;
    let result = parse_tree_file(&tree_path)?;

    let _ = fs::remove_dir_all(&tmp_dir);

    Ok(result)
}

/// 社区发现：优先 Infomap，不可用或执行失败时回退 graphrs Leiden。
pub fn detect_communities(
    links: &HashMap<(usize, usize), f32>,
    num_nodes: usize,
) -> Result<Vec<(usize, u32)>, FcesError> {
    if links.is_empty() {
        return Ok(Vec::new());
    }
    if has_infomap() {
        if let Ok(result) = run_infomap(links, num_nodes) {
            return Ok(result);
        }
    }
    run_leiden(links)
}

fn run_leiden(links: &HashMap<(usize, usize), f32>) -> Result<Vec<(usize, u32)>, FcesError> {
    let (graph, node_list) = build_undirected_graph(links)
        .map_err(|e| FcesError::CommunityDetection(format!("图构建失败: {}", e)))?;

    let partitions = graphrs::algorithms::community::leiden::leiden(
        &graph,
        true,
        graphrs::algorithms::community::leiden::QualityFunction::Modularity,
        None,
        None,
        None,
    )
    .map_err(|e| FcesError::CommunityDetection(format!("Leiden 失败: {}", e)))?;

    if partitions.is_empty() {
        return Ok(Vec::new());
    }

    let mut results: Vec<(usize, u32)> = Vec::new();
    for (community_idx, community) in partitions.iter().enumerate() {
        for &internal_idx in community {
            results.push((node_list[internal_idx], community_idx as u32));
        }
    }
    results.sort_by_key(|(n, _)| *n);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn m(edges: &[(usize, usize, f32)]) -> HashMap<(usize, usize), f32> {
        edges.iter().map(|&(s, d, w)| ((s, d), w)).collect()
    }

    fn nodes_in_links(links: &HashMap<(usize, usize), f32>) -> Vec<usize> {
        let mut nodes: Vec<usize> = links.keys().flat_map(|&(s, d)| vec![s, d]).collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }

    #[test]
    fn test_leiden_empty() {
        let links: HashMap<(usize, usize), f32> = HashMap::new();
        let result = run_leiden(&links).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_leiden_single_edge() {
        let links = m(&[(0, 1, 1.0)]);
        let result = run_leiden(&links).unwrap();
        assert_eq!(result.len(), 2);
        let l0 = result.iter().find(|(n, _)| *n == 0).unwrap().1;
        let l1 = result.iter().find(|(n, _)| *n == 1).unwrap().1;
        assert_eq!(l0, l1);
    }

    #[test]
    fn test_leiden_two_components() {
        let links = m(&[(0, 1, 1.0), (2, 3, 1.0)]);
        let result = run_leiden(&links).unwrap();
        assert_eq!(result.len(), 4);
        let l0 = result.iter().find(|(n, _)| *n == 0).unwrap().1;
        let l1 = result.iter().find(|(n, _)| *n == 1).unwrap().1;
        let l2 = result.iter().find(|(n, _)| *n == 2).unwrap().1;
        let l3 = result.iter().find(|(n, _)| *n == 3).unwrap().1;
        assert_eq!(l0, l1);
        assert_eq!(l2, l3);
        assert_ne!(l0, l2);
    }

    #[test]
    fn test_leiden_triangle() {
        let links = m(&[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
        let result = run_leiden(&links).unwrap();
        assert_eq!(result.len(), 3);
        let labels: Vec<u32> = result.iter().map(|(_, l)| *l).collect();
        assert!(labels.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn test_leiden_all_nodes_present() {
        let links = m(&[(5, 10, 1.0), (10, 15, 1.0), (20, 25, 1.0)]);
        let result = run_leiden(&links).unwrap();
        let result_nodes: Vec<usize> = result.iter().map(|(n, _)| *n).collect();
        let expected = nodes_in_links(&links);
        assert_eq!(result_nodes, expected);
    }

    #[test]
    fn test_leiden_sorted_output() {
        let links = m(&[(3, 0, 1.0), (2, 1, 1.0), (4, 5, 1.0)]);
        let result = run_leiden(&links).unwrap();
        for i in 1..result.len() {
            assert!(result[i].0 > result[i - 1].0);
        }
    }

    #[test]
    fn test_bidirectional_edge_merged() {
        let links = m(&[(0, 1, 0.1), (1, 0, 0.1)]);
        let result = run_leiden(&links).unwrap();
        assert_eq!(result.len(), 2);
        let l0 = result.iter().find(|(n, _)| *n == 0).unwrap().1;
        let l1 = result.iter().find(|(n, _)| *n == 1).unwrap().1;
        assert_eq!(l0, l1, "双向边合并后应形成同一社区");
    }

    #[test]
    fn test_unidirectional_edge_same_community() {
        let links = m(&[(0, 1, 1.0)]);
        let result = run_leiden(&links).unwrap();
        let l0 = result.iter().find(|(n, _)| *n == 0).unwrap().1;
        let l1 = result.iter().find(|(n, _)| *n == 1).unwrap().1;
        assert_eq!(l0, l1, "单向边在 Leiden 中应退化为无向边，归入同一社区");
    }

    #[test]
    fn test_asymmetric_vs_symmetric_divergence() {
        let links = m(&[(0, 1, 1.0), (1, 0, 0.01)]);
        let result = run_leiden(&links).unwrap();
        assert_eq!(result.len(), 2);
        let l0 = result.iter().find(|(n, _)| *n == 0).unwrap().1;
        let l1 = result.iter().find(|(n, _)| *n == 1).unwrap().1;
        assert_eq!(l0, l1, "Leiden 对称化后非对称边仍应形成同簇");
    }

    // ========== Infomap vs Leiden 对比测试 ==========

    fn to_clusters(results: &[(usize, u32)]) -> Vec<Vec<usize>> {
        let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
        for &(node, label) in results {
            map.entry(label).or_default().push(node);
        }
        let mut clusters: Vec<Vec<usize>> = map.into_values().collect();
        for c in &mut clusters {
            c.sort();
        }
        clusters.sort_by_key(|c| (c.len(), c[0]));
        clusters
    }

    fn cluster_sizes(clusters: &[Vec<usize>]) -> Vec<usize> {
        let mut sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
        sizes.sort();
        sizes
    }

    /// 比较两种算法的分区结构。若 Infomap 不可用则跳过。
    fn compare(name: &str, infomap: &[(usize, u32)], leiden: &[(usize, u32)], n: usize) {
        let i_clusters = to_clusters(infomap);
        let l_clusters = to_clusters(leiden);

        eprintln!("\n=== {} ===", name);
        eprintln!(
            "  Infomap: {} 簇 {:?}, 覆盖 {}/{} 节点",
            i_clusters.len(),
            cluster_sizes(&i_clusters),
            infomap.len(),
            n
        );
        eprintln!(
            "  Leiden:  {} 簇 {:?}, 覆盖 {}/{} 节点",
            l_clusters.len(),
            cluster_sizes(&l_clusters),
            leiden.len(),
            n
        );

        // 分区完全一致判定
        if i_clusters == l_clusters {
            eprintln!("  ✓ 分区完全一致");
            return;
        }

        // 成对一致性: 对每对节点检查是否同簇
        let mut agreements = 0u64;
        let mut total = 0u64;
        for a in 0..n {
            for b in (a + 1)..n {
                total += 1;
                let i_same = same_cluster(infomap, a, b);
                let l_same = same_cluster(leiden, a, b);
                if i_same == l_same {
                    agreements += 1;
                }
            }
        }
        let ratio = if total > 0 {
            agreements as f64 / total as f64
        } else {
            1.0
        };
        eprintln!(
            "  成对一致性: {}/{} ({:.1}%)",
            agreements,
            total,
            ratio * 100.0
        );

        // 差异不在少数节点
        if ratio < 0.95 {
            eprintln!("  ⚠ 显著差异 (一致性 < 95%)");
        }
    }

    fn same_cluster(results: &[(usize, u32)], a: usize, b: usize) -> bool {
        let label_a = results.iter().find(|(n, _)| *n == a).map(|(_, l)| *l);
        let label_b = results.iter().find(|(n, _)| *n == b).map(|(_, l)| *l);
        match (label_a, label_b) {
            (Some(la), Some(lb)) => la == lb,
            _ => false,
        }
    }

    #[test]
    fn test_infomap_vs_leiden_symmetric() {
        if !has_infomap() {
            eprintln!("跳过: Infomap 不可用");
            return;
        }
        // 全对称三角图: 每条边双向等权，两者应一致
        let links = m(&[
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
        ]);
        let infomap = run_infomap(&links, 3).unwrap();
        let leiden = run_leiden(&links).unwrap();
        compare("全对称三角图", &infomap, &leiden, 3);
    }

    #[test]
    fn test_infomap_vs_leiden_asymmetric_chain() {
        if !has_infomap() {
            eprintln!("跳过: Infomap 不可用");
            return;
        }
        // 单向链 A→B→C: Infomap 保留方向, Leiden 退化为无向
        let links = m(&[(0, 1, 1.0), (1, 2, 1.0)]);
        let infomap = run_infomap(&links, 3).unwrap();
        let leiden = run_leiden(&links).unwrap();
        compare("单向链 A→B→C", &infomap, &leiden, 3);
    }

    #[test]
    fn test_infomap_vs_leiden_weak_bridge() {
        if !has_infomap() {
            eprintln!("跳过: Infomap 不可用");
            return;
        }
        // 两个强团由弱单向边桥接: A↔B(强) C↔D(强) B→C(弱单向)
        let links = m(&[
            (0, 1, 1.0),
            (1, 0, 1.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
            (1, 2, 0.1),
        ]);
        let infomap = run_infomap(&links, 4).unwrap();
        let leiden = run_leiden(&links).unwrap();
        compare("弱单向桥接的两团", &infomap, &leiden, 4);
    }

    #[test]
    fn test_infomap_vs_leiden_disconnected() {
        if !has_infomap() {
            eprintln!("跳过: Infomap 不可用");
            return;
        }
        // 两个完全分离的组件: 两者应一致
        let links = m(&[(0, 1, 1.0), (2, 3, 1.0), (3, 4, 1.0)]);
        let infomap = run_infomap(&links, 5).unwrap();
        let leiden = run_leiden(&links).unwrap();
        compare("两个分离组件", &infomap, &leiden, 5);
    }
}

fn build_undirected_graph(
    links: &HashMap<(usize, usize), f32>,
) -> Result<(Graph<usize, ()>, Vec<usize>), graphrs::Error> {
    let mut nodes: Vec<usize> = links
        .keys()
        .flat_map(|&(s, d)| [s, d])
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    nodes.sort();

    let index: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut merged: HashMap<(usize, usize), f64> = HashMap::new();
    for ((src, dst), weight) in links {
        let u = index[src];
        let v = index[dst];
        let key = if u <= v { (u, v) } else { (v, u) };
        *merged.entry(key).or_default() += *weight as f64;
    }

    let mut graph = Graph::<usize, ()>::new(GraphSpecs::undirected_create_missing());
    let edges: Vec<Arc<Edge<usize, ()>>> = merged
        .iter()
        .map(|((u, v), w)| Edge::with_weight(*u, *v, *w))
        .collect();
    graph.add_edges(edges)?;
    Ok((graph, nodes))
}

fn write_edge_list(links: &HashMap<(usize, usize), f32>, path: &Path) -> Result<(), FcesError> {
    let mut file = File::create(path)?;
    writeln!(file, "# FC-ES edge list")?;
    for ((src, dst), weight) in links {
        writeln!(file, "{} {} {}", src, dst, weight)?;
    }
    Ok(())
}

fn find_tree_file(dir: &PathBuf) -> Result<PathBuf, FcesError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "tree") {
            return Ok(path);
        }
    }
    Err(FcesError::InfomapParse(
        "未找到 InfoMap 输出的 .tree 文件".into(),
    ))
}

pub fn parse_tree_file(path: &Path) -> Result<Vec<(usize, u32)>, FcesError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut results = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') || line.starts_with('*') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }

        let path_parts: Vec<&str> = parts[0].split(':').collect();
        if path_parts.is_empty() {
            continue;
        }

        let module_index: u32 = path_parts[0].parse().unwrap_or(0);
        let node_id: usize = parts[3].parse().unwrap_or(0);

        results.push((node_id, module_index));
    }

    Ok(results)
}
