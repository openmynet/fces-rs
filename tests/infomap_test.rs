use ndarray::Array2;
use ndarray_npy::read_npy;

const INFOMAP_DIR: &str = r"./local";

fn setup_infomap_env() {
    unsafe {
        std::env::set_var("FCES_INFOMAP_DIR", INFOMAP_DIR);
        std::env::set_var("FCES_TEMP_DIR", INFOMAP_DIR);
    }
}

#[test]
fn test_has_infomap_with_temp_dir() {
    setup_infomap_env();

    assert!(
        fces::community::has_infomap(),
        "Infomap 不可用，请确认 Infomap 可执行文件在 PATH 或当前目录中"
    );
}

#[test]
fn test_cluster_infomap_pipeline() {
    setup_infomap_env();

    if !fces::community::has_infomap() {
        eprintln!("跳过 cluster_infomap 管线测试：Infomap 不可用");
        return;
    }

    let features: Array2<f32> =
        read_npy("data/features.npy").expect("读取 features.npy 失败");

    let (n, _dim) = features.dim();

    let clusters = fces::cluster_infomap(&features, None, Some(0.12), None, None)
        .expect("cluster_infomap 执行失败");

    let total: usize = clusters.iter().map(|c| c.len()).sum();
    assert_eq!(total, n, "所有 {} 个节点都应被分配到簇，实际分配 {}", n, total);

    for (i, c) in clusters.iter().enumerate() {
        assert!(!c.is_empty(), "簇 {} 为空", i);
    }

    let num_clusters = clusters.len();
    assert!(num_clusters <= n, "簇数 {} 超过节点数 {}", num_clusters, n);

    eprintln!("======== cluster_infomap 结果 ========");
    eprintln!("节点总数: {}", n);
    eprintln!("簇数量:   {}", num_clusters);
    for (i, c) in clusters.iter().enumerate() {
        eprintln!("  簇 {} ({} 个节点): {:?}", i, c.len(), c);
    }
    eprintln!("======================================");
}
