use ndarray::Array2;
use ndarray_npy::read_npy;

#[test]
fn test_read_features() {
    let features: Array2<f32> =
        read_npy("data/features.npy").expect("读取 features.npy 失败");

    let (n, dim) = features.dim();
    assert_eq!(n, 12, "期望 12 个样本，实际 N={}", n);
    assert_eq!(dim, 512, "期望 512 维特征，实际 dim={}", dim);
}

#[test]
fn test_cluster_pipeline() {
    let features: Array2<f32> =
        read_npy("data/features.npy").expect("读取 features.npy 失败");

    let (n, _dim) = features.dim();

    if !fces::infomap::has_infomap() {
        eprintln!("跳过聚类管线测试：Infomap 未安装或不在 PATH/当前目录中");
        eprintln!("请将 Infomap 可执行文件放入 PATH 或项目根目录");
        return;
    }

    let clusters = fces::cluster(&features, Some(0.12), None, None)
        .expect("聚类执行失败");

    let total: usize = clusters.iter().map(|c| c.len()).sum();
    assert_eq!(total, n, "所有 {} 个节点都应被分配到簇，实际分配 {}", n, total);

    for (i, c) in clusters.iter().enumerate() {
        assert!(!c.is_empty(), "簇 {} 为空", i);
    }

    let num_clusters = clusters.len();
    assert!(num_clusters <= n, "簇数 {} 超过节点数 {}", num_clusters, n);

    eprintln!("======== 聚类结果 ========");
    eprintln!("节点总数: {}", n);
    eprintln!("簇数量:   {}", num_clusters);
    for (i, c) in clusters.iter().enumerate() {
        eprintln!("  簇 {} ({} 个节点): {:?}", i, c.len(), c);
    }
    eprintln!("=========================");
}
