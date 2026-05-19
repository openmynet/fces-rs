use ndarray::Array2;
use ndarray_npy::read_npy;

fn main() {
    let features: Array2<f32> = read_npy("data/features.npy")
        .expect("读取 data/features.npy 失败，请确认文件存在");

    let (n, dim) = features.dim();
    println!("加载特征: {} × {}", n, dim);

    if !fces::infomap::has_infomap() {
        eprintln!("Infomap 未安装，跳过聚类。请将 Infomap 放入 PATH 或项目根目录。");
        return;
    }

    match fces::cluster(&features, Some(0.12), None) {
        Ok(clusters) => {
            println!("聚类结果: {} 个节点 → {} 个簇", n, clusters.len());
            for (i, c) in clusters.iter().enumerate() {
                println!("  簇 {} ({} 人): {:?}", i, c.len(), c);
            }
        }
        Err(e) => {
            eprintln!("聚类失败: {}", e);
        }
    }
}
