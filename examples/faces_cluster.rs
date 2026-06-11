use std::collections::HashMap;
use std::fs;

use ndarray::{s, Array2};
use ndarray_npy::read_npy;

fn load_id_path_map(csv_path: &str) -> HashMap<u32, String> {
    let content = fs::read_to_string(csv_path).expect("读取 CSV 失败");
    let mut map = HashMap::new();
    for line in content.lines().skip(1) {
        if let Some((id_str, path)) = line.split_once(',') {
            if let Ok(id) = id_str.parse::<u32>() {
                map.insert(id, path.to_string());
            }
        }
    }
    map
}

fn main() {
    let data: Array2<f32> = read_npy("data/faces_features.npy")
        .expect("读取 data/faces_features.npy 失败，请先运行 local/dump.py");
    let id_path = load_id_path_map("data/faces_export.csv");

    let (n, cols) = data.dim();
    let ids = data.column(0).to_owned();
    let features: Array2<f32> = data.slice(s![.., 1..]).to_owned();
    let dim = cols - 1;
    println!("加载特征: {} × {}", n, dim);

    if !fces::community::has_infomap() {
        eprintln!("Infomap 未安装，跳过聚类。请将 Infomap 放入 PATH 或项目根目录。");
        return;
    }

    match fces::cluster(&features, Some(0.12), Some(true), Some(0.5)) {
        Ok(clusters) => {
            println!("聚类结果 (theta=0.12, cosine_threshold=0.5): {} 个节点 → {} 个簇", n, clusters.len());
            for (i, c) in clusters.iter().enumerate() {
                let mut members: Vec<(u32, &str)> = c
                    .iter()
                    .map(|&idx| {
                        let fid = ids[idx] as u32;
                        (fid, id_path.get(&fid).map(|s| s.as_str()).unwrap_or("?"))
                    })
                    .collect();
                members.sort_unstable_by_key(|(fid, _)| *fid);
                println!("  簇 {} ({} 人):", i, c.len());
                for (fid, path) in &members {
                    let fname = std::path::Path::new(path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(path);
                    println!("    {}  {}", fid, fname);
                }
            }
        }
        Err(e) => {
            eprintln!("聚类失败: {}", e);
        }
    }
}
