use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::FcesError;

/// 检查 Infomap 是否可用。
pub fn has_infomap() -> bool {
    find_infomap_path().is_some()
}

/// 查找 Infomap 可执行文件路径。
fn find_infomap_path() -> Option<String> {
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(entries) = fs::read_dir(&cwd) {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let name_lower = file_name.to_string_lossy().to_lowercase();
                let is_match = name_lower == "infomap" || name_lower == "infomap.exe";
                if is_match {
                    let full_path = cwd.join(&file_name);
                    if full_path.is_file() {
                        return Some(full_path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    let probe = Command::new("Infomap").arg("--version").output();
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

    let tmp_dir = std::env::temp_dir().join(format!("fces_infomap_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir)?;

    let edge_path = tmp_dir.join("edges.txt");
    write_edge_list(links, &edge_path)?;

    let infomap_path = find_infomap();
    let output = Command::new(&infomap_path)
        .current_dir(&tmp_dir)
        .arg("edges.txt")
        .arg(".")
        .arg("--two-level")
        .arg("--directed")
        .arg("--silent")
        .output()
        .map_err(|e| FcesError::InfomapExecution(format!(
            "无法执行 Infomap: {}. 请确认已安装并在 PATH 或当前目录中", e
        )))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(FcesError::InfomapExecution(format!(
            "stdout: {}\nstderr: {}", stdout, stderr
        )));
    }

    let tree_path = find_tree_file(&tmp_dir)?;
    let result = parse_tree_file(&tree_path)?;

    let _ = fs::remove_dir_all(&tmp_dir);

    Ok(result)
}

fn write_edge_list(
    links: &HashMap<(usize, usize), f32>,
    path: &Path,
) -> Result<(), FcesError> {
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
    Err(FcesError::InfomapParse("未找到 InfoMap 输出的 .tree 文件".into()))
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
