# fces

FC-ES (Face Cluster Early Stop) 无监督人脸聚类算法 Rust 实现。

参考实现：[FC-ESER](https://github.com/jumptoliujj/FC-ESER)

## 管线

```
特征矩阵 (N × D) → KNN 图构建 → NEP 二阶距离 → FC-ES 聚类 → 簇列表
```

| 阶段 | 模块 | 说明 |
|------|------|------|
| 一 | `knn.rs` | USearch 内积搜索，构建 KNN 图 |
| 二 | `nep.rs` | Neighbor-based Edge Probability 二阶距离 |
| 三 | `clustering.rs` | Early Stop 连接 + InfoMap 社区发现 |

## 依赖

- [usearch](https://github.com/unum-cloud/usearch) — HNSW 近似最近邻搜索
- [ndarray](https://github.com/rust-ndarray/ndarray) — N 维数组运算
- [Infomap](https://mapequation.github.io/infomap/) — 社区发现 CLI（需单独安装）

## 使用

```rust
use ndarray::Array2;
use ndarray_npy::read_npy;
use fces::cluster;

let features: Array2<f32> = read_npy("data/features.npy")?;
let clusters = cluster(&features, Some(0.22), None, None);
let clusters = cluster(&features, None, Some(true), None);       // 默认 theta, 去掉单元素簇
```

## 示例

```bash
cargo run --example basic
```

输出：

```
加载特征: 12 × 512
聚类结果: 12 个节点 → 7 个簇
  簇 0 (4 人): [11, 8, 6, 7]
  簇 1 (2 人): [3, 9]
  簇 2 (2 人): [4, 5]
  簇 3 (1 人): [0]
  ...
```

## 测试

```bash
# 全部测试
cargo test

# 仅集成测试（需要 Infomap CLI 和 data/features.npy）
cargo test --test cluster_test

# 仅单元测试
cargo test --lib

# 带输出运行
cargo test -- --nocapture
```

## License

MIT
