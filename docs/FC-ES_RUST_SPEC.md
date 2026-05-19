# FC-ES: 完整 Rust 实现规范

> 目标：用 Rust 从零实现 FC-ES (Face Clustering via Early Stopping) 无监督人脸聚类算法。
> 本文档面向编程代理，给出每一步的精确输入/输出规格、算法伪代码、数据格式和边缘情况。

---

## 目录

1. [管线总览](#1-管线总览)
2. [阶段一：KNN 图构建](#2-阶段一knn-图构建)
3. [阶段二：NEP 二阶距离](#3-阶段二nep-二阶距离)
4. [阶段三：FC-ES 聚类](#4-阶段三fc-es-聚类)
5. [阶段四：评估指标](#5-阶段四评估指标)
6. [数据格式规范](#6-数据格式规范)
7. [Rust 实现建议](#7-rust-实现建议)
8. [测试验证方案](#8-测试验证方案)

---

## 1. 管线总览

```
┌─────────────────────┐
│  阶段一：KNN 图构建   │  features/*.bin  →  knn_nbrs.npz / knn_dists.npz
├─────────────────────┤
│  阶段二：NEP 距离     │  knn_nbrs.npz + knn_dists.npz  →  knn_dists_trans2.npz
├─────────────────────┤
│  阶段三：FC-ES 聚类   │  knn_nbrs.npz + knn_dists_trans2.npz  →  聚类标签
├─────────────────────┤
│  阶段四：评估         │  聚类标签 + 真实标签  →  pairwise / bcubed / nmi
└─────────────────────┘
```

---

## 2. 阶段一：KNN 图构建

### 2.1 输入

| 输入 | 格式 | 说明 |
|------|------|------|
| 特征文件 | 原始二进制 `.bin` | `N × 256` 个 float32，行主序 (row-major) |
| k | 整数 | 默认 80 |
| 标签文件 | 文本 `.meta` | N 行，每行一个整数类别 ID（仅用于读取 N 值，不参与 KNN） |

### 2.2 处理步骤

```
Step 1: 读取特征
  - 从 .bin 读取所有 float32 值
  - reshape 为 (N, 256)
  - feature_dim = 256

Step 2: L2 归一化 (每行)
  for i in 0..N:
      norm = sqrt(sum(features[i][j]^2 for j in 0..256))
      for j in 0..256:
          features[i][j] /= norm

Step 3: KNN 内积搜索 (k=80)
  - 使用内积 (Inner Product / dot product) 作为相似度度量
  - 对每个查询向量 q，找到与数据库向量集中内积最大的 k 个向量
  - 内积距离 = 1.0 - 内积值  (只存内积值)
  - 输出: nbrs[N][k] (邻居索引, int32), dists[N][k] (内积值, float32)
  - 注意: 自己与自己的内积 ≈ 1.0 (经 L2 归一化后)，包含在结果中
```

### 2.3 输出

两个 `.npz` 文件（可替换为自定义二进制格式）：

| 文件 | 键名 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| `knn_nbrs.npz` | `data` | `(N, k)` | int32 | 第 i 行的 k 个最近邻居索引 |
| `knn_dists.npz` | `data` | `(N, k)` | float32 | 对应的内积值，值域约 [0, 1] |

**关键点：邻居按内积降序排列（相似度从高到低）。内积值越高表示越相似。**

### 2.4 KNN 算法细节

```
输入: features[N][D], k=80
输出: nbrs[N][k], dists[N][k]

算法: 内积全搜索 (Inner Product Brute Force)
  - 计算 query[i] 与所有 target[j] 的内积: dot(query[i], target[j])
  - 取内积最大的 k 个作为邻居
  - nbrs[i] 按内积降序排列
  - dists[i] 存储原始内积值（不是 1 - 内积）
```

### 2.5 内存估算

| 规模 | N | 特征内存 | KNN 输出 | 总计 |
|------|---|---------|----------|------|
| 小 | 50K | 50MB | ~30MB | ~80MB |
| 中 | 500K | 500MB | ~300MB | ~800MB |
| 大 | 5.8M (MS1M) | ~6GB | ~3.5GB | ~10GB |

> KNN 搜索需 O(N²) 时间，大规模数据建议 GPU 加速或近似搜索（如 FAISS/HNSW）。

---

## 3. 阶段二：NEP 二阶距离

### 3.1 输入

| 输入 | 来源 | 形状 | 类型 |
|------|------|------|------|
| nbrs | 阶段一输出 | `(N, k)` | int32 |
| dists | 阶段一输出 | `(N, k)` | float32（内积值） |

### 3.2 算法完整伪代码

```python
# ============================================================
# NEP 距离计算 (Neighbor-based Edge Probability)
# 输入: knn[N][k] (邻居索引), ip_dists[N][k] (内积值)
# 输出: nep_dists[N][k] (NEP距离, 值域[0, 1])
# ============================================================

SIGMA = 0.5  # softmax温度参数

# Step 1: 内积距离 → L2平方距离
#   l2² = 2 - 2 * inner_product
#   (因为向量已 L2 归一化，|x|² + |y|² - 2*xy = 2 - 2*xy)
l2_dists = clip(2.0 - 2.0 * ip_dists, 0.0, 1.0)

# Step 2: Softmax 概率归一化 (每行)
#   P[i][j] = exp(-l2_dists[i][j] / SIGMA) / sum(exp(-l2_dists[i][l] / SIGMA) for l in 0..k-1)
P = new float[N][k]
for i in 0..N:
    for j in 0..k:
        P[i][j] = exp(-l2_dists[i][j] / SIGMA)
    row_sum = sum(P[i])  # 如果 row_sum == 0, 则设 row_sum = 1 (防除零)
    for j in 0..k:
        P[i][j] = P[i][j] / row_sum

# Step 3: 二阶邻居距离 (核心)
# 对每个节点 i，计算它与其k个邻居之间的 NEP 距离
nep_dists = new float[N][k]

# 并行: 将 N 个节点均匀分配到 P 个线程
# 分配方式: sid = thread_i * ceil(N / num_threads)
#          eid = min((thread_i+1) * ceil(N / num_threads), N)

for i in sid..eid:
    # 3a. 构建位置映射
    # pos 是一个临时数组 [N]，初始化为 -1
    # pos[knn[i][j]] = j  (记录节点i的第j个邻居在i的邻居列表中的位置)
    for j in 0..k:
        pos[knn[i][j]] = j

    # 3b. 对每个邻居 j，计算二阶概率 y
    for j in 0..k:
        # knn_tmp = knn[ knn[i][j] ]  即邻居j 的k个邻居
        # P_tmp   = P[ knn[i][j] ]    即邻居j 的概率分布
        
        # y 是 (k, k) 的临时矩阵，初始化为全0
        y = zeros(k, k)
        
        # 遍历 knn_tmp 的每一行和每一列:
        for r in 0..k:           # 邻居j 的第r个邻居
            for c in 0..k:       # （遍历k列，实际上看邻居j的第r个邻居在i的邻居列表中的位置）
                nbr_idx = knn_tmp[r][c]
                if pos[nbr_idx] >= 0:   # 如果这个邻居在i的邻居列表中
                    ppos = pos[nbr_idx]
                    y[r][ppos] = P_tmp[r][c]
        
        # 计算 NEP 距离: 对于节点i和它的第j个邻居
        # tmp_dist[r] = (P[i][r] + y[r][:]) * (y[r][:] != 0)
        #              对于每一列，如果 y[r][c] != 0:
        #                  value = (P[i][r] + y[r][c]) / 2
        #              否则 value = 0
        # tmp_dist[r] = sum of all such values
        tmp_dist = zeros(k)
        for r in 0..k:
            for c in 0..k:
                if y[r][c] != 0.0:
                    tmp_dist[r] += (P[i][r] + y[r][c]) / 2.0
        
        # nep_dists[i][j] = 1 - 相应的二阶概率
        nep_dists[i][j] = 1.0 - 上述 tmp_dist 的对应值
        # 实际上原代码是: nep_dists[i][j] = 1 - tmp_dist[j]
        # 因为对于i和它的第j个邻居，我们关心的是第j个位置
    
    # 3c. 清理: 恢复 pos 数组
    for j in 0..k:
        pos[knn[i][j]] = -1

# 更精确的等价写法 (参考原代码):
for i in sid..eid:
    # 设置 pos
    for j in 0..k:
        pos[knn[i][j]] = j
    
    for j in 0..k:
        y = zeros(k, k)
        
        # 遍历邻居j的k个邻居
        nbrs_of_j = knn[knn[i][j]]  # shape (k,)
        P_of_j = P[knn[i][j]]       # shape (k,)
        
        for r in 0..k:
            neighbor_idx = nbrs_of_j[r]
            if pos[neighbor_idx] >= 0:
                col = pos[neighbor_idx]
                y[r][col] = P_of_j[r]
        
        # 计算: alpha_i_j = (P[i] + y_col) · mask / 2
        # 即对每一列，如果该列有非零值，取 (P[i][r] + y[r][col])/2 的和
        # 实际上原代码是:
        #   tmp_dist = (P[i] + y) * (y != 0)   # 逐元素: (P[i][r] + y[r][c]) if y[r][c]!=0 else 0
        #   tmp_dist = tmp_dist.sum(axis=-1) / 2  # 对列求和/2
        #   result[j] = tmp_dist[j]  # 取第j个位置的值，再1-x
        
        for r in 0..k:
            for c in 0..k:
                if y[r][c] != 0.0:
                    tmp_dist[r] += (P[i][r] + y[r][c])
        
        for r in 0..k:
            tmp_dist[r] /= 2.0
        
        nep_dists[i][j] = 1.0 - tmp_dist[j]  # 注意是取 j 位置
    
    # 清理 pos
    for j in 0..k:
        pos[knn[i][j]] = -1
```

### 3.3 NEP 算法的直观理解

1. **P[i]** 是节点 i 对其 k 个邻居的概率分布（越相似 / 距离越近，概率越高）
2. **y** 矩阵表示邻居 j 的概率分布 P[j] 在"与 i 共享的邻居"上的投影
3. **tmp_dist[r]** = (P[i][r] + y[r][:]) / 2 的平均值，表示节点 i 和节点 j 通过各自的第 r 个邻居形成的"二阶概率"
4. **1 - tmp_dist[j]** 表示 i 和其第 j 个邻居之间的 NEP 距离——越小越相似

### 3.4 输出

| 文件 | 键名 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| `knn_dists_trans2.npz` | `data` | `(N, k)` | float32 | NEP 距离，值域 [0, 1] |

### 3.5 并行化说明

- 外层节点循环 `i in 0..N` 完全可并行
- 每个线程需要独立的 `pos` 数组（或使用原子操作合理拍平）
- 推荐使用 `rayon` 并行迭代器，chunk_size = N / num_threads
- 原代码使用 64 个进程，Rust 中推荐使用 `num_cpus::get()` 个线程

### 3.6 时间/空间复杂度

- 时间: O(N · k²)，因为内层遍历每个邻居的 k 个邻居，寻找在 pos 中的位置
- 空间: O(N · k) 存储 P 矩阵和结果 + O(N) 的 pos 数组（每个线程一个）

---

## 4. 阶段三：FC-ES 聚类

### 4.1 输入

| 输入 | 来源 | 形状 | 类型 |
|------|------|------|------|
| nbrs | 阶段一输出 | `(N, k)` | int32 |
| nep_dists | 阶段二输出 | `(N, k)` | float32 |
| theta | 参数 | 标量 | float32, 默认 0.22 |

### 4.2 算法完整伪代码

```python
# ============================================================
# FC-ES: Face Clustering via Early Stopping
# 输入: nbrs[N][k], nep_dists[N][k], theta
# 输出: 每个节点的聚类标签 pred_labels[N]
# ============================================================

def fc_es_cluster(nbrs, nep_dists, theta):
    N = nbrs.shape[0]
    k = nbrs.shape[1]
    
    # ---- Step 1: 构建连接图 (get_links) ----
    # links: HashMap<(usize, usize), f32>  有向边 (src → dst) 与相似度
    # singles: Vec<usize>  没有连接到任何其他节点的孤立节点
    
    links = HashMap::new()
    singles = Vec::new()
    
    for i in 0..N:
        count = 0        # 节点i建立的连接数
        early_stop = false
        
        for j in 0..k:
            neighbor = nbrs[i][j]
            dist = nep_dists[i][j]
            
            # 跳过自环
            if i == neighbor:
                continue
            
            # 相似度转换: sim = 1 - dist
            sim = 1.0 - dist
            
            # 检查是否满足连接条件
            if sim >= theta:
                # 条件满足: 添加连接
                links.insert((i, neighbor), sim)
                count += 1
            else:
                # 触发 early stop 信号
                early_stop = True
            
            # FC-ES 核心: 一旦触发 early_stop, 立即 break
            if early_stop:
                break
        
        # 如果节点 i 没有连接到任何其他节点
        if count == 0:
            singles.push(i)
    
    # ---- Step 2: 社区发现 (InfoMap) ----
    # 用 InfoMap 对有向加权图做社区发现
    # 参数: "--two-level --directed"
    # 
    # InfoMap 等价操作:
    #   输入: 有向图 G=(V, E), 边权重 w(e) > 0
    #   输出: 每个节点的社区 (cluster) 标签
    
    # 构建 InfoMap 图
    graph = DirectedGraph::new()
    for ((src, dst), sim) in links:
        graph.add_edge(src, dst, weight=sim)
    
    # 运行 InfoMap
    clusters = infomap.run(graph, two_level=true, directed=true)
    # clusters: Vec<Vec<usize>>  每个 Vec 是一个聚类簇
    
    # ---- Step 3: 处理特殊节点 ----
    # InfoMap 的根节点 (moduleIndex == 0) 包含额外的哨兵元素
    # 需要剔除前2个元素（如果是根模块）或前1个元素（其他模块）
    
    label_to_nodes = HashMap::new()   # moduleIndex -> Vec<node_id>
    node_to_label = HashMap::new()    # node_id -> moduleIndex
    
    for node in infomap.tree:
        module = node.module_index
        node_to_label[node.id] = module
        label_to_nodes[module].push(node.id)
    
    # 剔除根模块的哨兵节点
    for (module_id, nodes) in label_to_nodes:
        if module_id == 0:
            label_to_nodes[module_id] = nodes[2..]   # 去掉前2个
        else:
            label_to_nodes[module_id] = nodes[1..]   # 去掉前1个
    
    # ---- Step 4: 合并孤立节点 ----
    # 每个孤立节点成为一个单独的簇
    next_label = label_to_nodes.len()
    for single_node in singles:
        node_to_label[single_node] = next_label
        label_to_nodes[next_label] = vec![single_node]
        next_label += 1
    
    # ---- Step 5: 生成最终标签 ----
    pred_labels = vec![0; N]  # 初始填充 0
    for (node_id, label) in node_to_label:
        pred_labels[node_id] = label
    
    return pred_labels, label_to_nodes
```

### 4.3 Early Stopping 决策流程图

```
遍历节点 i 的 KNN 邻居 j (按内积降序排列):

  ┌─→ sim(i, j) >= theta ?
  │      │
  │      YES ──→ 添加连接 links[(i,j)] = sim, count++
  │      │      继续下一个邻居
  │      │
  │      NO  ──→ early_stop = true ──→ break (FC-ES)
  │
  └── 下一个节点 i+1
```

**关键行为：**
- 邻居按内积降序排列，所以相似度是递减的
- 一旦遇到第一个 `sim < theta` 的邻居，立即停止（因为后面的只会更不相似）
- 这避免了检查所有 k 个邻居，显著加速

### 4.4 输出

```rust
struct ClusteringResult {
    /// 每个节点的簇标签, 长度 N
    pred_labels: Vec<u32>,
    /// 每个簇包含的节点列表
    clusters: Vec<Vec<usize>>,
    /// 孤立节点数
    num_singletons: usize,
    /// 簇总数
    num_clusters: usize,
    /// 总节点数
    num_nodes: usize,
}
```

### 4.5 与 Python 实现的对应关系

| Python 函数 | Rust 对应 | 行号 |
|------------|----------|------|
| `read_meta()` | `read_labels(path)` | clusters.py:22 |
| `intdict2ndarray()` | 不需要，用 Vec | clusters.py:15 |
| `get_links()` | `build_links()` | clusters.py:40 |
| `cluster_by_infomap()` | `fc_es_cluster()` | clusters.py:76 |
| `l2norm()` | `l2_normalize()` | clusters.py:10 |

---

## 5. 阶段四：评估指标

### 5.1 输入

- `gt_labels: &[u32]` — 真实标签，长度 N
- `pred_labels: &[u32]` — 聚类预测标签，长度 N

### 5.2 Pairwise (Fowlkes-Mallows) Score

```rust
/// 计算 Pairwise Precision, Recall, F-score
/// 
/// 算法:
///   1. 构建列联表 C (contingency matrix): C[i][j] = |gt_cluster_i ∩ pred_cluster_j|
///   2. tk = sum(C[i][j] * C[i][j]) - N     (正确配对数)
///   3. pk = sum(col_j_sum²) - N            (预测配对数)
///   4. qk = sum(row_i_sum²) - N            (真实配对数)
///   5. precision = tk / pk
///   6. recall = tk / qk
///   7. fscore = 2 * precision * recall / (precision + recall)
///
fn pairwise(gt_labels: &[u32], pred_labels: &[u32]) -> (f64, f64, f64) {
    let n = gt_labels.len();
    
    // 构建列联表 (用 HashMap 作为稀疏矩阵)
    let mut contingency: HashMap<(u32, u32), u32> = HashMap::new();
    let mut col_sum: HashMap<u32, u32> = HashMap::new();
    let mut row_sum: HashMap<u32, u32> = HashMap::new();
    
    for i in 0..n {
        let gt = gt_labels[i];
        let pred = pred_labels[i];
        *contingency.entry((gt, pred)).or_insert(0) += 1;
        *col_sum.entry(pred).or_insert(0) += 1;
        *row_sum.entry(gt).or_insert(0) += 1;
    }
    
    // tk = sum of squared intersection sizes - N
    let tk: f64 = contingency.values().map(|&v| (v as f64).powi(2)).sum::<f64>() - n as f64;
    
    // pk = sum of squared column sums - N
    let pk: f64 = col_sum.values().map(|&v| (v as f64).powi(2)).sum::<f64>() - n as f64;
    
    // qk = sum of squared row sums - N
    let qk: f64 = row_sum.values().map(|&v| (v as f64).powi(2)).sum::<f64>() - n as f64;
    
    let precision = if pk > 0.0 { tk / pk } else { 0.0 };
    let recall = if qk > 0.0 { tk / qk } else { 0.0 };
    let fscore = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    (precision, recall, fscore)
}
```

### 5.3 B-Cubed Score

```rust
/// 计算 B-Cubed Precision, Recall, F-score
///
/// 算法 (按真实类别迭代):
///   对每个真实类别 G:
///     G 中的元素被预测模型分配到了多个预测簇中
///     对于 G 在预测簇 P 中的交集:
///       N_intersection = |G ∩ P|
///       precision[G] += N_intersection² / |P|
///       recall[G]    += N_intersection² / |G|
///   avg_precision = sum(precision) / N
///   avg_recall = sum(recall) / N
///
fn bcubed(gt_labels: &[u32], pred_labels: &[u32]) -> (f64, f64, f64) {
    let n = gt_labels.len();
    
    // 构建索引: label -> Vec<node_id>
    let gt_map = group_by_label(gt_labels);
    let pred_map = group_by_label(pred_labels);
    
    let mut total_precision = 0.0f64;
    let mut total_recall = 0.0f64;
    
    for (gt_label, gt_nodes) in &gt_map {
        // 找出该真实类别中的元素被分配到了哪些预测簇
        let mut pred_labels_in_gt: HashSet<u32> = HashSet::new();
        for &node_id in gt_nodes {
            pred_labels_in_gt.insert(pred_labels[node_id]);
        }
        
        let gt_size = gt_nodes.len() as f64;
        
        for pred_label in pred_labels_in_gt {
            let pred_nodes = &pred_map[&pred_label];
            let pred_size = pred_nodes.len() as f64;
            
            // 交集大小
            let intersection = gt_nodes.iter()
                .filter(|&&id| pred_labels[id] == pred_label)
                .count() as f64;
            
            total_precision += intersection.powi(2) / pred_size;
            total_recall += intersection.powi(2) / gt_size;
        }
    }
    
    let avg_precision = total_precision / n as f64;
    let avg_recall = total_recall / n as f64;
    let fscore = if avg_precision + avg_recall > 0.0 {
        2.0 * avg_precision * avg_recall / (avg_precision + avg_recall)
    } else {
        0.0
    };
    
    (avg_precision, avg_recall, fscore)
}
```

### 5.4 NMI (Normalized Mutual Information)

```rust
/// 归一化互信息
/// NMI = 2 * I(GT; Pred) / (H(GT) + H(Pred))
/// 其中 I 是互信息, H 是信息熵
fn nmi(gt_labels: &[u32], pred_labels: &[u32]) -> f64 {
    let n = gt_labels.len() as f64;
    
    let gt_map = group_by_label(gt_labels);
    let pred_map = group_by_label(pred_labels);
    
    // 互信息 I(GT; Pred)
    let mut mi = 0.0f64;
    for (gt_label, gt_nodes) in &gt_map {
        let gt_prob = gt_nodes.len() as f64 / n;
        
        // 找出 gt_nodes 中的元素被分配到的预测簇
        let mut pred_in_gt: HashMap<u32, usize> = HashMap::new();
        for &node in gt_nodes {
            *pred_in_gt.entry(pred_labels[node]).or_insert(0) += 1;
        }
        
        for (&pred_label, &count) in &pred_in_gt {
            let pred_prob = pred_map[&pred_label].len() as f64 / n;
            let joint_prob = count as f64 / n;
            if joint_prob > 0.0 {
                mi += joint_prob * (joint_prob / (gt_prob * pred_prob)).ln();
            }
        }
    }
    
    // 熵 H(GT)
    let h_gt: f64 = gt_map.values()
        .map(|nodes| {
            let p = nodes.len() as f64 / n;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    
    // 熵 H(Pred)
    let h_pred: f64 = pred_map.values()
        .map(|nodes| {
            let p = nodes.len() as f64 / n;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    
    let denominator = h_gt + h_pred;
    if denominator > 0.0 {
        2.0 * mi / denominator
    } else {
        0.0
    }
}
```

---

## 6. 数据格式规范

### 6.1 特征文件 `.bin`

```
二进制格式: 连续 float32 值, little-endian
布局: [f[0][0], f[0][1], ..., f[0][255], f[1][0], ..., f[N-1][255]]
总字节数: N × 256 × 4
读取方式:
  let bytes = std::fs::read(path)?;
  let floats: &[f32] = bytemuck::cast_slice(&bytes); // 或手动 transmute
  // reshape 为 N×256 矩阵
```

### 6.2 标签文件 `.meta`

```
文本格式, UTF-8
每行: "整数\n"
示例:
  0
  0
  1
  1
  2
行数 = N
```

### 6.3 中间文件 (自定义二进制格式建议)

不依赖 `.npz`，直接用自定义布局：

```
文件头 (16 bytes):
  magic:      [u8; 4]  = b"KNN\0"
  version:    u32      = 1
  num_rows:   u32      = N
  num_cols:   u32      = k

数据体:
  N × k 个连续的元素，类型由文件名决定:
  - knn_nbrs: u32 (4 bytes each), 总大小 N × k × 4
  - knn_dists: f32 (4 bytes each), 总大小 N × k × 4
  - knn_dists_trans2: f32 (4 bytes each), 总大小 N × k × 4
```

或者直接使用 `.npz` 通过 `npyz` crate 读取。

### 6.4 输出格式

聚类结果写入文本文件：

```
每行: "node_id cluster_label\n"
示例:
  0 42
  1 42
  2 17
  ...
```

---

## 7. Rust 实现建议

### 7.1 推荐 Crate

| 用途 | Crate | 说明 |
|------|-------|------|
| 矩阵运算 | `nalgebra` 或 `ndarray` | N×D 和 N×k 矩阵操作 |
| 并行计算 | `rayon` | NEP 阶段和 KNN 搜索的主循环并行 |
| KNN 搜索 | `faiss` (binding) 或手写 SIMD | GPU 加速或 CPU SIMD 内积 |
| 近似 KNN | `hnsw_rs` 或 `annoy-rs` | 大规模数据的近似邻居搜索 |
| InfoMap | 自行移植或 FFI 调用 C++ 库 | InfoMap 核心是随机游走 + 霍夫曼编码 |
| NPZ 读写 | `npyz` | 读/写 `.npz` 格式 |
| 内存映射 | `memmap2` | 大文件零拷贝读取 |
| SIMD | `std::arch` 或 `wide` | 内积运算加速 |
| CLI | `clap` | 命令行参数解析 |
| 日志 | `log` + `env_logger` | 进度输出 |
| 序列化 | `serde` + `bincode` | 中间结果持久化 |

### 7.2 项目结构建议

```
fc-es/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI 入口
│   ├── knn.rs               # KNN 图构建 (阶段一)
│   ├── nep.rs               # NEP 二阶距离 (阶段二)
│   ├── clustering.rs        # FC-ES 聚类 (阶段三)
│   ├── evaluation.rs        # 评估指标 (阶段四)
│   ├── infomap.rs           # InfoMap 社区发现 (移植或FFI)
│   ├── io.rs                # 文件读写 (.bin, .meta, 中间格式)
│   ├── types.rs             # 核心数据结构
│   └── math.rs              # L2归一化, 距离计算, softmax 等
└── tests/
    ├── test_knn.rs
    ├── test_nep.rs
    ├── test_clustering.rs
    └── test_evaluation.rs
```

### 7.3 关键数据结构

```rust
/// KNN 图
struct KnnGraph {
    nbrs: Array2<u32>,      // (N, k) 邻居索引
    dists: Array2<f32>,     // (N, k) 内积值
}

/// NEP 距离
struct NepDists {
    data: Array2<f32>,      // (N, k) NEP距离, 值域[0,1]
}

/// 聚类结果
struct ClusteringResult {
    labels: Vec<u32>,       // 长度N, 每个节点的簇标签
    num_clusters: usize,
    num_singletons: usize,
}

/// 连接边
struct Edge {
    src: usize,
    dst: usize,
    similarity: f32,
}
```

### 7.4 性能关键路径

1. **KNN 内积搜索** (阶段一): O(N²D)，是主要瓶颈。Rust 实现建议：
   - 使用 SIMD (AVX2/AVX-512) 做批量内积
   - 使用 `rayon` 做外循环并行
   - N > 100K 时考虑 HNSW 近似搜索
   - 使用 `half` crate (f16) 减少内存带宽

2. **NEP 二阶距离** (阶段二): O(Nk²)，k=80 时每个节点约 6400 次操作：
   - 使用 `rayon` 并行外层节点循环
   - 内层 `pos` 查找是关键：k 不大 (80)，线性扫描即可
   - 预分配每线程的临时缓冲区避免重复分配

3. **边构建** (阶段三): O(Nk)，直接遍历：
   - 串行即可，不需要并行
   - `HashMap` 预分配容量

### 7.5 内存优化建议

```rust
// 阶段一: KNN 使用 chunked 处理避免 O(N²) 全内存
// 将 target 分块加载，每块与 query 计算内积:
for chunk_start in (0..N).step_by(chunk_size) {
    let target_chunk = load_chunk(chunk_start, chunk_size);
    let inner_products = query @ target_chunk.t();  // (N, chunk_size)
    // 更新每行的 top-k
    update_topk(&mut nbrs, &mut dists, &inner_products, chunk_start);
}
```

### 7.6 InfoMap 替代方案

由于 InfoMap C++ 库的移植成本较高，可考虑以下替代：

| 方案 | 说明 |
|------|------|
| **FFI 调用 libInfomap** | 通过 `cc` crate 编译 C++ 源码，FFI 调用 |
| **自实现简版** | 基于 Louvain 或 Label Propagation 的社区发现 |
| **CLI 调用** | 生成边列表文件，调用系统 `Infomap` 命令行工具 |

推荐 CLI 调用方案（最简）：
```rust
// 生成边列表文件 (src dst weight)
// 调用: Infomap edge_list.txt output_dir --two-level --directed
// 解析输出的 .tree 文件
```

---

## 8. 测试验证方案

### 8.1 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = ndarray::arr2(&[[3.0, 4.0], [0.0, 5.0]]);
        let normed = l2_normalize(&v);
        assert!((normed[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((normed[[0, 1]] - 0.8).abs() < 1e-6);
        assert!((normed[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_perfect() {
        let gt = vec![0, 0, 1, 1, 2, 2];
        let pred = vec![0, 0, 1, 1, 2, 2];
        let (p, r, f) = pairwise(&gt, &pred);
        assert!((p - 1.0).abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((f - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nmi_perfect() {
        let gt = vec![0, 0, 1, 1];
        let pred = vec![0, 0, 1, 1];
        let score = nmi(&gt, &pred);
        assert!((score - 1.0).abs() < 1e-6);
    }
}
```

### 8.2 集成测试 - 小规模端到端验证

```rust
#[test]
fn test_fc_es_tiny_dataset() {
    // 构造一个简单数据集:
    // 4个节点，2个类别: [0,0,1,1]
    // 特征: 使得同类内积高，异类内积低
    // 预期: FC-ES 正确将 4 个节点分为 2 类
    
    let features = ndarray::arr2(&[
        [1.0, 0.0], [0.99, 0.0],  // 类别0 (几乎相同的向量)
        [0.0, 1.0], [0.0, 0.99],  // 类别1
    ]);
    let gt = vec![0, 0, 1, 1];
    
    // 归一化
    let features = l2_normalize(&features);
    
    // KNN (k=2)
    let (nbrs, dists) = knn_inner_product(&features, 2);
    
    // NEP 距离
    let nep_dists = compute_nep(&nbrs, &dists);
    
    // FC-ES 聚类
    let result = fc_es_cluster(&nbrs, &nep_dists, 0.22);
    
    // 验证
    let (p, r, f) = pairwise(&gt, &result.labels);
    assert!(f > 0.9, "F-score should be high, got {}", f);
}
```

### 8.3 与 Python 参考实现对拍

1. 在 Python 中运行完整流程，保存各阶段的中间结果
2. Rust 实现读取相同的输入，输出中间结果
3. 逐阶段对比数值误差（允许浮点误差 < 1e-6）
4. 对比最终聚类评估指标

```bash
# Python 端: 导出中间结果供 Rust 对比
python -c "
import numpy as np
# 导出 NEP 结果
d = np.load('data/knns/part1_test/knn_dists_trans2.npz')
d['data'].astype(np.float32).tofile('nep_dists_py.bin')
# 导出聚类标签
...
"

# Rust 端: 对比
cargo run -- compare \
  --nep-rust nep_dists_rs.bin \
  --nep-python nep_dists_py.bin \
  --tolerance 1e-5
```

### 8.4 精度验证检查点

| 阶段 | 验证内容 | 方法 |
|------|---------|------|
| KNN | nbrs 和 dists 与 Python 完全一致 | 逐元素对比，允许顺序差异 |
| NEP | NEP 距离与 Python 误差 < 1e-5 | 逐元素 max_abs_diff |
| 聚类 | pairwise/bcubed/nmi 与 Python 误差 < 0.001 | 直接对比评估指标 |
| 端到端 | 在 MS1M part1_test 上 F-score ≥ 0.875 | 与论文结果对比 |

### 8.5 最小可验证示例

```rust
// tests/integration_test.rs
// 使用人工构造的 10 节点 × 3 类的微型数据集验证完整管线

#[test]
fn test_mini_pipeline() {
    // 3 类: 类别0有4个节点, 类别1有3个, 类别2有3个
    // 特征设计使类内距离 << 类间距离
    let gt = vec![0,0,0,0, 1,1,1, 2,2,2];
    let features = generate_clustered_features(&gt, dim=256, noise=0.01);
    
    // 运行完整管线
    let result = run_fc_es_pipeline(&features, k=5, theta=0.22);
    
    // 验证
    let (_, _, f) = pairwise(&gt, &result.labels);
    assert!(f > 0.95, "Expected F > 0.95, got {:.4}", f);
    
    let nmi_score = nmi(&gt, &result.labels);
    assert!(nmi_score > 0.9, "Expected NMI > 0.9, got {:.4}", nmi_score);
}
```

---

## 附录 A: 核心参数表

| 参数 | 符号 | 默认值 | 作用 | 调参方向 |
|------|------|--------|------|----------|
| k | k | 80 | KNN 邻居数 | 越大越精确，越慢 |
| sigma | σ | 0.5 | NEP softmax 温度 | 越小分布越尖锐 |
| theta | θ | 0.22 | Early Stopping 阈值 | 越大聚类越保守，越小越激进 |

## 附录 B: 与原论文的对应关系

| 论文概念 | Python 实现 | Rust 实现 |
|---------|-----------|-----------|
| KNN Graph Construction | `knn/knn.py`, `knn/faiss_gpu.py` | `knn.rs` |
| Neighbor-based Edge Probability | `knn/nep_distance2.py` | `nep.rs` |
| Early Stopping | `clusters.py:get_links()` lines 49-52, 60-69 | `clustering.rs:build_links()` |
| Graph-based Label Propagation (InfoMap) | `clusters.py:cluster_by_infomap()` lines 82-97 | `clustering.rs:community_detection()` |
| Pairwise F-measure | `metrics.py:fowlkes_mallows_score()` | `evaluation.rs:pairwise()` |
| B-Cubed F-measure | `metrics.py:bcubed()` | `evaluation.rs:bcubed()` |
| NMI | `metrics.py:nmi()` | `evaluation.rs:nmi()` |
