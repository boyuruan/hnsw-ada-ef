# MS MARCO 18M: HNSW baseline ef 扫描 与 Ada-ef 结果

## 配置

- **数据**：`/data/gv/csv/msmarco_embeddings[18000000].fvecs` — 18,000,000 × 768 维（Cohere MS MARCO V2.1 embedding，L2 度量）
- **查询**：`/data/gv/csv/queries.fvecs`（共 100 条）**前 50 条**，768 维
- **GT**：`/data/gv/csv/top100000_[18m]_results_csv.ivecs` — 50 行 × 100,000 个 int32（每行截取前 k 个用于 recall@k）
- **索引**：`/data/byr/hnsw-index/msmarco_embeddings[18000000].hnsw`
  - 构建：M=16, ef_construction=500, threads=12
  - 构建耗时：6,965,080 ms（约 1.9 小时）
  - 构建自检 recall（searchKnn(1) 命中自身）：前 1000 点 0.423，后 1000 点 0.895
- **度量**：L2；时延为中位值（repeat=3），单位为整批 50 条查询的总耗时

## 1. Baseline：固定 ef 扫描（recall vs latency）

命令：`run_fvecs sweep`（ef 1000..20000，step 1000，repeat 3，50 queries）

| ef | time_ms | per-query(ms) | recall@10 | recall@100 | recall@1000 |
|-----|---------|---------------|-----------|------------|-------------|
| 1000 | 359 | 7.2 | 0.766 | 0.900 | 0.790 |
| 2000 | 696 | 13.9 | 0.786 | 0.919 | 0.845 |
| 3000 | 980 | 19.6 | 0.786 | 0.925 | 0.866 |
| 4000 | 1273 | 25.5 | 0.806 | 0.932 | 0.884 |
| 5000 | 1567 | 31.3 | 0.830 | 0.938 | 0.893 |
| 6000 | 1873 | 37.5 | 0.836 | 0.941 | 0.899 |
| 7000 | 2166 | 43.3 | 0.856 | 0.945 | 0.904 |
| 8000 | 2461 | 49.2 | 0.866 | 0.948 | 0.907 |
| 9000 | 2746 | 54.9 | 0.870 | 0.950 | 0.910 |
| 10000 | 3045 | 60.9 | 0.912 | 0.955 | 0.912 |
| 11000 | 3321 | 66.4 | 0.912 | 0.956 | 0.914 |
| 12000 | 3593 | 71.9 | 0.912 | 0.957 | 0.915 |
| 13000 | 3878 | 77.6 | 0.914 | 0.958 | 0.917 |
| 14000 | 4165 | 83.3 | 0.934 | 0.960 | 0.919 |
| 15000 | 4444 | 88.9 | 0.934 | 0.961 | 0.920 |
| 16000 | 4716 | 94.3 | 0.954 | 0.963 | 0.921 |
| 17000 | 4999 | 100.0 | 0.954 | 0.964 | 0.922 |
| 18000 | 5267 | 105.3 | 0.954 | 0.964 | 0.923 |
| 19000 | 5548 | 111.0 | 0.954 | 0.964 | 0.924 |
| 20000 | 5795 | 115.9 | 0.954 | 0.965 | 0.925 |

观察：

- 延迟与 ef 近似线性（约 0.29 ms/ef-unit/query）。
- recall@10 在 ef≈10000 后趋于饱和（0.91 → 0.95）；recall@100 接近饱和（0.965）。
- recall@1000 到 ef=20000 仍在缓慢爬升（0.925），饱和需要更高 ef。
- 绝对召回率偏低，与该索引构建质量有关（M=16/efc=500/12 线程，构建自检 recall 0.895）。

## 2. Ada-ef：target recall 0.95

离线阶段：`run_fvecs offline`（ks={10,100,1000}，target 0.95，ef 上限 100000，sampling 200）
在线阶段：`run_fvecs ada`（dataset msmarco18m，50 queries，repeat 3）

| k | 估计 ef (wae) | 中位时延(50q) | per-query(ms) | 真实 recall | 5th pct | 1st pct | recall≥0.95 的 query |
|----|--------------|---------------|---------------|-------------|---------|---------|---------------------|
| 10 | 25 | 35 ms | 0.7 | 0.592 | 0.000 | 0.000 | 22/50 |
| 100 | 239 | 89 ms | 1.8 | 0.824 | 0.600 | 0.020 | 14/50 |
| 1000 | 19392 | 7534 ms | 150.7 | 0.929 | 0.703 | 0.610 | 22/50 |

**三个 k 均未达到 0.95 目标**。k=10/100 严重低估所需 ef：

- k=10：估计 ef=25，真实 recall 仅 0.592；而 baseline 在 ef=1000 时 recall@10 已有 0.766。
- k=100：估计 ef=239，真实 recall 0.824；baseline 在 ef=1000 时 recall@100 已有 0.900。
- k=1000：估计 ef=19392，真实 recall 0.929，接近但未达目标（baseline 在 ef=20000 时为 0.925，与 wae≈19392 基本一致，说明该档估计本身合理，只是目标没够到）。

## 3. 输出异常

### 3.1 offline 阶段（`run_offline_ada`，ef_adaptor 构建）

- **负的 "Initial average recall"**（k=10）：
  ```
  Initial average recall with ef=10: -4.71428
  Initial average recall with ef=15: -4.71328
  ```
  recall 为负值说明 `RecallEstimator::compute_average_recall` 计算的不是标准 recall@k（疑似基于 sketch/score 的代理指标，且存在异常）。

- **大于 1 的 "Initial average recall"**（k=100）：
  ```
  Initial average recall with ef=100: 1.20055
  Initial average recall with ef=150: 1.2395
  ```
  recall > 1，同样不符合标准 recall 语义。

- **"Recall diff is too small, break."** 大量出现：k=10 约 63 次、k=100 约 18 次、k=1000 约 5 次。ef 估计循环几乎未迭代就中断，导致 wae 停留在很低的值（k=10 时 ef 从 10/15 起几乎没涨）。

- 上述异常共同导致 offline 的 ef→recall 估计与 online 真实 GT recall 严重偏离（k=10 估计 ef=25 即可达 0.95，实测仅 0.592）。

### 3.2 online 阶段（`run_fvecs ada`）

- k=10 时 recall 分布尾部很差：5th percentile = 0，1st percentile = 0（50 条查询中有相当一部分 recall 为 0）。
- k=100 时 1st percentile = 0.02，5th percentile = 0.60，分布同样偏斜。

## 复现命令

```bash
# baseline sweep
./build/run_fvecs sweep \
  --query /data/byr/hnsw-index/queries50.fvecs \
  --neighbors "/data/gv/csv/top100000_[18m]_results_csv.ivecs" \
  --index "/data/byr/hnsw-index/msmarco_embeddings[18000000].hnsw"

# offline（生成 estimator / ef_adaptor）
EXPERIMENTS_ROOT=/data/byr/hnsw-index/experiments ./build/run_fvecs offline \
  --base "/data/gv/csv/msmarco_embeddings[18000000].fvecs" \
  --index "/data/byr/hnsw-index/msmarco_embeddings[18000000].hnsw" \
  --dataset msmarco18m --k 10 --k 100 --k 1000 \
  --expected-recall 0.95 --ef-upper 100000 --sampling-size 200

# online ada（每个 k 一次）
EXPERIMENTS_ROOT=/data/byr/hnsw-index/experiments ./build/run_fvecs ada \
  --query /data/byr/hnsw-index/queries50.fvecs \
  --neighbors "/data/gv/csv/top100000_[18m]_results_csv.ivecs" \
  --index "/data/byr/hnsw-index/msmarco_embeddings[18000000].hnsw" \
  --dataset msmarco18m --k 10 --expected-recall 0.95 --repeat 3
```

中间产物（experiments 目录）：`/data/byr/hnsw-index/experiments/{statistics,estimation_table,sampling}/`
