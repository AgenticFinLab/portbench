# 资产相关性 — PortBench 核心创新点

## 设计动机

现有金融基准（FinBen、InvestorBench、QuantBench）的评估单元是单资产买卖决策，天然无法衡量模型对**跨资产关联结构**的理解能力。PortBench 将相关性提升为一等公民：它贯穿数据生产、快照构建、提示注入、S3 评分、基线策略和可视化全链路，是区分"会做投资"与"只会答题"的关键维度。

两层相关性框架：
- **类内（intra-class）**：同一资产类别（如权益类）内各 ticker 之间的相关性 → 集中度惩罚
- **类间（inter-class）**：跨资产类别（如权益 vs. 债券）之间的相关性 → 对冲效果奖励

---

## 全链路数据流

```
预处理
  _write_correlation_artifacts()
       │  correlation_matrix.csv
       │  covariance_matrix.csv
       │  asset_class_map.json
       ▼
数据质量校验
  CrossAssetQualityChecker._check_correlation_structure()
       │
       ▼
Sandbox 运行时
  SnapshotBuilder.build()  →  MarketSnapshot
       │  .correlation_matrix        (实时，按回望窗口重算)
       │  .asset_class_map
       │  .get_intra_class_correlation()
       │  .get_inter_class_correlation()
       ▼
评估流水线
  S3 提示注入  ←  _format_correlation()
  S3 评分       ←  _intra_class_diversification_score()
                    _inter_class_hedging_score()
                    _correlation_awareness_score()  (fallback)
       ▼
基线策略
  CovarianceRiskParityBaseline  ←  ERC (协方差矩阵)
       ▼
可视化 & 实验报告
  correlation_plots.py / correlation_graph.py
  experiments/figures.py
```

---

## 1. 相关性计算

### 1.1 预处理阶段（冻结矩阵）

**文件：** `examples/data_preprocess/preprocess_all.py` → `_write_correlation_artifacts()`

计算逻辑：

1. 筛选价格列：列名含 `close`、`price`、`value`、`adj` 的列视为价格序列
2. 转换为**简单日收益率**：`prices.pct_change()`
3. 要求至少 30 个有效数据点，否则跳过该 ticker
4. 计算 **Pearson 相关矩阵**：`ret_df.corr()`（pandas 默认，pairwise complete observations）
5. 计算**年化协方差矩阵**：`ret_df.cov() * 252`

输出（写入 `datasets/processed/`）：

| 文件 | 内容 |
|------|------|
| `correlation_matrix.csv` | N×N Pearson 相关矩阵（所有资产） |
| `covariance_matrix.csv` | N×N 年化协方差矩阵 |
| `asset_class_map.json` | `{ticker: asset_class}` 映射 |

> **PiT 安全性**：该矩阵在预处理阶段一次性冻结，下游读取时不会重算，避免未来信息泄漏。

### 1.2 运行时（动态矩阵）

**文件：** `portbench/sandbox/snapshot_builder.py` → `SnapshotBuilder.build()`

在每个再平衡日，用回望窗口（默认 60 天）的历史收益率实时重算相关矩阵：

```python
ret_df = pd.DataFrame(return_data)   # 窗口内日收益率
corr = ret_df.corr()                 # Pearson，pairwise
```

结果注入 `MarketSnapshot.correlation_matrix`，供当期决策使用。

---

## 2. MarketSnapshot 相关性接口

**文件：** `portbench/agent_eval/base.py`

`MarketSnapshot` 暴露三个相关性方法，均延迟计算并缓存：

### `get_correlation()`
返回完整 N×N Pearson 相关矩阵。若尚未缓存，从 `return_data` 构建并缓存。

### `get_intra_class_correlation()`
返回 `dict[str, pd.DataFrame]`，每个资产类别 → 该类内 ticker 的相关矩阵子块。
需要 `asset_class_map` 已设置，且该类至少有 2 个成员。

### `get_inter_class_correlation()`
返回类别×类别的 DataFrame，每格为两类之间所有 ticker 对的平均 Pearson 相关值。
对角线为类内平均（排除自相关）。

---

## 3. S3 权重优化评分

**文件：** `portbench/agent_eval/stages.py`

S3 总分组成：

```
S3 score = 70% × weight_accuracy
         + 30% × correlation_awareness
```

`correlation_awareness` 有两条路径：

```
有 asset_class_map:
    30% = 15% × intra_class_score + 15% × inter_class_score

无 asset_class_map（fallback）:
    30% = variance_ratio_score
```

### 3.1 `_intra_class_diversification_score()` — 类内集中度惩罚

```python
penalty = Σ_class  class_weight × max(avg_intra_corr, 0)
score   = clip(1.0 - penalty, 0, 1)
```

- `class_weight`：该类所有成员权重之和
- `avg_intra_corr`：该类内部所有 ticker 对的平均相关系数（排除对角线）
- 含义：若把大量资金集中在高度相关的同类资产中，惩罚高；分散则惩罚低

### 3.2 `_inter_class_hedging_score()` — 类间对冲奖励

```python
avg_xclass_corr = Σ_{i≠j} w_i × w_j × corr(i,j)
                / Σ_{i≠j} w_i × w_j

score = clip((1.0 - avg_xclass_corr) / 2.0, 0, 1)
```

| 情形 | avg_xclass_corr | score |
|------|-----------------|-------|
| 完全正相关 (+1) | +1 | 0 |
| 不相关 (0) | 0 | 0.5 |
| 完全负相关 (−1) | −1 | 1.0 |

- 含义：权重越向负相关资产类别分散，得分越高（体现对冲价值）

### 3.3 `_correlation_awareness_score()` — 方差比率（fallback）

```python
var_actual = w_actual @ cov @ w_actual
var_gt     = w_gt @ cov @ w_gt
score      = clip(2.0 - var_actual / var_gt, 0, 1)
```

当没有 `asset_class_map` 时使用，奖励构建出比基准方差更低的组合。

---

## 4. S1/S3 提示注入

**文件：** `portbench/agent_eval/stages.py` → `_format_correlation()`

在注入给 LLM 的市场快照提示中，相关性信息以三部分呈现：

1. **资产间相关矩阵表**（最多 `max_assets=6` 个资产）
2. **类内平均相关**（每个资产类别一行，来自 `_intra_class_avgs()`）
3. **类间相关矩阵**（资产类别×资产类别，来自 `_inter_class_matrix()`）

LLM 工具调用模式下，还可主动调用 `correlation` 工具计算任意两个收益率序列的 Pearson 相关系数（`portbench/agent_eval/tools.py`）。

---

## 5. 基线策略：协方差风险平价

**文件：** `portbench/baselines/covariance_risk_parity.py` → `CovarianceRiskParityBaseline`

与朴素 `RiskParityBaseline`（仅用波动率倒数）的区别：`CovarianceRiskParityBaseline` 使用完整协方差矩阵，求解 **等风险贡献（ERC）** 优化：

```
目标：令每个资产的边际风险贡献相等
      RC_i = w_i × (Σw)_i = 1/N × w^T Σ w
```

使用 Spinu（2013）坐标下降法求解，逐资产更新：

```python
# 更新第 i 个资产权重
a_i = cov[i, i]
b_i = (cov[i, :] @ w) - cov[i, i] * w[i]
w_i_new = (-b_i + sqrt(b_i² + 4 × a_i × target_rc)) / (2 × a_i)
```

加入 ridge 正则化保证数值稳定性：`cov += λ × mean(diag) × I`

---

## 6. 数据质量校验

**文件：** `portbench/data_quality/cross_asset_quality.py` → `CrossAssetQualityChecker._check_correlation_structure()`

校验逻辑：
1. 每个资产类别取中位数价格序列 → 转日收益率
2. 构建类级别相关矩阵（6×6）
3. 计算非对角线 NaN 比例

| 结果 | 条件 |
|------|------|
| FAIL | 所有非对角线均为 NaN |
| WARN | NaN 比例 > 50% |
| PASS | NaN 比例 ≤ 50% |

`details` 字段返回：`n_classes`、`nan_ratio`、`mean_off_diag`、`min_off_diag`、`max_off_diag`

---

## 7. 可视化

### 7.1 矩阵视图（`correlation_plots.py`）

| 函数 | 用途 |
|------|------|
| `plot_correlation_heatmap(corr, asset_class_map)` | 资产×资产热力图，同类资产相邻排列，用色块标注类别边界 |
| `plot_inter_class_correlation(corr, asset_class_map)` | 两图联排：左=类间相关热力图，右=各类类内平均相关柱状图 |
| `plot_correlation_evolution(snapshot_dir, asset_class_map, window=6)` | 跨回测快照的滚动相关演化曲线（总体 + 分类别） |
| `load_processed_correlation(processed_dir)` | 从 `datasets/processed/` 加载冻结矩阵 |

**内部辅助函数：**
- `_group_by_class()` — 按资产类别重排矩阵行列，返回类别边界坐标
- `_intra_class_avgs()` — 各类类内非对角均值（对应 S3 集中度惩罚项）
- `_inter_class_matrix()` — 类别×类别平均相关（对应 S3 对冲奖励项）
- `_load_snapshot_returns()` — 从 BacktestEngine JSON 快照加载历史收益率

### 7.2 网络视图（`correlation_graph.py`）

| 函数 | 用途 |
|------|------|
| `plot_correlation_mst(corr, asset_class_map)` | Mantegna 最小生成树，揭示跨资产关联骨架 |
| `plot_correlation_threshold(corr, asset_class_map, threshold=0.5)` | 阈值过滤力导图，仅保留 `|ρ| ≥ threshold` 的边 |

**Mantegna 距离：** $d_{ij} = \sqrt{2(1 - \rho_{ij})}$，满足超度量不等式，构成树结构的理论基础。

边着色规则：正相关 → 红色，负相关 → 蓝色；边宽正比于 $|\rho|$；节点颜色 = 资产类别；节点大小 = 图内度数。

### 7.3 实验流水线图表（`experiments/figures.py`）

| 调用 | 时机 | 输出文件 |
|------|------|----------|
| `render_dataset_correlation_figures()` | 每个 batch 一次 | `correlation_heatmap.png`, `inter_class_correlation.png` |
| `render_experiment_figures()` → `plot_correlation_evolution()` | 每个 (model, profile) 结束后 | `correlation_evolution_{phase}.png` |

---

## 8. 关键设计决策

| 决策 | 说明 |
|------|------|
| Pearson 而非 Spearman | 与现有金融文献一致；日频收益率分布接近正态时偏差可忽略 |
| 简单收益率而非对数收益率 | 组合加权可加性；日频下两者差异 < 0.1% |
| 冻结矩阵 + 运行时矩阵双轨 | 冻结矩阵保证 PiT 完整性，运行时矩阵捕捉相关性的时变动态 |
| pairwise complete observations | 不同资产缺失日期不同，比 listwise 删除保留更多有效观测 |
| Ridge 正则化（ERC 基线） | 防止协方差矩阵在样本量不足时奇异，提升数值稳定性 |
| 类内 + 类间双层框架 | 单层相关矩阵混淆了"同类集中"与"跨类分散"两个独立维度 |
