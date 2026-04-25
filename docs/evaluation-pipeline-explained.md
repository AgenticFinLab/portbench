# PortBench 评测机制详解

## 修改后的总览

层级 1：QA 静态问答
  输入：qa_dataset/test.jsonl
  输出：准确率（按模板 T1-T7 分项）
  脚本：examples/agent_eval/run_qa_eval.py

层级 2：Sandbox × 3 Profiles × (正常 + 压力) 
  ┌─ 压力阶段（风险门槛）
  │   对每种画像分别跑三个危机窗口
  │   按画像阈值判断通过/失败
  │   通过 → 进入正常阶段
  │
  └─ 正常阶段（性能排名）
      对每种画像跑 2024 全年
      输出：CEPS（每步副产品）+ 画像对齐分 + PnL
      三种画像汇总 → adaptation score


---

本文档拆解 PortBench 的两种核心评测机制：**CEPS 静态流水线**和 **Sandbox 回测**，以及它们的输入输出、五个决策阶段的具体含义。

## 唯一的外部输入：MarketSnapshot

无论哪种评测机制，每一步的原始输入都是同一个数据结构 `MarketSnapshot`：

| 字段 | 内容 | 示例 |
|------|------|------|
| `decision_date` | 做决策的日期 | `2024-02-01` |
| `price_data` | 每个资产最近 N 天的**收盘价序列** | `{"SPY": [470, 473, 476, ...]}` |
| `return_data` | 每个资产最近 N 天的**日收益率序列** | `{"SPY": [0.003, -0.001, ...]}` |
| `macro_data` | 宏观指标标量 | `{"fed_rate": 5.33, "vix": 13.88}` |
| `news_text` | 截至该日期的 SEC 公告 / 新闻文本 | `"AAPL Q4 earnings beat..."` |
| `current_weights` | 当前实际持仓权重 | `{"SPY": 0.30, "BTC": 0.10, ...}` |
| `portfolio_value` | 当前组合净值（美元） | `1_037_094.0` |
| `correlation_matrix` | 资产间相关系数矩阵（惰性计算） | 18×18 DataFrame |

---

## 五个决策阶段

### S1 — 市场解读（调用 LLM）

**输入**：`MarketSnapshot`（价格、收益率、宏观指标、新闻文本）

**输出**：
```
asset_views: dict[str, float]   # 每个资产一个 [-1, +1] 情绪分
detected_regime: str            # "bull" / "bear" / "sideways"
confidence: float               # 整体置信度 [0, 1]
```

**LLM 在做什么**：读历史价格走势和新闻，给每个资产打"看多/看空"分数。+1 表示极度看多，-1 表示极度看空。

**标准答案怎么来**：
```
view = clip(过去21天涨幅 / 10%, -1, 1)
```
涨了 10% → view = +1.0；跌了 10% → view = -1.0。

**打分方式**：`score = 1 - MAE(模型输出, 标准答案) / 2`

---

### S2 — 信号生成（调用 LLM）

**输入**：`S1Output`（各资产情绪分）+ `news_text`

**输出**：
```
signals: dict[str, "buy"|"hold"|"sell"]   # 每个资产的交易方向
strengths: dict[str, float]               # 信号强度 [0, 1]
```

**LLM 在做什么**：把连续的情绪分（-1 到 +1）转成离散的交易指令（买/持有/卖）。当有新闻时，模型可以覆盖纯数值判断——例如数值看空但财报超预期则改为 buy。

**标准答案怎么来**：
```
view > 0.2  → buy
view < -0.2 → sell
否则        → hold
```

**打分方式**：`score = 方向正确的资产数 / 总资产数`

---

### S3 — 权重优化（调用 LLM）

**输入**：`S2Output`（buy/hold/sell 信号列表）

**输出**：
```
weights: dict[str, float]   # 各资产仓位，必须满足 sum = 1.0，每项 ∈ [0, 1]
```

**LLM 在做什么**：根据买卖信号决定资金如何分配，类似做马科维茨组合优化，但由 LLM 直接输出权重。

**标准答案怎么来**：买入信号资产等权分配，卖出信号资产权重为 0。

**打分方式**：`score = 70% × 权重精度(MAE) + 30% × 相关性意识`
- 相关性意识：模型分配的权重是否避免了把资金堆在高相关资产上（即是否有效分散）。

---

### S4 — 执行模拟（**纯确定性，不调用 LLM**）

**输入**：`S3Output`（目标权重）+ 当前价格

**输出**：
```
executed_weights: dict[str, float]   # 实际执行后的权重（略低于目标）
total_cost: float                    # 交易总成本（美元）
orders: list[TradeOrder]             # 每笔交易明细
```

**在做什么**：模拟真实下单产生的摩擦成本，纯数值计算：
- **滑点**：10 bps（市场冲击，大单推高/压低价格）
- **佣金**：5 bps（每笔交易额的固定费率）
- 最终持仓 = 目标权重 − 成本拖累

S4 不需要判断，成本是机械的。

---

### S5 — 风险监控（**纯确定性，不调用 LLM**）

**输入**：`S4Output`（实际执行后的权重）+ `return_data`

**输出**：
```
portfolio_var: float        # 1日 VaR（95% 置信度）
portfolio_drawdown: float   # 当前最大回撤
weight_drift: float         # 持仓偏离目标权重的程度
rebalance_needed: bool      # 是否触发强制再平衡
alerts: list[RiskAlert]     # 触发的风险预警列表
```

**在做什么**：数值计算风险指标，判断是否需要被动再平衡：
- VaR 用**历史模拟法**（用过去收益率分布估计尾部风险）
- 超过阈值触发 warning 或 critical 预警

---

## CEPS 是什么

五个阶段各产生一个 [0, 1] 分数，CEPS 把它们聚合成一个指标：

```
isolated_avg      = mean(s1, s2, s3, s4, s5)
cascade_drops     = Σ max(s[i] - s[i+1], 0)   # 相邻阶段分数下降的总量
propagation_penalty = 0.1 × cascade_drops
CEPS = clip(isolated_avg - propagation_penalty, 0, 1)
```

**为什么有级联惩罚**：如果 S1 严重出错，错误信号会流入 S2、S3……越传越偏。CEPS 惩罚这种"前面烂掉后面全崩"的情况，而不是简单平均。

举例：
- 模型 A：[1.0, 1.0, 0.3, 0.3, 0.3] → isolated_avg=0.58，cascade_drops=0.7，CEPS=**0.51**
- 模型 B：[0.7, 0.7, 0.7, 0.7, 0.7] → isolated_avg=0.70，cascade_drops=0，CEPS=**0.70**

模型 A 前两步很好但中途崩了，CEPS 低于均匀表现的模型 B。

---

## 两种评测机制对比

### CEPS 静态流水线

```
snapshot_1 → S1→S5 → CEPS分_1   (无状态，下一步与上一步无关)
snapshot_2 → S1→S5 → CEPS分_2
snapshot_3 → S1→S5 → CEPS分_3
```

- 每个 snapshot 从**固定的 equal-weight 假设**出发
- 输出是**决策质量分数**
- 衡量的问题：**"模型在某一时刻的单步决策有多正确？"**

### Sandbox 回测

```
snapshot_1 → S1→S5 → weights_1 → 执行 → portfolio_state_1
                                              ↓ (真实持仓传入下一步)
snapshot_2 → S1→S5 → weights_2 → 执行 → portfolio_state_2
                                              ↓
snapshot_3 → ...
```

- 每步的 `current_weights` 和 `nav` 是**上一步真实执行后的结果**
- 非再平衡日每天按市场涨跌 mark-to-market 更新 NAV
- 输出是**真实 PnL**（Sharpe、回撤、CAGR……）
- 衡量的问题：**"模型的决策序列在真实市场里能赚多少钱？"**

### 两者为什么都需要

| 能力 | CEPS 流水线 | Sandbox 回测 |
|------|:-----------:|:------------:|
| 诊断哪个阶段出错 | ✓ | ✗ |
| 衡量单步决策质量 | ✓ | 间接（被市场噪声淹没）|
| 衡量长期策略盈利能力 | ✗ | ✓ |
| 考虑交易成本的累积 | ✗ | ✓ |
| 验证 CEPS 与 PnL 的相关性 | — | ✓（pipeline_logs 同时记录 CEPS）|
| API 消耗 | 中（每个 snapshot 调 LLM 3次）| 高（每个再平衡日调 LLM 3次）|
| 可用 MockData 快速跑 | ✓ | ✓ |

### 一个模型可能 CEPS 高但 Sharpe 低，原因

1. **标准答案基于历史动量，市场是均值回归的**：CEPS 的 ground truth 是"过去涨了就看多"，但动量策略在震荡市里会频繁亏损。
2. **单步正确 ≠ 序列一致**：CEPS 不惩罚月度翻转，但来回切换产生大量交易成本。
3. **权重精度 ≠ 风险控制**：S3 ground truth 是等权分配，精确复现等权的模型 S3 分高，但等权在高波动资产上暴露过多。
4. **执行成本累积**：一年 12 次再平衡 × (10+5) bps = 约 1.8% 的成本拖累，CEPS 看不到。

---

## 相关性报告

Sandbox 运行时会自动保存每个再平衡日的 S1–S5 prompt/response/score（存入 `pipeline_logs/`）。Fig 10 (`ceps_vs_pnl`) 直接读取 `backtest_result.json` 中的 `mean_ceps` 字段，可通过可视化脚本生成：

```bash
python examples/visualization/generate_all_figures.py --figures 10 --sandbox-dir outputs/sandbox
```
