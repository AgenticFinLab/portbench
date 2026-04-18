# PortBench Coding Plan

> 更新于 2026-04-16

---

## 现状概览

### 已完成模块

| 模块 | 路径 | 状态 |
|------|------|------|
| 数据收集 | `portbench/data_collect/` | ✅ 完成（Yahoo: 72个ticker；FRED: 60个序列） |
| 数据预处理 | `portbench/data_preprocess/` | ✅ 完成 |
| 数据质量评估 | `portbench/data_quality/` | ✅ 完成 |
| 评测指标库 | `portbench/metrics/` | ✅ 完成 |
| QA 数据集构建 | `portbench/qa_builder/` | ✅ 完成（T1–T7 + MockDataProvider + ProcessedDataProvider） |
| Agent 评测框架 | `portbench/agent_eval/` | ✅ 完成（S1–S3 含真实 LLM 提示词 + AnthropicAdapter / OpenAIAdapter / LiteLLMAdapter） |
| 基线策略 | `portbench/baselines/` | ✅ 完成（EqualWeight / 60-40 / RiskParity / SmartFolio） |
| 示例脚本 | `examples/qa_builder/` `examples/agent_eval/` | ✅ 完成 |

### 已收集数据（`datasets/`）

- **Kaggle**: 加密货币 OHLCV（115K 行）、商品价格、纳斯达克 100 历史（514K 行）、新闻文本
- **SEC**: AAPL / MSFT / JPM / XOM 的 10-K、10-Q HTML 文件
- **Yahoo / FRED**: 尚未下载（需 API Key）
- **已处理**: `datasets/processed/equities.csv`、`cryptocurrency.csv`

### Benchmark 目标

PortBench 针对现有金融 LLM 评测基准的三大系统性缺陷：

1. **多异构资产组合评测**（Gap 1）— 评测单元是六类资产（Equities, Bonds, Commodities, Real Estate, Cryptocurrency, Cash）的组合权重分配，而非单资产买卖决策
2. **风险优先评测范式**（Gap 2）— 模型须先通过压力测试（2008 金融危机、2020 COVID 闪崩、2022 加密崩盘）方可进入绩效排名；进一步对三类投资者画像（保守/稳健/激进）做个性化评测
3. **端到端全流程评测**（Gap 3）— 评测从市场信息解读到风险监控的完整五阶段流程，以 CEPS 指标量化跨阶段错误传播

### 双组件实现

1. **QA Dataset** — 7 种问题模板（T1–T7）× 4 个复杂度级别，基于历史市场数据生成问答对
2. **端到端 Agent 评测** — 五阶段流程（S1→S5），CEPS 量化跨阶段错误传播

---

---

## 各 Phase 完成状态

| Phase | 模块 | 代码状态 | 数据状态 |
|-------|------|---------|---------|
| Phase 1 | 数据层（collect / preprocess / quality） | ✅ 代码完成 | ⏳ Yahoo/FRED 数据待下载（需 API Key） |
| Phase 2 | `portbench/qa_builder/` T1–T7 | ✅ 完成，含 MockDataProvider | ⏳ 待接入真实 processed 数据 |
| Phase 3 | `portbench/metrics/` | ✅ 完成 | — |
| Phase 4 | `portbench/agent_eval/` | ✅ 完成 | — |
| Phase 5 | `portbench/baselines/` | ✅ 完成 | — |

**当前瓶颈**：所有代码逻辑已用 MockDataProvider（合成 GBM 数据）验证可运行。接下来需要：
1. 配置 API Key，运行 `examples/data_collect/get_all.py` 补全真实数据
2. 将 `MockDataProvider` 替换为 `ProcessedDataProvider`（读取 `datasets/processed/`）
3. 配置 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`，将 `MockAgentAdapter` 替换为 `AnthropicAdapter` / `OpenAIAdapter`
4. 运行 `examples/agent_eval/run_evaluation.py` 进行真实 LLM 评测

---

## 后续开发计划

### Phase 1：数据层补全（当前阶段）

目标：使所有六类资产的处理数据达到质量基线，为 QA 生成提供可靠输入。

#### 1.1 补充下载缺失数据

运行已有脚本，补全 Yahoo Finance 和 FRED 数据：

```bash
python examples/data_collect/get_all.py
python examples/data_preprocess/preprocess_all.py
python examples/data_quality/run_quality_check.py
```

根据 `datasets/quality_reports/report.json` 中的 FAIL / WARN 项决定是否需要补充数据源。

#### 1.2 数据质量迭代

- 对每个 FAIL 项分析根因（覆盖缺口 / 数据损坏 / 时间范围不足）
- 针对压力期（2008-09 ~ 2009-03、2020-02 ~ 2020-05、2022-05 ~ 2022-12）优先保证覆盖率 ≥ 90%
- 生成并保存市场状态标注：`datasets/processed/market_regimes.csv`（由 `label_market_regimes()` 输出）

---

### Phase 2：QA Dataset 构建

**目标路径**: `portbench/qa_builder/`

#### 2.1 基础类 `portbench/qa_builder/base.py`

```
QAConfig          — 配置：时间窗口、horizon、资产列表、每类样本数
ContextWindow     — 数据结构：date, asset_class, price_history, macro_context, news_text
QAPair            — 数据结构：context, question, answer, explanation, template_id, complexity_level, market_regime
QABuilder (ABC)   — 抽象基类：build_context(), generate_question(), compute_answer()
```

#### 2.2 各模板实现

每个模板对应一个 checker 文件，参考 `data_quality/` 的风格：

| 文件 | 模板 | 核心计算 |
|------|------|---------|
| `t1_return_prediction.py` | T1 | 未来 N 日收益方向（上涨 / 下跌 / 横盘），ground truth 来自真实价格 |
| `t2_risk_assessment.py` | T2 | 历史模拟法 VaR / CVaR，基于过去 252 日收益分位数 |
| `t3_position_sizing.py` | T3 | 基于最大回撤约束的凯利仓位或固定分数法 |
| `t4_pairwise_allocation.py` | T4 | 最小方差双资产配置（解析解：σ₁²、σ₂²、ρ） |
| `t5_multiasset_optimization.py` | T5 | 最大化 Sharpe（scipy 数值优化），权重约束 Σwᵢ=1, wᵢ≥0 |
| `t6_rebalancing.py` | T6 | 当前权重偏离目标超过阈值时触发再平衡，考虑交易成本 |
| `t7_regime_detection.py` | T7 | 调用 `label_market_regimes()` 给出市场状态 + 推荐配置调整方向 |

#### 2.3 数据集生成脚本 `examples/qa_builder/build_qa_dataset.py`

```
输入:  datasets/processed/*.csv + market_regimes.csv
输出:  datasets/qa_dataset/
         train.jsonl  (2015-2022)
         val.jsonl    (2023)
         test.jsonl   (2024-2025)
         stats.json   (模板分布、复杂度分布、市场状态分布)
```

每条记录格式：
```json
{
  "id": "T1_equities_20200315_001",
  "template": "T1",
  "complexity": 1,
  "market_regime": "crisis",
  "split": "train",
  "context": { "date": "...", "assets": [...], "price_history": {...}, "news": "..." },
  "question": "...",
  "answer": "down",
  "explanation": "..."
}
```

#### 2.4 PiT 校验

在 `QABuilder` 基类中强制：构建任何 context 时，`price_history` 和 `news` 的最新时间戳必须严格 < `decision_date`。通过单元测试覆盖边界条件。

---

### Phase 3：评测指标库

**目标路径**: `portbench/metrics/`

在 QA 评测和端到端评测中共用的指标计算，避免重复实现。

| 文件 | 内容 |
|------|------|
| `base.py` | `MetricsConfig`、`PortfolioMetrics` dataclass |
| `return_metrics.py` | `total_return()`, `cagr()` |
| `risk_metrics.py` | `volatility()`, `max_drawdown()`, `var()`, `cvar()` |
| `risk_adjusted.py` | `sharpe()`, `sortino()`, `calmar()`, `information_ratio()` |
| `allocation_metrics.py` | `weight_mae()`, `portfolio_return_gap()` |
| `ceps.py` | `CrossStageErrorPropagation` — 计算各阶段输出与理想输出的偏差，及级联放大系数 |

---

### Phase 4：端到端 Agent 评测框架

**目标路径**: `portbench/agent_eval/`

#### 4.1 五阶段流程接口 `portbench/agent_eval/base.py`

```
StageInput / StageOutput   — 各阶段输入输出数据结构
PipelineStage (ABC)        — 单阶段抽象：name, run(input) -> output
EvalPipeline               — 串联五个 stage，记录每阶段输出供 CEPS 计算
AgentAdapter (ABC)         — LLM 接入接口（OpenAI / Anthropic / HuggingFace）
```

#### 4.2 五阶段实现

| 文件 | 阶段 | 职责 |
|------|------|------|
| `s1_market_interpretation.py` | S1 | 将价格数据 + 新闻文本格式化为 LLM prompt，解析模型输出为结构化市场观点 |
| `s2_signal_generation.py` | S2 | 基于 S1 输出生成各资产的方向性信号（多 / 空 / 中性）及置信度 |
| `s3_weight_optimization.py` | S3 | 基于 S2 信号生成组合权重（调用 `metrics/` 中的优化工具或让 LLM 直接输出） |
| `s4_execution_simulation.py` | S4 | 给定目标权重和当前持仓，计算交易列表，模拟成交（含滑点、手续费） |
| `s5_risk_monitoring.py` | S5 | 计算持仓组合的实时风险指标，判断是否触发再平衡或止损 |

#### 4.3 压力测试场景注入 `portbench/agent_eval/stress_scenarios.py`

```
StressScenario         — 数据结构：name, start, end, shock_description
ScenarioInjector       — 从 processed 数据中截取压力期片段，注入 pipeline 作为测试输入
```

预定义场景：2008 金融危机、2020 COVID 闪崩、2022 加密货币崩盘。

#### 4.4 评测运行脚本 `examples/agent_eval/run_evaluation.py`

```
输入:  LLM adapter + 测试数据集（来自 processed/ 的 test split）
输出:  datasets/eval_results/{model_name}/
         per_stage_scores.json
         ceps_scores.json
         stress_test_results.json
         risk_first_ranking.json
```

---

### Phase 5：基线实现

**目标路径**: `portbench/baselines/`

为端到端评测提供非 LLM 对照组：

| 文件 | 策略 |
|------|------|
| `equal_weight.py` | 等权重 wᵢ = 1/n |
| `sixty_forty.py` | 60% 权益 + 40% 债券 |
| `risk_parity.py` | 风险平价 wᵢ ∝ 1/σᵢ |
| `smart_folio.py` | SmartFolio（IJCAI 2025）接口包装，作为非 LLM SOTA 基线 |

---

## 模块依赖关系

```
data_collect  ──►  data_preprocess  ──►  data_quality
                                              │
                                              ▼
                                        qa_builder  ──►  QA Dataset
                                              │
                                         metrics/
                                              │
                                        agent_eval  ──►  端到端评测结果
                                              │
                                         baselines  ──►  对照组结果
```

---

## 开发优先级

| 优先级 | 模块 | 前置依赖 | 备注 |
|--------|------|---------|------|
| P0 | Phase 1 数据补全 | — | 所有后续模块的基础 |
| P1 | `metrics/` | 无 | 可与 qa_builder 并行开发 |
| P1 | `qa_builder/` | Phase 1 | QA 数据集是论文核心产出之一 |
| P2 | `agent_eval/` | `metrics/` | 端到端框架 |
| P2 | `baselines/` | `agent_eval/` | 基线对照 |
| P3 | Sandbox（MarS 集成） | `agent_eval/` | 高保真仿真，工作量最大 |
