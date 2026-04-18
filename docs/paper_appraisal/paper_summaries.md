# 论文详细概述

> 项目背景：构建包含多种异构资产（股票、债券、商品、加密货币等）的数据集，并搭建sandbox环境，系统评测大语言模型的投资组合管理能力。

---

## 第一类：金融AI评测基准

---

### 1. FinRL-Meta
**标题**：FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning
**会议**：NeurIPS 2022 Datasets & Benchmarks Track
**链接**：https://arxiv.org/abs/2211.03107

- **Motivation**：金融RL研究领域缺乏标准化的市场环境和可复现的评测基准，导致不同研究之间难以横向比较，且数据处理流程不透明、结果可信度低。
- **Insight**：将金融市场的数据管道、环境构建、基线算法和回测评估统一为一个标准化框架，使得不同方法可在相同条件下公平竞争，类比OpenAI Gym对RL研究的贡献。
- **Method**：提供了包含股票、期货、外汇等多市场的标准化gym环境，集成了十余种经典RL算法（PPO、A2C、DDPG等），统一了数据下载、特征工程和回测接口。
- **Result**：在多个市场上验证了框架的可用性，提供了可复现的基准结果，并开源了完整代码和数据管道。
- **与本项目的关系**：**高度相关**。本项目构建的benchmark与FinRL-Meta目标一致，FinRL-Meta的环境设计（gym接口、多市场支持、标准化回测）可直接作为sandbox架构的参考。其多资产市场环境设计和评测指标体系值得借鉴。

---

### 2. PIXIU
**标题**：PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance
**会议**：NeurIPS 2023 Datasets & Benchmarks Track
**链接**：https://arxiv.org/abs/2306.05443

- **Motivation**：金融领域缺乏专门用于指令微调LLM的高质量数据集和系统性评测框架，已有工作要么只覆盖单一任务，要么评测设计不够严谨，无法全面衡量模型的金融能力。
- **Insight**：金融LLM的评测需要同时覆盖理解、推理和决策三个层次，且需要专业标注的指令数据才能让模型学会金融领域特有的推理模式。
- **Method**：构建了FIT（金融指令调优数据集，含136K条样本）、FLARE评测基准（覆盖情感分析、NER、关系抽取、问答、股价预测等9类任务）和基于LLaMA微调的FinMA模型。
- **Result**：FinMA在FLARE基准上显著优于GPT-4等通用模型，证明了领域专用指令微调的有效性；同时揭示了当前模型在复杂金融推理上的不足。
- **与本项目的关系**：**直接参考**。PIXIU的benchmark设计思路（多任务、分层评测）可指导本项目的评测维度设计，其将"股价预测"纳入评测的做法可扩展为完整的投资组合管理评测。

---

### 3. FinBen
**标题**：The FinBen: An Holistic Financial Benchmark for Large Language Models
**会议**：NeurIPS 2024 Datasets & Benchmarks Track
**链接**：https://arxiv.org/abs/2402.12659

- **Motivation**：现有金融LLM基准覆盖面窄，尤其缺乏对高级金融能力（如风险管理、投资决策、智能体交易）的评测，无法满足金融从业者对LLM实际能力的了解需求。
- **Insight**：金融AI能力应从信息提取、文本分析到复杂决策形成能力谱系，最高层次应包含智能体在真实市场环境中的端到端交易表现。
- **Method**：整合42个数据集，覆盖24类金融任务，从基础NLP任务（情感、NER）到高阶任务（股票交易Agent、风险评估），首次将基于智能体的市场交易纳入标准化评测。
- **Result**：系统评测了20+主流LLM，发现即使是GPT-4在智能体交易任务上表现也远不如人类专家，揭示了显著的能力缺口。
- **与本项目的关系**：**核心参考**。FinBen是目前最接近本项目目标的工作，其将"智能体交易评测"纳入benchmark的框架设计是本项目的直接参照。但FinBen以单一资产交易为主，本项目在此基础上扩展为多异构资产组合场景。

---

### 4. FinEval
**标题**：FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models
**会议**：NAACL 2025
**链接**：https://arxiv.org/abs/2308.09975

- **Motivation**：中文金融领域的LLM评测极度匮乏，现有通用中文评测（如C-Eval）不包含专业金融知识，无法衡量模型在中国金融市场场景下的真实能力。
- **Insight**：金融专业知识具有强领域性，需要从学术知识、行业实践知识、监管合规知识等多维度进行系统评测，单一维度评测存在严重片面性。
- **Method**：构建含8,351道题的中文金融评测基准，覆盖金融学（Finance）、会计（Accounting）、证券（Certificate）、经济学（Economics）四大类，支持零样本和少样本CoT评测。
- **Result**：主流中文LLM（GPT-4、ChatGPT、文心一言等）在专业金融知识上表现差异显著，GPT-4以约70%准确率领先，但仍远低于人类专家水平。
- **与本项目的关系**：**间接参考**。若本项目需要覆盖中文金融场景（如A股市场），FinEval的评测维度设计和中文数据构建方法具有参考价值。

---

### 5. FinQA
**标题**：FinQA: A Dataset of Numerical Reasoning over Financial Data
**会议**：EMNLP 2021
**链接**：https://arxiv.org/abs/2109.00122

- **Motivation**：金融分析的核心能力是对财务报告中的数字进行复杂推理（如计算同比增长率、利润率等），但现有QA数据集缺乏此类挑战，导致LLM在金融实战中的数值推理能力被高估。
- **Insight**：将数值推理过程显式化为可执行的程序（DSL），既能评测模型的推理正确性，又能提供可解释的推理链，比单纯的数值答案评测更严格。
- **Method**：由金融专家在SEC财报（10-K/10-Q）上标注8,281个问答对，每个问题附带多步数学运算的程序化推理链（使用专门的DSL），支持精确的自动化评测。
- **Result**：最优模型准确率约为50%，而人类专家达91%，揭示了巨大的人机差距，成为金融NLP领域最重要的基准之一。
- **与本项目的关系**：**参考价值**。本项目若需要评测LLM对财务数据（如财报、资产负债表）的理解能力，FinQA的数据构建方法和程序化推理标注方式具有借鉴意义。

---

### 6. ConvFinQA
**标题**：ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance QA
**会议**：EMNLP 2022
**链接**：https://arxiv.org/abs/2210.03849

- **Motivation**：真实金融分析场景往往是多轮交互的，分析师会在一次对话中逐步深入追问，而FinQA等单轮QA基准无法捕捉这种链式推理模式。
- **Insight**：多轮对话中的数值推理具有"链式依赖"特性——后续问题的答案往往依赖前面轮次的计算结果，这对LLM的长程推理追踪能力提出了更高要求。
- **Method**：基于FinQA扩展，构建包含3,892个对话（含11,100个问答轮次）的多轮数值推理数据集，每个对话包含2-4轮递进式追问，并标注完整的跨轮推理链。
- **Result**：最优模型在多轮设置下的准确率显著低于单轮设置，证明链式推理追踪是当前LLM的主要瓶颈。
- **与本项目的关系**：**参考价值**。若本项目的评测场景包含"投资顾问对话"或"多轮投资决策交互"，ConvFinQA的对话式评测设计提供了重要参考。

---

### 7. QuantBench
**标题**：QuantBench: Benchmarking AI Methods for Quantitative Investment
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2504.18600

- **Motivation**：量化投资领域AI方法层出不穷，但缺乏系统性的横向比较基准，各工作使用不同数据、不同评测协议，导致结论难以互信。
- **Insight**：量化投资的完整流程（因子挖掘→信号生成→组合构建→风险控制→回测）需要端到端的统一评测，而非只评测某一子模块。
- **Method**：构建覆盖股票、债券、商品等多资产类别的统一评测框架，标准化了数据预处理、特征工程、模型训练、回测协议，系统比较了ML/DL/LLM等各类方法。
- **Result**：发现传统因子模型在特定场景下仍有竞争力，DL方法优势主要体现在高频和另类数据处理上；LLM在定性分析上具有独特价值但定量预测能力有限。
- **与本项目的关系**：**高度相关**。QuantBench的多资产类别评测框架与本项目构建异构资产数据集的目标直接重叠，其评测协议设计（特别是多资产横向比较方案）是最直接的参考。

---

## 第二类：LLM金融智能体与投资组合管理

---

### 8. EconAgent
**标题**：EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities
**会议**：ACL 2024
**链接**：https://arxiv.org/abs/2310.10436

- **Motivation**：传统宏观经济模拟依赖手工设定的行为规则，难以捕捉真实经济主体的异质性决策行为，导致模拟结果与现实存在显著偏差。
- **Insight**：LLM具备理解经济情境、推理因果关系的能力，可作为经济主体的"大脑"驱动更真实的宏观经济仿真，实现从规则驱动到认知驱动的范式转换。
- **Method**：为每个经济主体（居民、企业）配备感知、记忆和决策三模块的LLM智能体，通过自然语言描述经济状态，由LLM生成消费、就业、投资等决策。
- **Result**：仿真系统能够复现失业率变化、通货膨胀等宏观经济现象，与历史数据的拟合度优于传统ABM（Agent-Based Model）。
- **与本项目的关系**：**架构参考**。EconAgent的多智能体宏观经济仿真与本项目的金融市场sandbox在架构上高度相似——都需要用LLM智能体在模拟环境中做出经济/投资决策。其感知-记忆-决策三模块设计可直接迁移到投资组合Agent的设计中。

---

### 9. FinTral
**标题**：FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models
**会议**：ACL 2024 Findings
**链接**：https://arxiv.org/abs/2402.10986

- **Motivation**：金融分析天然是多模态的（K线图、财务表格、文本报告、数值数据），而现有金融LLM大多只处理文本，无法处理图表等视觉信息，限制了其实际应用价值。
- **Insight**：通过在Mistral-7B基础上整合视觉编码器并用金融多模态数据微调，可以构建在多模态金融任务上达到GPT-4水平的开源模型。
- **Method**：基于Mistral-7B，整合LLaVA视觉模块，构建包含直接微调、指令微调、RLHF三个版本的模型家族；同时提出多模态金融幻觉缓解策略。
- **Result**：在图表问答、财报分析、金融情感等任务上达到GPT-4水平，开源权重使学术界可复现。
- **与本项目的关系**：**能力参考**。本项目评测LLM的投资组合能力时，若包含对K线图、技术图表的理解，FinTral所使用的多模态评测数据集和方法具有参考价值。

---

### 10. FinAgent
**标题**：FinAgent: A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist
**会议**：KDD 2024
**链接**：https://arxiv.org/abs/2402.18485

- **Motivation**：金融交易决策需要综合多来源信息（数值数据、新闻文本、技术图表），单一模态的交易系统存在信息利用不充分的问题；同时已有工作泛化性差，换一个市场就性能骤降。
- **Insight**：将LLM作为多模态信息的统一处理核心，配合工具调用实现实时数据获取，并通过双层反思机制（单次决策反思 + 跨资产经验迁移）提升泛化能力。
- **Method**：构建含多模态感知模块、市场情报工具集、双层记忆与反思系统的金融交易Agent，在股票、ETF、加密货币等多类资产上进行评测。
- **Result**：在6类资产、多个市场的交易任务上均优于现有基线，展示了工具增强型多模态Agent的泛化潜力。
- **与本项目的关系**：**强相关**。FinAgent的多资产、多模态评测设计与本项目的异构资产数据集高度契合。其在多类资产上的泛化评测方法可作为本项目评测协议的参考，双层反思机制也为被评测Agent的能力上限提供了参照。

---

### 11. TradingAgents
**标题**：TradingAgents: Multi-Agents LLM Financial Trading Framework
**会议**：arXiv 2024
**链接**：https://arxiv.org/abs/2412.20138

- **Motivation**：单一LLM在金融交易中面临信息过载、专业分工不足的问题，而真实投资机构通过分工协作（分析师、研究员、风险官、基金经理）来完成复杂决策，这一组织架构优势尚未被AI系统充分利用。
- **Insight**：模拟专业交易公司的组织结构，将不同类型的市场分析（基本面、技术面、情绪面、宏观面）分配给专门角色的LLM智能体，由基金经理Agent综合各方意见做最终决策。
- **Method**：构建含基本面分析师、技术分析师、新闻分析师、风险控制员、基金经理等多角色的协作框架，各角色通过结构化消息传递信息，最终生成具体的交易指令。
- **Result**：在累计收益、夏普比率、最大回撤等指标上均优于单体LLM基线和经典RL方法，证明了分工协作的有效性。
- **与本项目的关系**：**强相关**。TradingAgents为本项目提供了一个清晰的被评测Agent架构参照，同时其多角色分工框架揭示了投资组合管理能力的多维度性——本项目的评测可从基本面分析、技术分析、风险控制等多维度设计评测任务。

---

### 12. FinGPT
**标题**：FinGPT: Open-Source Financial Large Language Models
**会议**：NeurIPS/IJCAI 2023 Workshop
**链接**：https://arxiv.org/abs/2306.06031

- **Motivation**：BloombergGPT等金融专用LLM训练成本极高（数百万美元），学术界和中小机构无力负担，导致金融LLM研究资源高度集中，阻碍了领域创新。
- **Insight**：以数据为中心的方法论可以低成本实现金融LLM能力：通过高质量指令数据的LoRA微调，在消费级GPU上即可获得媲美专有模型的金融任务性能。
- **Method**：构建自动化金融数据采集管道（覆盖新闻、SEC文件、社交媒体、财报等34类数据源），结合LoRA/QLoRA低参数微调，在LLaMA系列基础上实现金融能力对齐。
- **Result**：金融情感分析等任务上达到或超越BloombergGPT，微调成本低至数百美元，推动了金融LLM的民主化。
- **与本项目的关系**：**基础参考**。FinGPT的数据采集方法和低成本微调范式为本项目构建数据集提供了工程参考，其情感分析能力可作为本项目中信息处理类评测任务的能力基线。

---

### 13. FinMem
**标题**：FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design
**会议**：AAAI 2024 Workshop
**链接**：https://arxiv.org/abs/2311.13743

- **Motivation**：简单将LLM用于交易决策时，模型缺乏对历史经验的有效利用机制，且对风险偏好的感知缺乏稳定性，导致决策质量不稳定。
- **Insight**：模拟人类交易员的认知结构——区分短期工作记忆（当日信息）、中期情景记忆（近期交易经验）和长期语义记忆（市场规律知识），并通过角色设定稳定风险偏好。
- **Method**：设计三层记忆模块（实时市场数据 → 近期经验摘要 → 长期市场知识），通过动态检索为决策提供相关上下文；同时引入可配置的风险偏好角色（保守/平衡/激进）。
- **Result**：相比无记忆LLM基线，在累计收益和夏普比率上有显著提升，角色化设计使不同风险偏好的模拟交易结果符合预期。
- **与本项目的关系**：**设计参考**。本项目设计测试场景时，FinMem的分层记忆框架提示我们评测LLM的长期记忆利用能力和风险偏好一致性，这两个维度应纳入评测指标体系。

---

### 14. BloombergGPT
**标题**：BloombergGPT: A Large Language Model for Finance
**会议**：arXiv 2023（NeurIPS 2023 Workshop）
**链接**：https://arxiv.org/abs/2303.17564

- **Motivation**：通用LLM（如GPT-3）在金融专业任务上表现不佳，因为金融语言（含专业术语、数值表达、监管文本）与通用文本差异显著，需要专门训练。
- **Insight**：将领域专用语料（363B token金融文本）与通用语料混合训练，可以在保持通用能力的同时显著提升金融领域表现，这一"混合训练"策略优于纯金融语料训练。
- **Method**：使用363B token的金融专用语料（彭博内部数据）+ 345B token通用语料，训练500亿参数的decoder-only模型，在9类金融任务上与通用模型对比评测。
- **Result**：在情感分析、命名实体识别、新闻分类等金融NLP任务上显著优于GPT-NeoX、OPT等同规模通用模型，同时在通用基准上保持竞争力。
- **与本项目的关系**：**基础参考**。BloombergGPT确立了金融LLM的能力基线，本项目可将其作为被评测模型之一，其在金融NLP任务上的强表现也为本项目设计难度适中的评测任务提供了参照。

---

## 第三类：强化学习投资组合优化

---

### 15. DeepTrader
**标题**：DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions
**会议**：AAAI 2021
**链接**：https://ojs.aaai.org/index.php/AAAI/article/view/16144

- **Motivation**：传统RL投资组合方法忽视市场整体状态（牛市/熊市/震荡市）对资产选择的影响，导致模型在市场状态切换时表现急剧退化。
- **Insight**：将投资组合管理分解为资产评分和市场状态感知两个子问题，通过感知宏观市场状态来动态调整资产配置策略，可显著提升风险调整后收益。
- **Method**：双流网络：资产评分流（图神经网络处理股票间关联）+ 市场状态流（LSTM感知宏观市场状态），两路信息融合后输出最终仓位决策，以夏普比率为优化目标。
- **Result**：在A股和美股数据集上，在年化收益、夏普比率、最大回撤等指标上均优于基线方法，市场状态感知模块贡献约15%的性能提升。
- **与本项目的关系**：**方法参考**。DeepTrader揭示了投资组合管理中市场状态感知的重要性，本项目可在sandbox中设计不同市场状态（牛市/熊市）的测试场景，评测LLM对市场状态的适应能力。

---

### 16. EarnMore
**标题**：Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools
**会议**：WWW 2024
**链接**：https://arxiv.org/abs/2311.10801

- **Motivation**：真实投资场景中，股票池经常变化（新股上市、退市、调仓限制），而现有RL方法通常假设固定股票池，缺乏处理动态可变资产集合的能力。
- **Insight**：通过可遮蔽的股票表示（Maskable Representation），允许模型在不重新训练的情况下处理任意子集的股票，实现对动态股票池的泛化能力。
- **Method**：设计可遮蔽的Transformer股票编码器，对不在当前池中的股票进行掩码处理，并结合对比学习增强跨股票池的表示一致性；在PPO框架下训练组合管理策略。
- **Result**：在多个股票池配置下均保持稳定性能，在动态股票池场景下显著优于固定池假设的基线，具有良好的泛化能力。
- **与本项目的关系**：**数据集设计参考**。本项目构建异构资产数据集时，EarnMore揭示的"动态资产集合"问题很重要：评测应包含资产集合变化的场景，测试LLM能否灵活应对可投资资产的动态变化。

---

### 17. FreQuant
**标题**：FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition
**会议**：KDD 2024
**链接**：https://dl.acm.org/doi/10.1145/3637528.3671668

- **Motivation**：金融时间序列包含多个频率成分（日内高频震荡、中期趋势、低频宏观周期），单一时间尺度的策略难以同时捕捉短期交易机会和长期配置价值。
- **Insight**：将金融信号分解为不同频率成分（类似小波变换），分别用不同策略处理不同频率信号，再通过RL学习最优的频率权重组合，实现自适应多频策略融合。
- **Method**：使用离散小波变换将资产价格序列分解为高、中、低频成分，为每个频率成分训练专门的子策略，通过主RL智能体动态调整各子策略的权重。
- **Result**：在股票和加密货币数据上，在年化收益和夏普比率上优于单一频率基线，且在市场状态变化时表现出更强的适应性。
- **与本项目的关系**：**方法参考**。多频率分解的思路对本项目有启发：在sandbox中，可以设计要求LLM兼顾短期和长期收益的组合优化任务，评测其多时间尺度决策能力。

---

### 18. TRA
**标题**：Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport
**会议**：KDD 2021
**链接**：https://arxiv.org/abs/2106.12950

- **Motivation**：不同股票呈现出截然不同的交易模式（价值股、成长股、周期股等），用单一模型统一预测所有股票会导致模式混淆，降低预测精度。
- **Insight**：利用最优传输理论将不同股票样本自适应地路由到最合适的专家预测器，通过"路由"而非"分类"来处理模式分配，避免了离散分类的梯度问题。
- **Method**：设计时序路由适配器（Temporal Routing Adaptor），包含多个专家预测器和一个软路由机制，使用最优传输（Sinkhorn算法）实现样本到专家的可微分分配。
- **Result**：在A股和美股数据集上，股票收益预测的Rank IC和RankICIR均优于单一模型基线，在回测中获得更高的年化收益。
- **与本项目的关系**：**参考价值**。TRA揭示了股票交易模式的多样性，本项目在构建异构资产数据集时，应纳入具有不同交易特征的资产（高波动/低波动、周期性/防御性等），以全面测试LLM对多样化资产的理解能力。

---

### 19. EarnHFT
**标题**：EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading
**会议**：AAAI 2024

- **Motivation**：高频交易的决策频率极高（毫秒级），且需要同时考虑短期价格冲击和长期持仓成本，单层RL架构难以协调不同时间尺度的优化目标。
- **Insight**：将高频交易分解为宏观层（决定交易方向和时机）和微观层（优化订单执行细节）的层次结构，各层专注于其时间尺度内的优化，通过层级接口传递信息。
- **Method**：构建两层RL架构：宏观策略网络（以分钟级K线数据为输入，输出目标持仓方向）+ 微观执行网络（以订单簿数据为输入，优化具体的下单时机和数量）。
- **Result**：在真实高频交易数据上，相比单层RL基线提升了约20%的收益，同时降低了交易成本和滑点损失。
- **与本项目的关系**：**场景参考**。EarnHFT的层次化决策架构对本项目有参考意义：在sandbox设计中，可区分战略层（资产配置决策）和执行层（具体交易操作），评测LLM在不同层次的决策能力。

---

## 第四类：金融市场仿真环境

---

### 20. MarS
**标题**：MarS: A Financial Market Simulation Engine Powered by Generative Foundation Model
**会议**：ICLR 2025
**链接**：https://arxiv.org/abs/2409.07486

- **Motivation**：现有金融市场模拟环境（如gym环境）基于简化的统计假设，与真实市场行为差异显著，导致在模拟环境中训练的策略在真实市场中泛化性差（sim-to-real gap）。
- **Insight**：将订单流（Order Flow）类比为NLP中的Token序列，用大规模生成式模型（LMM: Large Market Model）学习市场的生成机制，从而创建高保真、可交互、可控的市场仿真器。
- **Method**：在真实逐笔成交数据上预训练大市场模型（LMM），学习条件生成下一笔订单的能力；支持研究者注入外生冲击（如政策变化、市场事件）并观察市场反应，用于策略测试和风险压测。
- **Result**：仿真市场在统计特性（收益分布、波动聚集、量价相关性等）上与真实市场高度一致；在仿真环境中训练的策略在真实市场中具有更好的迁移性。
- **与本项目的关系**：**最直接相关**。MarS是本项目sandbox环境构建的最核心参考。其生成式市场仿真框架解决了sim-to-real gap问题，LMM的条件生成能力可用于为本项目的sandbox提供高保真的市场数据流。本项目可在MarS基础上扩展多资产、多市场的仿真环境。

---

### 21. FinRL
**标题**：FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance
**会议**：NeurIPS 2021 Workshop
**链接**：https://arxiv.org/abs/2011.09607

- **Motivation**：量化金融RL研究高度碎片化，各工作使用不同的数据源、不同的环境接口、不同的评测指标，极大地阻碍了方法间的比较和知识积累。
- **Insight**：参照OpenAI Gym的标准接口设计，为金融RL提供统一的"数据→环境→算法→回测"全栈框架，使研究者能够以最小代价复现和扩展他人工作。
- **Method**：实现了标准化的金融gym环境（含观测空间、动作空间、奖励函数的统一定义），集成10+种RL算法（PPO、SAC、TD3等），提供股票、期货、外汇、加密货币的一键数据下载和特征工程。
- **Result**：被广泛采用，成为金融RL领域最重要的开源基础设施之一，GitHub星标超过10K。
- **与本项目的关系**：**工程参考**。FinRL的gym环境接口设计可直接用于本项目sandbox的技术实现，其多市场支持和标准化回测模块提供了成熟的工程实践参考。

---

### 22. Qlib
**标题**：Qlib: An AI-oriented Quantitative Investment Platform
**会议**：IJCAI 2021 Workshop
**链接**：https://arxiv.org/abs/2009.11189

- **Motivation**：量化投资的AI研究需要高质量、标准化的数据基础设施，但学术界普遍缺乏覆盖数据管理、模型训练、回测全流程的统一平台，导致研究效率低下。
- **Insight**：以数据为核心，构建覆盖因子挖掘、信号生成、组合优化、回测全流程的AI量化投资平台，通过标准化数据接口降低研究门槛，加速方法迭代。
- **Method**：实现了高性能时序数据引擎（支持高效的截面数据和时序数据查询）、标准化特征工程管道、多种预测模型（LightGBM、LSTM、Transformer等）和仿真回测引擎，支持A股和美股数据。
- **Result**：提供了系列基准模型和可复现的回测结果，成为量化AI研究的重要开源平台，微软亚研院持续维护和更新。
- **与本项目的关系**：**工程参考**。Qlib的数据管理和回测架构对本项目构建异构资产数据集具有工程借鉴价值，特别是其高效的截面数据处理方式和因子库设计。

---

### 23. mbt-gym
**标题**：mbt-gym: Reinforcement Learning for Model-Based Limit Order Book Trading
**会议**：ACM ICAIF 2023
**链接**：https://arxiv.org/abs/2209.15162

- **Motivation**：限价订单簿（LOB）交易是金融市场微结构的核心，但现有RL环境多基于K线级数据，缺乏能够准确模拟LOB动态的训练环境，限制了微结构级别策略的研究。
- **Insight**：基于随机过程模型（Avellaneda-Stoikov等）构建LOB模拟环境，既保证了数学严格性，又提供了比历史数据回放更具多样性的训练样本，有助于避免过拟合。
- **Method**：实现了基于随机控制理论的LOB仿真模型，支持做市商策略（Market Making）和最优执行（Optimal Execution）两类问题，提供标准gym接口，支持参数化的市场冲击和价差模型。
- **Result**：框架在做市策略优化任务上，相比纯历史数据回放环境，训练出的策略泛化性更强；并为多种LOB随机模型提供了统一实现。
- **与本项目的关系**：**场景参考**。若本项目的sandbox需要模拟市场微结构（如订单执行场景），mbt-gym的设计思路和接口规范具有参考价值，特别是其"基于模型的仿真"与"历史数据回放"的混合策略。

---

## 补充论文（第一批新增）

---

## 第一类（续）：金融AI评测基准 — 新兴范式（2024-2026）

---

### 24. LLM Agents for Investment Management (Survey)
**标题**：Large Language Model Agents for Investment Management: Foundations, Benchmarks, and Research Frontiers
**会议**：ACM MM 2024（调查综述）
**链接**：https://dl.acm.org/doi/full/10.1145/3768292.3770387

- **Motivation**：LLM智能体在金融投资领域的应用文献分散，缺乏系统性的分类梳理与研究前沿识别，从业者和研究者难以把握全局。
- **Insight**：投资管理类LLM智能体的核心能力集中在多智能体协作与工具增强管道两个维度；鲁棒性与可解释性是当前最关键的未解难题，而非单纯的性能提升。
- **Method**：文献综述方法，按使用场景（组合优化、风险管理、信息检索、策略生成）和架构创新（多智能体协作、反思机制、工具增强管道）系统分类；梳理现有评测框架与基准数据集。
- **Result**：识别出当前最关键的三大开放挑战：鲁棒性（应对分布外市场状态）、可解释性（决策过程的可审计性）和真实部署就绪性（sim-to-real gap）。
- **与本项目的关系**：**高度相关**。本综述提供了最完整的LLM投资智能体分类法，直接指出我们benchmark必须涵盖鲁棒性和可解释性评测才能代表SOTA水准。其对现有评测基准的梳理揭示了本项目填补的空白。
- **评测维度启示**：鲁棒性（压力场景通过率）和可解释性（决策理由质量评分）应作为我们benchmark的一级评测维度，而非仅报告收益率。

---

### 25. XFinBench
**标题**：XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning
**会议**：ACL 2025 Findings
**链接**：https://aclanthology.org/2025.findings-acl.457/

- **Motivation**：现有金融基准测试的是浅层语言理解，而非研究生级别的专业金融问题求解能力，导致LLM在真实金融任务中的能力被高估。
- **Insight**：将LLM金融能力分解为五个核心维度——术语理解、时序推理、未来预测、场景规划、数值建模——可以精确定位短板，尤其时序推理与场景规划是最大瓶颈。
- **Method**：构建含4,235个样例的多模态基准，覆盖研究生级别金融话题；评测18个前沿模型；构建3,032条金融术语的知识库用于知识增强实验；进行详细误差分析。
- **Result**：最优模型（o1）准确率67.3%，仍落后人类专家12.5%；时序推理和场景规划差距最大；知识增强仅对小规模开源模型有稳定提升；视觉上下文和曲线位置理解是关键失败模式。
- **与本项目的关系**：**直接参考**。本项目benchmark应纳入时序推理任务（市场状态变迁识别、历史模式识别）与场景规划任务（宏观情景下的组合调整），这两个维度是当前LLM的系统性弱点，在多资产组合场景中尤为关键。
- **评测维度启示**：时序推理（跨资产的时序相关性分析）与场景规划（假设性宏观冲击下的组合再平衡）是我们benchmark独特且有难度的评测轴。

---

### 26. INVESTORBENCH
**标题**：InvestorBench: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent
**会议**：ACL 2025 Long Papers
**链接**：https://aclanthology.org/2025.acl-long.126/

- **Motivation**：金融LLM智能体领域存在两大核心挑战：缺乏适应多类金融任务的统一智能体框架，以及缺乏标准化基准和一致性数据集来评估智能体表现。
- **Insight**：不同金融产品（股票、加密货币、ETF）展现出截然不同的能力需求模式，单一资产类别的评测无法揭示跨产品的能力差异，统一框架下的多产品评测才能给出全面画像。
- **Method**：构建首个覆盖股票、加密货币和ETF的LLM智能体评测基准；测试13种不同LLM作为backbone；提供开源数据集与多市场环境；在多种市场条件和任务类型下评测推理与决策能力。
- **Result**：不同产品类型下没有单一模型占主导；加密货币任务整体最难；开源数据集和标准化环境提升了可复现性；LLM backbone的能力差异在不同任务类型下表现不一。
- **与本项目的关系**：**核心对比对象**。INVESTORBENCH是与本项目最接近的现有工作。关键区别在于：INVESTORBENCH评测的是每类产品的单资产决策，而本项目评测的是跨异构资产类别的**投资组合级决策**——权重分配、相关性管理、风险预算——这是INVESTORBENCH未涉及的维度。
- **评测维度启示**：资产类别条件化的能力画像（per-asset-class performance profile）应作为我们benchmark的标准输出之一，以便与INVESTORBENCH进行直接对比。

---

### 27. StockBench
**标题**：StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?
**会议**：arXiv 2024
**链接**：https://arxiv.org/abs/2510.02209

- **Motivation**：现有金融基准多为静态问答，无法捕捉真实市场交易的动态性；已有基准存在数据污染风险，无法真实反映模型的泛化能力。
- **Insight**：静态问答能力与动态交易能力是两种截然不同的能力——在QA基准上表现强的LLM在交易任务中不一定表现好，说明"语言能力≠交易技能"，针对交易任务的专项评测不可缺少。
- **Method**：构建无污染的基准，每日提供价格、基本面和新闻信号，LLM做出买/卖/持有序列决策；跨数月真实市场数据；使用金融指标（累计收益、最大回撤、Sortino比率）作为主要评估标准。
- **Result**：大多数LLM无法跑赢简单的买入持有基准；部分模型展现出更高收益和更强风险管理的潜力；QA表现与交易表现相关性低，结论不可互换。
- **与本项目的关系**：**重要对比**。本项目benchmark必须以金融绩效指标（而非QA准确率）作为主要评测标准，StockBench的"QA≠交易技能"发现是最有力的论据。本项目将此原则从单股交易扩展至多资产组合管理。
- **评测维度启示**：Sortino比率和最大回撤应作为标准评测指标，与累计收益并列；多月连续决策的动态评测优于静态快照评测。

---

### 28. FINSABER
**标题**：Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2505.07078

- **Motivation**：现有LLM交易策略评测集中于短时间窗口和有限资产，存在幸存者偏差和数据窥探偏差，过度高估了LLM策略的有效性。
- **Insight**：LLM策略的"优势"在更长回测期和更广泛资产域下显著消退，且呈现系统性的市场状态失配——牛市中过于保守，熊市中过于激进，这是LLM策略的根本缺陷，而非调参可以解决的问题。
- **Method**：FINSABER框架，20年回测、100+股票品种；市场状态分析（牛/熊市分解）；跨时间段、跨资产域的泛化评测；与被动基准（买入持有）的系统比较。
- **Result**：LLM策略优势在更长时间段和更大资产域下显著退化；牛市中过保守、熊市中过激进是系统性失配；现有LLM策略缺乏趋势检测和状态感知能力。
- **与本项目的关系**：**高度相关**。本项目benchmark应明确将回测期划分为牛/熊/横盘/危机等市场状态，分状态报告绩效，从而系统揭示LLM策略的市场状态失配问题。这是对FINSABER发现的直接响应，也是有别于现有benchmark的重要维度。
- **评测维度启示**：分市场状态的绩效分解（per-regime performance decomposition）和市场状态转换下的适应性评测应作为一级评测轴。

---

### 29. BizFinBench
**标题**：BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2505.19457

- **Motivation**：现有基准难以评估LLM在逻辑密集、精确度要求高的实际金融应用场景（如财务计算、多概念交叉推理）中的可靠性。
- **Insight**：当前LLM在复杂场景下表现显著下滑，尤其是需要跨概念推理的场景，且不同能力维度上没有任何一个模型占据主导，说明LLM能力呈高度任务特异性。
- **Method**：6,781条中文标注查询，覆盖五大维度（数值计算、推理、信息抽取、预测识别、知识问答）；九类细分任务；测试25个模型；引入IteraJudge减少LLM-as-evaluator偏差。
- **Result**：数值计算领域Claude-3.5-Sonnet与DeepSeek-R1并列领先；推理能力专有模型领先开源模型19.49分；信息抽取性能差距最大；预测识别方差最小；复杂推理是普遍弱点。
- **与本项目的关系**：**间接参考**。若本项目覆盖A股或中国市场场景，BizFinBench的五维度框架和IteraJudge评测方法具有借鉴价值。跨资产信息聚合能力（类比跨概念推理）应作为我们benchmark的重要难度维度。
- **评测维度启示**：跨资产类别的信息整合推理（如同时分析股债商品的相关性关系）是类比于跨概念推理的核心难度来源，应在任务设计中显式体现。

---

### 30. Standard Benchmarks Fail
**标题**：Standard Benchmarks Fail — Auditing LLM Agents in Finance Must Prioritize Risk
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2502.15865

- **Motivation**：现有benchmark以收益指标为核心，忽视了LLM金融智能体在真实部署中的安全风险——幻觉事实、过期数据、对抗提示操控——仅靠准确率和收益指标无法判断智能体是否安全可部署。
- **Insight**：金融LLM智能体的评测应首先建立风险画像，而非优化绩效点估计。三层风险审计框架（模型层、工作流层、系统层）可以揭示传统指标完全忽略的隐藏弱点，"安全预算"应作为首要成功标准。
- **Method**：对6个API级和开源权重LLM智能体在3个高影响力任务上进行压力测试；识别传统benchmark遗漏的隐藏弱点；提出模型/工作流/系统三层压力测试框架；提出安全预算概念。
- **Result**：所有受测智能体均存在传统benchmark未能揭示的隐藏弱点；返回幻觉信息、使用过期数据、对对抗提示敏感是三类普遍问题；常规指标给出了虚假可靠性幻觉。
- **与本项目的关系**：**直接创新依据**。这是本项目"风险优先评测范式"最强的学术依据。我们可以明确引用此文，论证收益独占评测的不充分性，并将安全预算合规率、压力场景通过率、幻觉频率作为一级评测标准，而非次级指标。
- **评测维度启示**：安全预算（最大可接受风险预算的合规率）、压力场景通过率、幻觉频率和过期数据敏感性应作为我们benchmark的主要评测标准——这是与所有现有金融benchmark的最显著差异。

---

### 31. Look-Ahead-Bench
**标题**：Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance
**会议**：arXiv 2026
**链接**：https://arxiv.org/abs/2601.13770

- **Motivation**：金融LLM中的前视偏差（look-ahead bias）会导致评测虚高，但缺乏针对实用金融工作流（而非仅问答）的标准化前视偏差测量工具，现有方法无法区分真实预测能力与记忆驱动的表现。
- **Insight**：Alpha衰减分析（而非问答准确率）是测量前视偏差的正确方法；标准LLM在不同时间市场状态之间存在显著前视偏差；PiT（Point-in-Time）模型随规模提升表现出更好的泛化和推理能力，而非靠记忆作弊。
- **Method**：跨时间上区分市场状态的绩效衰减分析；alpha衰减作为偏差度量指标；对比标准LLM（Llama 3.1 8B/70B、DeepSeek）与PiT专用模型（Pitinf系列）；实用金融工作流场景而非问答场景。
- **Result**：标准LLM存在显著前视偏差；Pitinf模型随规模提升展现出泛化提升而非记忆依赖；alpha衰减是判断模型是否适合真实部署的有效指标。
- **与本项目的关系**：**直接创新依据**。本项目benchmark应系统审计前视偏差——将"时间完整性"（Point-in-Time合规性）作为一级评测标准。报告的绩效若来自信息泄漏而非真实预测能力，则毫无意义。我们的benchmark是首个将前视偏差审计纳入多资产组合评测的框架。
- **评测维度启示**：前视偏差分数（alpha衰减指标）和PiT合规审计应作为benchmark的质量控制轴，确保报告的绩效反映真实预测能力而非信息泄漏。

---

### 32. FinMaster
**标题**：FinMaster: A Holistic Benchmark for Mastering Full-Pipeline Financial Workflows with LLMs
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2505.13533

- **Motivation**：现有金融benchmark测试孤立技能，而不是端到端工作流执行，导致无法揭示多步骤任务中错误传播带来的绩效崩溃——单步准确率高并不保证完整流程成功。
- **Insight**：错误在流程中系统性传播——从基础任务90%+的准确率到复杂多步骤任务40%的准确率——单指标计算的58%准确率在多指标场景下降至37%，错误复合效应不可忽视。
- **Method**：三模块框架：FinSim（生成合成隐私合规金融数据的模拟器）、FinSuite（183个跨会计/审计/咨询领域的多难度任务）、FinEval（统一评测接口）；系统测试了多个SOTA模型。
- **Result**：准确率从基础任务（>90%）到复杂场景（约40%）的退化是系统性的；单指标计算（58%）在多指标场景（37%）显著退化；错误传播是主要机制；首个覆盖完整金融工作流的benchmark。
- **与本项目的关系**：**重要参考**。本项目应设计包含完整投资组合管理流程的任务（市场数据解读→信号生成→组合优化→执行模拟→风险监控），而不仅是孤立决策任务。流程各阶段的错误传播测量是本项目可以引入的独特评测维度。
- **评测维度启示**：流程完成率和各阶段错误传播测量应作为组合管理能力的重要评测指标，揭示LLM在端到端场景下与孤立任务下的能力差异。

---

### 33. FinRL Contests
**标题**：FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2504.02281

- **Motivation**：FinRL策略对个人研究者而言应用门槛高、易出错；缺乏标准化任务定义、高质量数据集、接近真实的市场环境和可靠基线，阻碍了开源社区与金融科技行业的一致性复现。
- **Insight**：竞赛格式通过明确的任务定义、标准化环境和真实竞争驱动了可复现的进步；LLM工程化信号是新兴的独立任务类别，需要专门的评测框架。
- **Method**：2023-2025年系列竞赛，覆盖股票交易、订单执行、加密货币交易和LLM工程化信号四类任务；200+参与者来自100+机构；提供GPU优化的并行市场环境和集成学习起步工具包。
- **Result**：竞赛格式识别出FinRL的三大核心挑战：非平稳性处理、低信噪比环境、市场摩擦建模；LLM工程化信号是表现最突出的新兴研究方向。
- **与本项目的关系**：**工程参考**。FinRL竞赛基础设施（并行环境、集成学习工具包、评测协议）为本项目benchmark的竞争性评测环境提供了直接工程参考。"LLM工程化信号"作为独立任务类别的确立，支持本项目将信号生成能力纳入评测框架。
- **评测维度启示**：LLM工程化信号评测应作为本项目的独立任务类别；竞赛式任务定义方式有助于提升benchmark的可复现性和社区参与度。

---

## 第二类（续）：LLM金融智能体与投资组合管理 — 新兴方向（2024-2026）

---

### 34. 3S-Trader
**标题**：3S-Trader: A Multi-LLM Framework for Adaptive Stock Scoring, Strategy, and Selection in Portfolio Optimization
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2510.17393

- **Motivation**：现有LLM交易方法聚焦于单股交易，缺乏在多候选股票上进行组合构建推理的能力；且现有方法在市场转换时策略调整能力不足。
- **Insight**：将组合构建分解为评分（单股信号汇总）、策略（市场状态分析与历史策略迭代优化）、选股（基于策略的组合组装）三个独立模块，使LLM能够对多股进行有原则的比较推理，无需额外训练。
- **Method**：免训练三模块框架；评分模块为每只股票生成多维度信号报告；策略模块分析历史策略与整体市场状态；选股模块基于策略在相关维度挑选最优股票；评测DJIA及行业专题股票池。
- **Result**：DJIA成分股上累计收益131.83%，夏普比率0.31，卡尔玛比率11.84；在其他行业板块也取得一致性强结果，优于现有多LLM框架和时序基线。
- **与本项目的关系**：**直接被评测对象参考**。3S-Trader是我们benchmark应评测的多股组合智能体的典型代表。其三模块分解表明，本项目benchmark应分别评测评分质量（信号生成）、策略质量（市场分析推理）和选股质量（组合构建决策）。
- 

---

### 35. When Agents Trade (AMA)
**标题**：When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents
**会议**：arXiv 2024
**链接**：https://arxiv.org/abs/2510.11695

- **Motivation**：LLM交易研究大多测试模型（而非智能体），覆盖期间有限、资产单一，且依赖未经验证的数据，无法判断LLM智能体在真实市场中能否有效推理和适应。
- **Insight**：智能体框架架构对行为模式的影响远大于模型backbone——框架选择是首要性能驱动因素，而非模型能力。这意味着benchmark必须在固定框架下比较模型，并在固定模型下比较框架。
- **Method**：首个终身实时benchmark，横跨加密货币和股票市场；集成经过验证的交易数据和专家审核的新闻；实现4种智能体架构（InvestorAgent、TradeAgent、HedgeFundAgent、DeepFundAgent）× 5种LLM（GPT-4o/4.1、Claude、Gemini）。
- **Result**：智能体框架展现出截然不同的行为模式（激进风险偏好到保守决策），而模型backbone的贡献相对有限；AMA建立了严格、可复现、持续进化的金融推理评测基础。
- **与本项目的关系**：**核心对比对象**。AMA的终身实时benchmark设计是本项目持续评测机制的重要参考。"框架>模型backbone"的发现直接说明我们benchmark应同时评测架构选择（而非仅模型能力），从而区分系统级行为与模型级能力。
- 

---

### 36. AlphaSharpe
**标题**：AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2502.00029

- **Motivation**：传统风险调整指标（如夏普比率）在动态波动市场条件下鲁棒性差、泛化能力有限，无法可靠预测未来绩效。
- **Insight**：LLM可以利用其隐式编码的领域知识，通过交叉、变异、评估的进化循环迭代生成并优化金融指标公式，发现的新指标在稳健性和预测力上显著优于人工设计的经典指标。
- **Method**：基于LLM的进化优化框架：LLM生成并精炼指标公式；评分机制确保新指标在未见数据上的泛化能力；在真实世界数据集上验证。
- **Result**：发现的指标对未来风险收益的预测力达传统夏普比率的3倍；组合绩效提升2倍；证明LLM是金融分析工具迭代的有效引擎。
- **与本项目的关系**：**方法创新参考**。AlphaSharpe提示本项目benchmark可以考虑使用LLM发现的指标（而非固定的夏普/索提诺比率）作为评测标准之一，也可以将"LLM能否正确推理新型风险指标"设计为一类评测任务。
- 

---

### 37. Enhancing LLM Performance in Asset Selection
**标题**：Enhancing LLM Performance in Asset Selection: Investigating the Integration Challenges of Traditional Quantitative Signals
**会议**：ACM MM 2024
**链接**：https://dl.acm.org/doi/full/10.1145/3762249.3762294

- **Motivation**：量化信号（OLS、XGBoost预测）被预期可以增强LLM的资产选择能力，但二者结合的实际效果和机制尚不清楚，存在大量直觉假设未经实证验证。
- **Insight**：将量化信号提供给LLM反而会降低其绩效——这揭示了LLM的信息整合能力存在根本性局限：LLM无法有效处理外部定量预测结果，更多信息并不带来更好决策。
- **Method**：ETF资产选择任务；对比LLM独立预测与LLM+OLS/XGBoost预测；调整提示词设计；测试多个不同LLM；设计一系列受控实验探究信息整合失败的机制。
- **Result**：LLM独立表现优于传统量化模型；但加入OLS/XGBoost预测后LLM绩效显著下降；在大多数情况下，更详细的信息或更先进的模型并不能改善结果。
- **与本项目的关系**：**重要发现**。信息整合失败的发现对本项目benchmark设计至关重要：我们必须专项评测LLM能否正确整合量化信号（因子值、优化器输出）。这揭示的不是边缘问题，而是当前LLM组合管理能力的核心缺陷，也是本项目benchmark的重要评测场景。
- 

---

### 38. LLM-Powered Multi-Agent Crypto Portfolio
**标题**：LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2501.00826

- **Motivation**：加密货币投资因历史数据短、多模态信息需求高、推理复杂度大，是LLM金融应用的高难度场景；单LLM在复杂综合任务上的局限性在加密资产中更为突出。
- **Insight**：团队内部（intra-team）与团队间（inter-team）双层协作机制通过置信度调整和信息共享，系统性地提升预测准确率；按子任务（数据分析、文献整合、投资决策）进行智能体专业化分工优于通才型智能体。
- **Method**：多模态、多智能体框架，专业团队处理数据分析、专业文献整合和投资决策；专家训练模块对历史数据和专业投资文献进行微调；实时数据驱动多智能体投资决策；评测市值前30加密货币。
- **Result**：在分类、资产定价、组合绩效和可解释性指标上均优于单智能体模型和市场基准；双层协作机制是绩效提升的主要来源。
- **与本项目的关系**：**强相关**。多智能体按资产类别专业化分工是本项目benchmark应评测的系统架构模式。可解释性指标的引入是本项目应采纳的评测维度之一。此论文也验证了多资产管理中专业化分工的必要性。
- 

---

### 39. Democratizing Alpha
**标题**：Democratizing Alpha: LLM-Driven Portfolio Construction for Retail Investors Using Public Financial Media
**会议**：ACM MM 2024
**链接**：https://dl.acm.org/doi/abs/10.1145/3768292.3770376

- **Motivation**：个人投资者因有限理性（时间、认知、信息处理约束）在资本市场决策中处于劣势；公开可获取的LLM和金融媒体信息是否能弥补这一差距，尚无系统性验证。
- **Insight**：LLM能够从非结构化视频内容中提取经济上有意义的投资逻辑并构建投资组合，实证表明在多个绩效指标上持续超越市场基准，"alpha民主化"在技术上是可行的。
- **Method**：使用Bloomberg TV和Yahoo Finance等YouTube公开视频转录文字提示四种LLM（LLaMA 3、Qwen2、Gemma、GPT 4o-mini）构建投资组合；2024年6月至2025年7月对标S&P 500和纳斯达克回测；分析定性推理质量。
- **Result**：LLM组合在CAGR、夏普比率和卡尔玛比率上持续优于市场基准；LLM成功提取出连贯且经济上有意义的投资逻辑；不同LLM之间存在显著的绩效差异。
- **与本项目的关系**：**评测场景参考**。本项目benchmark应包含"另类信息源"场景（社交媒体、视频转录、非结构化文本），评测LLM从非传统非结构化数据中提取组合相关信号的能力，这是当前评测框架的空白。
- 

---

### 40. GuruAgents
**标题**：GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2510.01664

- **Motivation**：传奇投资者（巴菲特等）的投资哲学是定性的、非正式化的，尚未被系统地转化为可复现的量化策略；LLM的提示工程能力为此提供了可能途径。
- **Insight**：将投资大师的哲学编码为LLM提示，可以可复现地将定性哲学转化为定量策略；不同哲学产生截然不同但内在一致的行为模式，说明"投资风格一致性"是LLM真实可测的属性。
- **Method**：五个风格各异的GuruAgent（分别模拟不同投资大师），将独特哲学编码为LLM提示并整合金融工具和确定性推理管道；在纳斯达克100成分股进行2023Q4-2025Q2的回测。
- **Result**：Buffett GuruAgent实现42.2% CAGR，显著超越基准；不同智能体展现出由提示哲学驱动的独特行为模式；提示工程可以成功将定性哲学转化为可复现的量化策略。
- **与本项目的关系**：**评测设计参考**。本项目benchmark可以引入"投资风格一致性"测试——LLM是否在不同市场条件下保持连贯的投资哲学？这是一个跨定性与定量的新评测轴，也是区分技能驱动绩效与风格-状态运气的关键。
- 

---

### 41. From Text to Returns
**标题**：From Text to Returns: Using Large Language Models for Mutual Fund Portfolio Optimization and Risk-Adjusted Allocation
**会议**：arXiv 2024
**链接**：https://arxiv.org/abs/2512.05907

- **Motivation**：共同基金行业配置决策复杂，传统方法无法处理宏观定性信号；LLM与标准金融优化方法的结合效果尚待系统验证。
- **Insight**：RAG管道为LLM提供实时外部数据和宏观经济背景，可生成情境感知的配置策略；Zypher 7B等中等规模模型通过RAG增强后可超越更大模型，说明框架设计比模型规模更关键。
- **Method**：RAG管道（实时数据 + 全球经济信号）+ 标准金融优化方法；测试Phi 2、Mistral 7B、Zypher 7B；基金行业板块配置任务；与基础配置方法对比。
- **Result**：Zypher 7B最优；RAG显著优于无增强基线；LLM在风险调整收益上超越传统基础配置方法；不同模型在风险偏好和信息处理方式上呈现差异。
- **与本项目的关系**：**方法参考**。RAG增强型组合配置是本项目benchmark应测试的具体能力场景。模型规模差异下的绩效分层也提示本项目的评测应覆盖不同能力层级的模型。
- 

---

## 第三类（新增）：LLM增强的组合优化方法

---

### 42. LLM Agents for Combinatorial Efficient Frontiers (CCPO)
**标题**：LLM Agents for Combinatorial Efficient Frontiers: Investment Portfolio Optimization
**会议**：arXiv 2026
**链接**：https://arxiv.org/abs/2601.00770

- **Motivation**：基数约束均值方差组合优化（CCPO）是NP难问题，精确求解不可行，启发式算法开发成本高、工作流复杂，需要大量手工工程。
- **Insight**：智能体框架在组合优化中同时具备两种价值：一是自动化复杂工作流，二是进行算法开发（有时超越人类水平），两者叠加可以在不牺牲质量的情况下大幅降低开发成本。
- **Method**：为CCPO问题实现智能体框架，探索多种具体架构；与标准CCPO求解器进行基准测试；评测在经典组合优化问题上的表现。
- **Result**：实现的智能体框架与最先进的启发式算法相当；工作流复杂度和算法开发工作量显著降低；最坏情况下误差在可接受范围内。
- **与本项目的关系**：**方法参考**。LLM-as-optimizer（LLM作为优化器）能力应作为本项目benchmark的独立评测维度——LLM不仅能生成信号或做买卖决策，还能直接求解数学优化问题。这种三层能力评测（决策者/信号生成者/优化器）是本项目框架设计的重要创新点。
- 

---

### 43. LLM-Enhanced Black-Litterman
**标题**：LLM-Enhanced Black-Litterman Portfolio Optimization
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2504.14345

- **Motivation**：Black-Litterman模型通过引入投资者观点解决了传统均值方差优化的敏感性问题，但系统性地生成这些观点仍是关键挑战。
- **Insight**：每个LLM都具有独特且一致的投资风格，这是绩效的主要驱动因素——选择LLM不是在寻找"最好的预测器"，而是在选择一种"投资风格"，其成功取决于该风格与当时市场状态的匹配程度。
- **Method**：LLM收益预测+预测不确定性 → Black-Litterman核心输入（投资者观点和置信度）；S&P 500成分股回测；LLM投资风格一致性分析；市场状态对齐分析。
- **Result**：顶级LLM驱动的组合在绝对收益和风险调整收益上显著优于传统基线；每个LLM均展现出独特且一致的投资风格；风格-市场状态对齐是绩效的主要决定因素。
- **与本项目的关系**：**核心发现**。"投资风格一致性"和"风格-市场状态对齐"是本项目benchmark应引入的新型评测指标——这是所有现有benchmark均未捕捉的维度。LLM的distinct风格使得"选对模型≈选对策略"，而本项目benchmark可以系统评测哪类风格在哪种市场状态下有效。
- 

---

### 44. Regret-Driven Portfolios
**标题**：Regret-Driven Portfolios: LLM-Guided Smart Clustering for Optimal Allocation
**会议**：arXiv 2026
**链接**：https://arxiv.org/abs/2601.17021

- **Motivation**：中长期组合管理中的风险-收益权衡问题持续存在，传统在线学习方法缺乏对市场情绪和下行风险的动态响应能力。
- **Insight**：将在线学习动态（follow-the-leader无悔算法）、市场情绪过滤（LLM情绪信号）、LLM驱动的对冲保护三者结合，形成有原则的无悔组合框架，在风险保护和收益增强上均有实质性改进。
- **Method**：Follow-the-leader基础算法 + LLM情绪信号的交易过滤 + LLM驱动的下行保护对冲策略；与S&P 500买入持有基线比较；构建面向风险厌恶型投资者和机构基金经理的组合。
- **Result**：年化收益超SPY买入持有69%；夏普比率超119%；LLM对冲组件是主要贡献来源；情绪过滤有效减少了过度交易。
- **与本项目的关系**：**方法参考**。LLM-as-hedging-engine（LLM作为对冲引擎）是一种新型能力，本项目benchmark可以纳入评测：LLM能否识别需要对冲保护的时机并执行合理的对冲逻辑？这是一个传统技术分析无法自动化的高级决策能力。
- 

---

### 45. SAPPO
**标题**：Leveraging LLM-based sentiment analysis for portfolio optimization with proximal policy optimization
**会议**：ACL 2025 REALM Workshop
**链接**：https://aclanthology.org/2025.realm-1.12/

- **Motivation**：标准PPO等RL组合优化方法依赖历史价格数据，忽视了投资者情绪的真实影响；仅靠价格信号的策略在情绪驱动行情中会错失重要信号。
- **Insight**：将LLM情绪信号通过情绪加权项整合进PPO优势函数，使配置策略能够同时响应价格变动和市场情绪，两者的结合优于单独任一信号，且存在最优整合强度（λ参数）。
- **Method**：LLaMA 3.3生成每日情绪分数；SAPPO（情绪增强PPO）将情绪信号加权整合进优势函数；三资产组合实验；消融研究和t检验验证；λ=0.1为最优配置。
- **Result**：夏普比率从1.55提升至1.90；最大回撤降低；最优λ=0.1经统计显著性验证（p<0.001）；情绪感知RL策略在风险调整绩效上优于纯价格策略。
- **与本项目的关系**：**方法参考**。SAPPO代表了LLM+RL混合范式，本项目benchmark应评测LLM作为信号提供者（而非直接决策者）用于优化算法的能力。三资产组合是进一步扩展为多异构资产的直接起点。
- 

---

### 46. EFS
**标题**：EFS: Evolutionary Factor Searching for Sparse Portfolio Optimization Using Large Language Models
**会议**：arXiv 2025
**链接**：https://arxiv.org/abs/2507.17211

- **Motivation**：传统alpha因子高度依赖历史收益统计和静态目标，难以适应动态市场状态；人工因子工程成本高、迭代慢、难以规模化。
- **Insight**：将资产选择问题重新表述为LLM生成因子引导的top-m排名任务，进化反馈循环可以自动精炼因子池；语言引导的进化是一种鲁棒且可解释的组合优化范式，在大规模资产池和高波动市场中尤为有效。
- **Method**：LLM自动生成并进化alpha因子；进化反馈循环（生成→评估→精炼）迭代改进因子池；因子以可读公式形式表达；在5个Fama-French基准数据集和3个真实市场数据集（US50、HSI45、CSI300）上评测。
- **Result**：显著优于统计类和优化类基线，在大规模资产池和高波动条件下尤为突出；消融研究证实了提示设计、因子多样性和LLM选择的重要性；语言引导进化验证为可解释性范式。
- **与本项目的关系**：**强相关**。因子生成质量是本项目benchmark应评测的LLM能力之一——LLM能否自动生成有效的、可解释的alpha因子，是量化投资中LLM实用价值的核心测量。EFS的可解释性属性（因子以可读公式表达）也为我们benchmark引入可解释性评测维度提供了具体方法。
- 

---

### 47. SmartFolio
**标题**：Enhancing portfolio optimization via heuristic-guided inverse reinforcement learning with multi-objective reward and graph-based policy learning
**会议**：IJCAI 2025
**链接**：https://dl.acm.org/doi/abs/10.24963/ijcai.2025/1054

- **Motivation**：RL奖励工程难以捕捉市场动态复杂性；传统DRL方法缺乏对金融专业知识的系统整合；现有方法在股间关系建模上不够显式。
- **Insight**：启发式引导的逆强化学习（从专家示范中学习奖励函数）+ 多目标奖励（收益-风险自适应平衡）+ 异构图神经网络（显式建模股间关系）三者协同，优于任何单一组件，是当前非LLM方法的SOTA。
- **Method**：逆RL框架（考虑行业多样化和相关性约束的专家策略生成）+ 多目标奖励优化 + 基于异构图的策略学习（层次注意力机制）；在真实金融市场数据上进行广泛实验和案例分析。
- **Result**：在风险调整收益上优于多项SOTA基线；案例研究证明了收益最大化与风险控制的平衡；异构图GNN是股间关系建模的关键。
- **与本项目的关系**：**方法对比参考**。SmartFolio是当前非LLM组合优化方法的最强基线之一，本项目benchmark应将其（及类似的非LLM SOTA）作为基线，以客观量化LLM方法与最先进数理方法之间的差距或优势。
- 

---

## 快速索引（共47篇）

| 论文 | 与项目关系 | 优先级 |
|------|-----------|--------|
| **原有论文（23篇）** | | |
| MarS (ICLR 2025) | Sandbox核心参考 | ⭐⭐⭐ |
| FinRL-Meta (NeurIPS 2022) | Benchmark框架参考 | ⭐⭐⭐ |
| FinBen (NeurIPS 2024) | 评测维度核心参考 | ⭐⭐⭐ |
| QuantBench (arXiv 2025) | 多资产评测参考 | ⭐⭐⭐ |
| TradingAgents (arXiv 2024) | Agent评测框架 | ⭐⭐⭐ |
| FinAgent (KDD 2024) | 多资产多模态评测 | ⭐⭐ |
| EconAgent (ACL 2024) | Sandbox Agent架构 | ⭐⭐ |
| FinRL (NeurIPS 2021) | 工程实现参考 | ⭐⭐ |
| Qlib (IJCAI 2021) | 数据基础设施参考 | ⭐⭐ |
| DeepTrader (AAAI 2021) | 评测场景设计参考 | ⭐ |
| EarnMore (WWW 2024) | 动态资产池评测 | ⭐ |
| FinMem (AAAI 2024) | 评测指标维度参考 | ⭐ |
| PIXIU (NeurIPS 2023) | 评测任务设计参考 | ⭐ |
| FinQA (EMNLP 2021) | 数值推理评测参考 | ⭐ |
| **新增论文（24篇）** | | |
| Standard Benchmarks Fail (arXiv 2025) | 风险优先评测范式核心依据 | ⭐⭐⭐ |
| FINSABER (arXiv 2025) | 市场状态适应性评测依据 | ⭐⭐⭐ |
| INVESTORBENCH (ACL 2025) | 最近似竞争对手，需明确区分 | ⭐⭐⭐ |
| Look-Ahead-Bench (arXiv 2026) | 时间完整性评测依据 | ⭐⭐⭐ |
| LLM-Enhanced Black-Litterman (arXiv 2025) | 投资风格一致性指标来源 | ⭐⭐⭐ |
| LLM Agents for Investment Management (Survey) | 领域全景与研究空白定位 | ⭐⭐⭐ |
| XFinBench (ACL 2025) | 时序推理与场景规划评测参考 | ⭐⭐ |
| StockBench (arXiv 2024) | QA≠交易技能论据 | ⭐⭐ |
| FinMaster (arXiv 2025) | 全流程错误传播评测参考 | ⭐⭐ |
| When Agents Trade / AMA (arXiv 2024) | 框架>模型backbone发现参考 | ⭐⭐ |
| EFS (arXiv 2025) | 因子生成能力评测参考 | ⭐⭐ |
| LLM-Powered Multi-Agent Crypto (arXiv 2025) | 多智能体专业化分工参考 | ⭐⭐ |
| 3S-Trader (arXiv 2025) | 多股组合构建被评测对象 | ⭐⭐ |
| Enhancing LLM in Asset Selection (ACM 2024) | 信息整合失败关键发现 | ⭐⭐ |
| AlphaSharpe (arXiv 2025) | LLM驱动指标发现参考 | ⭐ |
| GuruAgents (arXiv 2025) | 投资风格一致性测试设计 | ⭐ |
| LLM Agents for CCPO (arXiv 2026) | LLM-as-optimizer能力评测 | ⭐ |
| FinRL Contests (arXiv 2025) | 竞赛式评测工程参考 | ⭐ |
| Democratizing Alpha (ACM 2024) | 另类信息源评测场景 | ⭐ |
| SAPPO (ACL 2025) | LLM+RL混合范式参考 | ⭐ |
| Regret-Driven Portfolios (arXiv 2026) | LLM对冲引擎能力参考 | ⭐ |
| From Text to Returns (arXiv 2024) | RAG增强配置能力参考 | ⭐ |
| BizFinBench (arXiv 2025) | 中文金融跨概念推理参考 | ⭐ |
| SmartFolio (IJCAI 2025) | 非LLM SOTA基线参考 | ⭐ |
