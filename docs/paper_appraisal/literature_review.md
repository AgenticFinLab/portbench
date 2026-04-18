# 文献综述：大语言模型投资组合管理能力的评测基准研究

> 综述时间：2026-04-14
> 项目背景：构建包含多种异构资产的数据集与sandbox环境，系统评测大语言模型的投资组合管理能力

---

## 1. 引言

近年来，大语言模型（LLM）在金融领域的应用研究蓬勃发展，从金融文本理解到智能交易决策，LLM展现出巨大的应用潜力。然而，如何科学、系统地评测LLM在**投资组合管理**这一金融核心任务上的能力，仍是一个开放性问题。现有评测范式存在四大系统性缺陷：（1）资产范围局限于单一类别（通常为股票），无法评测跨异构资产的组合决策；（2）以收益指标为核心，忽视安全性与鲁棒性，产生虚假可靠性幻觉；（3）缺乏时间完整性控制，前视偏差导致评测结果虚高；（4）评测孤立子任务，忽视完整管理流程中的错误传播效应。

本综述围绕**异构资产数据集构建**和**LLM投资组合能力评测**两大核心问题，系统梳理了2021-2026年间发表在NeurIPS、ICLR、ACL、EMNLP、KDD、AAAI等顶会的47篇代表性论文，从金融AI评测基准、LLM金融智能体、LLM增强的组合优化方法、强化学习组合优化和市场仿真环境五个维度展开分析。通过系统梳理，我们识别出五个具体研究空白，并提出了一个多维创新框架，为构建面向投资组合管理的LLM评测benchmark提供理论基础和技术参考。

---

## 2. 金融AI评测基准的发展脉络

### 2.1 从单一任务到多任务综合评测

金融AI评测经历了从单任务专项评测到多任务综合基准的演进过程。

早期的评测工作主要聚焦于单一任务维度。**FinQA**（EMNLP 2021）开创性地将金融数值推理作为独立评测任务，通过8,281个专家标注的问答对，揭示了LLM在金融数值推理上与人类专家存在巨大差距（最优模型约50%准确率 vs 人类91%）。其续作**ConvFinQA**（EMNLP 2022）将评测扩展至多轮对话场景，发现了链式数值推理追踪是LLM的核心瓶颈。这些工作奠定了金融NLP评测的基础，但评测维度单一，无法全面衡量模型的金融综合能力。

随着LLM技术的发展，综合性评测基准开始涌现。**PIXIU**（NeurIPS 2023）首次将金融LLM评测系统化为"理解-推理-决策"三个层次，构建了覆盖情感分析、NER、问答、股价预测等9类任务的FLARE基准。**FinEval**（NAACL 2025）则聚焦中文金融场景，从学术知识、行业实践、监管合规四维度构建了8,351道题的评测体系。

### 2.2 面向投资决策的高级基准

最新的评测工作开始关注LLM在投资决策层面的能力。**FinBen**（NeurIPS 2024）代表了当前金融LLM评测的较高水平，整合42个数据集、覆盖24类金融任务，**首次将基于智能体的股票交易纳入标准化评测**，揭示了即使是GPT-4在端到端交易任务上也显著弱于人类专家。**QuantBench**（arXiv 2025）则从量化投资的全流程视角出发，构建覆盖因子挖掘、信号生成、组合构建、风险控制和回测的端到端评测框架，首次实现跨多资产类别的统一评测。

在复杂推理评测方向，**XFinBench**（ACL 2025）将LLM金融能力分解为五个核心维度：术语理解、时序推理、未来预测、场景规划和数值建模。对18个前沿模型的评测表明，o1的最优准确率为67.3%，仍落后人类专家12.5%，时序推理和场景规划是最大能力缺口。**FinMaster**（arXiv 2025）则将全流程金融工作流纳入评测，发现了系统性错误传播效应——基础任务准确率90%+在复杂多步骤任务中骤降至40%，揭示了孤立子任务评测的严重局限性。**BizFinBench**（arXiv 2025）在中文金融场景下进一步验证了跨概念推理是LLM的普遍瓶颈。

### 2.3 多产品智能体评测

**INVESTORBENCH**（ACL 2025）是首个覆盖股票、加密货币和ETF三类产品的LLM智能体评测基准，对13种LLM的评测揭示了显著的产品类别特异性——没有任何单一模型在所有产品类别上均表现领先。**StockBench**（arXiv 2024）通过无污染的多月股票交易评测，证明了"QA能力不能预测交易能力"——语言理解与实际交易技能是两种本质不同的能力。**When Agents Trade / AMA**（arXiv 2024）建立了首个跨加密货币和股票市场的终身实时智能体benchmark，揭示了关键发现：智能体框架架构对行为模式的影响远大于模型backbone，框架选择是首要性能驱动因素。

### 2.4 新兴评测范式：2024-2026的四条研究主线

2024年至2026年间，金融AI评测领域涌现出四条清晰的研究主线，共同指向评测范式的深层变革：

**主线A：真实交易评测运动**（StockBench、AMA、FINSABER）。这一主线的核心发现是：性能在静态问答上的表现无法预测在动态交易中的实际能力，短时间窗口的评测系统性高估了LLM策略的有效性。**FINSABER**（arXiv 2025）在20年回测和100+品种的大规模验证中发现，LLM的所谓"优势"在更长时间段和更广泛资产域下显著消退，且存在系统性市场状态失配（牛市过保守、熊市过激进）。这一发现有力地论证了必须进行分市场状态的细粒度评测。

**主线B：复杂推理评测运动**（XFinBench、FinMaster、BizFinBench）。这一主线证明，LLM的推理链在复杂金融场景下高度脆弱：时序推理和场景规划是系统性弱点，多步骤流程中的错误传播会导致能力骤降，跨概念/跨资产的信息整合是普遍难点。

**主线C：风险优先评测运动**（Standard Benchmarks Fail、Look-Ahead-Bench）。**Standard Benchmarks Fail**（arXiv 2025）明确提出，收益指标给出了虚假可靠性幻觉，安全预算应作为首要评测标准，并通过三层压力测试框架（模型层/工作流层/系统层）揭示了常规评测完全遗漏的隐藏弱点。**Look-Ahead-Bench**（arXiv 2026）则证明标准LLM存在显著前视偏差，alpha衰减（而非问答准确率）才是衡量时间完整性的正确方法。这两项工作共同构成了"评测范式必须转变"的强有力论据。

**主线D：综述与体系化视角**（LLM Agents for Investment Management survey、FinRL Contests）。LLM投资管理综述（ACM MM 2024）提供了最完整的研究图谱，明确指出鲁棒性和可解释性是当前最关键的未解挑战；FinRL竞赛系列（arXiv 2025）则通过竞赛格式识别了非平稳性、低信噪比和市场摩擦三大核心工程挑战，同时确立了"LLM工程化信号"作为新兴独立任务类别的地位。

---

## 3. LLM金融智能体研究

### 3.1 单体智能体

LLM在金融交易中的最早应用形态是单体智能体。**FinGPT**（NeurIPS/IJCAI 2023）和**BloombergGPT**（arXiv 2023）分别从开源和专有两个方向构建了金融LLM的基础能力。在此基础上，**FinMem**（AAAI 2024）引入了分层记忆机制，使交易Agent能够从历史经验中学习，显著提升了决策稳定性。**FinAgent**（KDD 2024）则进一步整合多模态信息（数值、文本、K线图），并通过工具调用获取实时市场数据，在多类资产上展现了更强的泛化能力。

### 3.2 多智能体协作系统

受真实投资机构组织架构的启发，多智能体协作系统成为新的研究热点。**TradingAgents**（arXiv 2024）模拟了专业交易公司的分工结构，将基本面分析师、技术分析师、风险控制员、基金经理等角色分配给不同的LLM Agent，通过结构化消息传递实现协作决策，在夏普比率等指标上显著优于单体模型。**EconAgent**（ACL 2024）则在宏观经济仿真领域展现了类似思路的有效性。

### 3.3 多资产与多智能体协同的新进展（2024-2026）

2024年至2026年间，多资产与多智能体方向出现了重要进展，呈现出几个共同主题：

**智能体框架的结构性优势**。AMA（arXiv 2024）通过跨模型跨框架的大规模对比实验发现，智能体框架架构对行为模式的决定性作用超过模型backbone，这一发现对benchmark设计具有根本性启示：我们不能仅仅评测模型能力，还必须评测系统架构选择。

**专业化分工的必要性**。**LLM-Powered Multi-Agent Crypto Portfolio**（arXiv 2025）验证了按子任务专业化分工（数据分析、文献整合、投资决策）的多智能体系统优于通才型智能体，团队内部和团队间双层协作机制进一步提升了预测置信度和可解释性。**3S-Trader**（arXiv 2025）则证明了将组合构建分解为评分、策略、选股三个显式步骤（免训练框架），可以实现在DJIA成分股上131.83%的累计收益。

**投资风格的LLM可测性**。**GuruAgents**（arXiv 2025）将传奇投资者的哲学编码为LLM提示，发现不同哲学确实产生截然不同但内在一致的行为模式，证明"投资风格一致性"是LLM真实可测的属性。**Democratizing Alpha**（ACM 2024）则证明LLM可以从非结构化视频媒体中提取经济上有意义的投资逻辑，"另类信息源"处理能力是当前benchmark的空白。

**信息整合的根本性局限**。**Enhancing LLM Performance in Asset Selection**（ACM 2024）揭示了一个反直觉但重要的发现：将量化预测信号（OLS/XGBoost输出）提供给LLM反而会降低其绩效，LLM的信息整合能力存在根本性局限，这是本项目benchmark必须专项评测的能力缺陷。

### 3.4 对本项目评测设计的启示

上述工作揭示了投资组合管理能力的多维度性：

- **信息处理能力**：对多源信息（新闻、财报、价格数据、另类信息源）的整合分析
- **决策一致性**：在不同市场状态下保持稳定的决策框架和投资风格
- **风险感知能力**：对组合风险（波动率、回撤、相关性）的准确评估
- **协作/综合能力**：将多方分析意见综合为统一的组合决策
- **多资产理解能力**：理解不同资产类别的风险收益特征和相关性结构
- **量化信号整合能力**：正确处理和使用外部定量预测，而非因整合失败导致性能退化
- **框架适应能力**：在不同系统架构下保持稳定且合理的决策行为

本项目的评测框架应全面覆盖上述能力维度。

---

## 4. 投资组合优化方法

### 4.1 从传统方法到深度强化学习

投资组合优化是金融学的经典问题，传统方法（如马科维茨均值-方差模型）依赖对资产收益分布的参数化假设，在处理实际市场的非线性、非平稳特性时存在局限。深度强化学习方法的引入为组合优化带来了新的可能。

**DeepTrader**（AAAI 2021）通过双流网络（资产评分流 + 市场状态流）实现风险-收益平衡的组合管理，验证了市场状态感知对组合管理的重要性。**TRA**（KDD 2021）利用最优传输理论将不同交易模式的样本自适应路由到专门的预测器，解决了"一刀切"模型在异质资产上的精度损失问题。**FreQuant**（KDD 2024）从频率分解的视角出发，将金融信号分解为高、中、低频成分，分别优化后动态融合，在多时间尺度的组合管理上取得了突破。

### 4.2 泛化性与动态资产池

**EarnMore**（WWW 2024）关注了投资组合管理中资产池动态变化的问题，通过可遮蔽的股票表示方法，实现了对任意子集的泛化。**EarnHFT**（AAAI 2024）则展示了层次化决策架构（宏观层战略方向 + 微观层执行优化）的有效性。**SmartFolio**（IJCAI 2025）将启发式引导的逆强化学习、多目标奖励优化和异构图GNN结合，成为当前非LLM组合优化方法的SOTA，可作为本项目benchmark中非LLM方法的最强基线。

### 4.3 LLM增强的组合优化新范式（2024-2026）

2024年至2026年间，LLM与传统金融数学优化方法的深度融合形成了一种新范式，LLM不再只是信号生成者或决策者，而是直接参与数学优化管道的各个环节：

**LLM作为优化器**。**LLM Agents for CCPO**（arXiv 2026）证明LLM智能体可以直接求解基数约束均值方差组合优化（NP难）问题，在基准测试中与最先进的启发式算法相当，同时大幅降低工作流复杂度。这确立了"LLM-as-optimizer"作为独立的能力评测类别的必要性。

**LLM作为贝叶斯观点生成器**。**LLM-Enhanced Black-Litterman**（arXiv 2025）将LLM的收益预测及其不确定性转化为Black-Litterman模型的投资者观点和置信度，实现了LLM与经典金融优化理论的有机融合。关键发现是：每个LLM都具有独特且一致的投资风格，风格与市场状态的对齐程度是绩效的主要决定因素——这一发现将"投资风格一致性"确立为一个新型且可测量的LLM属性。

**LLM作为进化引擎**。**AlphaSharpe**（arXiv 2025）通过LLM驱动的指标进化（交叉、变异、评估循环）发现了比传统夏普比率预测力高3倍的风险调整指标。**EFS**（arXiv 2025）将同样的进化思路应用于alpha因子生成，在稀疏组合优化中显著超越统计类和优化类基线，且因子以可读公式形式表达，兼顾了性能与可解释性。

**LLM作为RL信号增强器**。**SAPPO**（ACL 2025）通过LLaMA情绪信号加权的PPO优势函数，将夏普比率从1.55提升至1.90；**Regret-Driven Portfolios**（arXiv 2026）将在线学习（follow-the-leader）、LLM情绪过滤和LLM对冲保护三者结合，实现年化收益超SPY达69%、夏普超119%的突破性结果。

上述工作共同确立了"LLM在组合优化中的三层角色"：（1）直接决策者；（2）信号/观点提供者；（3）优化器/算法开发者。本项目benchmark必须分别评测这三种角色下的能力。

### 4.4 对本项目的启示

1. **异构资产处理**：TRA揭示的不同交易模式需要专门处理器的思路，提示本项目的评测应包含具有不同统计特征的资产类别
2. **动态场景评测**：EarnMore启发的动态资产池场景，应纳入本项目评测框架
3. **多频率评测**：FreQuant提示应设计不同投资期限（日内、周度、月度、季度）的评测场景
4. **三层能力评测**：LLM-as-optimizer、LLM-as-signal-provider、LLM-as-decision-maker三层能力应在benchmark中分别设计评测任务
5. **风格-状态匹配**：Black-Litterman论文的发现提示应设计不同市场状态下的评测，揭示LLM投资风格与市场状态的匹配关系

---

## 5. 金融市场仿真环境

### 5.1 从规则驱动到生成式仿真

仿真环境是评测LLM投资组合能力的基础设施。早期工作主要基于OpenAI Gym接口构建简化的市场环境。**FinRL**（NeurIPS 2021）和**Qlib**（IJCAI 2021）分别提供了标准化的RL交易环境和AI量化投资平台，但它们的市场模拟基于历史数据回放或简化的统计模型，与真实市场存在显著差异。**mbt-gym**（ICAIF 2023）从市场微结构角度改进了仿真环境，但保真度仍受限于模型假设。

### 5.2 生成式基础模型驱动的高保真仿真

**MarS**（ICLR 2025）代表了金融市场仿真技术的范式性突破。其核心创新是将订单流视为"语言Token"序列，用大规模生成式模型（Large Market Model, LMM）学习市场的生成机制，从而创建高保真、可交互、可控的市场仿真器。MarS的仿真市场在收益分布、波动聚集、量价相关性等统计特性上与真实市场高度一致，显著缩小了sim-to-real gap。**FinRL-Meta**（NeurIPS 2022）则从评测基础设施的角度提供了系统化的解决方案。

### 5.3 竞赛式评测对环境设计的启示

**FinRL Contests**（arXiv 2025）通过2023-2025年系列竞赛（200+参与者，100+机构）总结了标准化评测环境的关键要素：GPU优化的并行市场环境、集成学习工具包、清晰的任务定义和公平的评测协议。其识别出的核心挑战——非平稳性处理、低信噪比、市场摩擦建模——直接指向本项目sandbox的设计优先级。"LLM工程化信号"作为新兴独立任务类别在竞赛中的涌现，进一步验证了本项目benchmark评测此类能力的及时性。

### 5.4 对本项目Sandbox构建的启示

1. **仿真架构选择**：MarS的生成式仿真框架可作为本项目sandbox的核心引擎，解决传统回放式仿真的保真度不足问题
2. **多资产扩展**：当前MarS主要覆盖单一市场，本项目需要将仿真扩展至股票、债券、商品、加密货币等多资产类别
3. **环境接口标准化**：FinRL的gym接口设计和FinRL-Meta的数据管道可作为sandbox工程实现的基础
4. **可控性设计**：MarS支持注入外生冲击（政策变化、市场事件），本项目可利用此能力设计多市场状态的压力测试场景

---

## 6. 研究空白与本项目定位

通过对上述47篇代表性工作的系统梳理，我们识别出以下五个关键研究空白，这五个空白共同构成了本项目的多维创新框架。

### 6.1 研究空白一：缺乏面向多异构资产组合管理的LLM评测基准

现有金融LLM评测基准（FinBen、INVESTORBENCH、QuantBench等）主要评测NLP能力和单资产交易能力，或最多进行单一产品类别的评测（如INVESTORBENCH对股票/加密货币/ETF的分类评测）。但所有现有工作均缺乏对"投资组合管理"这一核心金融能力的系统评测，具体表现为：

- **资产多样性不足**：绝大多数基准仅覆盖股票，缺乏债券、商品、外汇、加密货币等资产的统一评测
- **组合层面评测缺失**：评测聚焦于单资产的买入/卖出决策，而非多资产的权重分配、相关性管理、风险预算等组合层面决策
- **跨资产理解未评测**：LLM是否理解不同资产类别之间的相关性结构（如股债负相关、黄金避险属性）尚未被系统评测

INVESTORBENCH虽覆盖多类产品，但每类产品仍以单资产决策为核心，而非真正的跨资产组合管理。本项目将填补这一空白，构建首个系统评测多异构资产组合管理能力的LLM benchmark。

### 6.2 研究空白二：缺乏风险优先的评测范式

**Standard Benchmarks Fail**（arXiv 2025）明确指出，以收益为核心的评测标准制造了虚假可靠性幻觉——所有被测试的智能体均存在常规基准完全遗漏的隐藏弱点（幻觉事实、过期数据敏感性、对抗提示脆弱性）。这一问题在多资产组合管理中尤为严峻：一个能够获得高收益但在极端市场压力下产生幻觉或做出灾难性决策的LLM，根本不适合部署。

然而，目前没有任何多资产组合benchmark将安全预算（safety budget）合规率、压力场景通过率或鲁棒性测试作为**一级评测标准**。本项目将率先建立以风险为优先的评测范式：只有满足风险合规要求的智能体才能进入绩效排名，而非仅仅将风险指标作为附属报告项。

### 6.3 研究空白三：缺乏时间完整性与前视偏差的系统控制

**Look-Ahead-Bench**（arXiv 2026）揭示了标准LLM中普遍存在显著的前视偏差，且现有方法（Q&A准确率）无法有效检测此类偏差，alpha衰减分析才是正确的衡量方式。这一问题对多资产组合benchmark尤为关键：由信息泄漏而非真实预测能力产生的高收益绩效，会严重误导LLM能力判断和部署决策。

现有多资产或组合管理基准均未系统审计前视偏差。本项目将强制执行Point-in-Time数据约束，并以alpha衰减指标系统评测时间完整性，确保报告的绩效反映真实预测能力。

### 6.4 研究空白四：缺乏市场状态适应性的系统评测

**FINSABER**（arXiv 2025）在20年回测和100+品种的大规模研究中发现，LLM策略在牛市中系统性过保守、在熊市中系统性过激进，且这种状态失配在短期评测中无法被检测。**DeepTrader**（AAAI 2021）也早已在RL研究中验证了市场状态感知对组合管理的关键作用。

然而，现有LLM评测benchmark（包括INVESTORBENCH、StockBench、AMA）均不进行分市场状态的绩效分解，也不将不同市场状态下的适应性作为独立评测维度。本项目benchmark将明确将测试数据划分为牛市/熊市/横盘/危机等市场状态，并分状态报告绩效，从而系统揭示LLM策略的状态感知能力与失配风险。

### 6.5 研究空白五：缺乏完整投资组合管理流程的端到端评测

**FinMaster**（arXiv 2025）通过183个跨难度任务证明了错误传播的系统性——基础任务准确率在复杂多步骤场景中骤降。**QuantBench**（arXiv 2025）虽然设计了全流程量化投资评测，但其评测对象是量化方法而非LLM智能体的端到端工作流能力。**Enhancing LLM in Asset Selection**（ACM 2024）揭示的量化信号整合失败，也只有在完整流程评测中才能完整暴露。

现有LLM金融benchmark均评测孤立子任务，无法捕捉从市场数据解读→信号生成→组合优化→执行模拟→风险监控的完整流程中的能力退化。本项目将引入全流程评测任务，测量各阶段的错误传播，揭示LLM在端到端场景下与孤立任务下的能力差异。

### 6.6 本项目的综合研究定位

基于上述五个研究空白，本项目旨在构建一个在多个维度上超越现有工作的LLM投资组合管理benchmark。以下定位矩阵展示了本项目与主要相关工作的关键差异：

| 评测维度 | FinBen | INVESTORBENCH | QuantBench | AMA | **本项目** |
|---------|--------|--------------|-----------|-----|-----------|
| 资产覆盖 | 单资产（股票为主） | 多产品单资产决策 | 多资产类别 | 股票+加密货币 | **多异构资产组合管理** |
| 评测范式 | 收益优先 | 收益优先 | 收益优先 | 收益优先 | **风险优先** |
| 时间完整性 | 未控制 | 未控制 | 未控制 | 部分控制 | **系统PiT审计** |
| 市场状态 | 未分状态 | 未分状态 | 未分状态 | 未分状态 | **分状态评测** |
| 流程深度 | 孤立任务 | 孤立任务 | 全流程（量化方法） | 孤立任务 | **端到端流程+错误传播** |

本项目的具体贡献包括：

1. **构建多异构资产数据集**：涵盖股票、债券、商品、外汇、加密货币等多类资产的价格数据、基本面数据、新闻文本等多模态信息，填补资产多样性空白
2. **搭建高保真Sandbox环境**：基于MarS生成式仿真框架构建多资产交互的市场模拟环境，支持分市场状态的策略评测和风险压力测试
3. **建立风险优先评测框架**：以安全预算合规率和压力场景通过率作为一级评测标准，区别于所有现有以收益为核心的benchmark
4. **引入时间完整性审计**：强制Point-in-Time数据约束，以alpha衰减方法系统检测前视偏差
5. **实现分市场状态的全流程评测**：在牛/熊/横盘/危机等多种市场状态下，评测LLM投资组合管理的完整流程能力，测量各阶段错误传播

---

## 7. 总结

本综述从金融AI评测基准、LLM金融智能体、LLM增强的组合优化方法、强化学习组合优化和市场仿真环境五个维度系统梳理了47篇顶会论文的核心贡献。

三个关键趋势从文献中清晰涌现：**第一**，评测社区正在形成共识——QA能力无法预测实际交易/投资能力，针对真实市场动态的评测是必要的；**第二**，风险优先、时间完整、分状态的多维评测正在成为新的高标准，收益独占评测被学界公认为不充分；**第三**，LLM在金融优化中的角色已从"决策者"扩展为"信号提供者"和"优化器"，三层能力评测是完整benchmark的必要条件。

然而，上述进展的评测对象仍局限于单资产交易或单一产品类别，尚无benchmark将这一新兴评测共识系统应用于多异构资产的组合管理场景。**本项目将在此研究空白上开展工作，构建首个同时满足风险优先、时间完整、分状态评测、端到端流程和多异构资产覆盖五大标准的LLM投资组合管理benchmark**，为学术界和工业界提供系统化、标准化的评测工具。

---

## 参考文献

**原有文献（23篇）**

1. Liu et al. FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial RL. NeurIPS 2022.
2. Xie et al. PIXIU: A Comprehensive Benchmark, Instruction Dataset and LLM for Finance. NeurIPS 2023.
3. Xie et al. The FinBen: An Holistic Financial Benchmark for LLMs. NeurIPS 2024.
4. Zhang et al. FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark. NAACL 2025.
5. Chen et al. FinQA: A Dataset of Numerical Reasoning over Financial Data. EMNLP 2021.
6. Chen et al. ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance QA. EMNLP 2022.
7. Wang et al. QuantBench: Benchmarking AI Methods for Quantitative Investment. arXiv 2025.
8. Li et al. EconAgent: LLM-Empowered Agents for Simulating Macroeconomic Activities. ACL 2024.
9. Yang et al. FinTral: A Family of GPT-4 Level Multimodal Financial LLMs. ACL 2024 Findings.
10. Zhang et al. FinAgent: A Multimodal Foundation Agent for Financial Trading. KDD 2024.
11. Xiao et al. TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv 2024.
12. Yang et al. FinGPT: Open-Source Financial Large Language Models. NeurIPS/IJCAI 2023.
13. Yu et al. FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory. AAAI 2024.
14. Wu et al. BloombergGPT: A Large Language Model for Finance. arXiv 2023.
15. Wang et al. DeepTrader: A DRL Approach for Risk-Return Balanced Portfolio Management. AAAI 2021.
16. Zhang et al. EarnMore: RL with Maskable Stock Representation for Portfolio Management. WWW 2024.
17. Li et al. FreQuant: RL-based Adaptive Portfolio with Multi-frequency Decomposition. KDD 2024.
18. Lin et al. TRA: Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor. KDD 2021.
19. Qin et al. EarnHFT: Efficient Hierarchical RL for High Frequency Trading. AAAI 2024.
20. Li et al. MarS: A Financial Market Simulation Engine Powered by Generative Foundation Model. ICLR 2025.
21. Liu et al. FinRL: Deep RL Framework for Quantitative Finance. NeurIPS 2021.
22. Yang et al. Qlib: An AI-oriented Quantitative Investment Platform. IJCAI 2021.
23. Jerome et al. mbt-gym: RL for Model-Based Limit Order Book Trading. ICAIF 2023.

**新增文献（24篇）**

24. Dao et al. Large Language Model Agents for Investment Management: Foundations, Benchmarks, and Research Frontiers. ACM MM 2024.
25. Yue et al. XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning. ACL 2025 Findings.
26. Nie et al. InvestorBench: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent. ACL 2025.
27. Anonymous. StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets? arXiv 2024.
28. Anonymous. Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? (FINSABER). arXiv 2025.
29. Anonymous. BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs. arXiv 2025.
30. Banerjee et al. Standard Benchmarks Fail — Auditing LLM Agents in Finance Must Prioritize Risk. arXiv 2025.
31. Anonymous. Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance. arXiv 2026.
32. Anonymous. FinMaster: A Holistic Benchmark for Mastering Full-Pipeline Financial Workflows with LLMs. arXiv 2025.
33. Liu et al. FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents. arXiv 2025.
34. Anonymous. 3S-Trader: A Multi-LLM Framework for Adaptive Stock Scoring, Strategy, and Selection in Portfolio Optimization. arXiv 2025.
35. Anonymous. When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents. arXiv 2024.
36. Anonymous. AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics. arXiv 2025.
37. Anonymous. Enhancing LLM Performance in Asset Selection: Investigating the Integration Challenges of Traditional Quantitative Signals. ACM MM 2024.
38. Anonymous. LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management. arXiv 2025.
39. Anonymous. Democratizing Alpha: LLM-Driven Portfolio Construction for Retail Investors Using Public Financial Media. ACM MM 2024.
40. Anonymous. GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents. arXiv 2025.
41. Anonymous. From Text to Returns: Using Large Language Models for Mutual Fund Portfolio Optimization and Risk-Adjusted Allocation. arXiv 2024.
42. Anonymous. LLM Agents for Combinatorial Efficient Frontiers: Investment Portfolio Optimization. arXiv 2026.
43. Anonymous. LLM-Enhanced Black-Litterman Portfolio Optimization. arXiv 2025.
44. Anonymous. Regret-Driven Portfolios: LLM-Guided Smart Clustering for Optimal Allocation. arXiv 2026.
45. Anonymous. Leveraging LLM-based Sentiment Analysis for Portfolio Optimization with Proximal Policy Optimization (SAPPO). ACL 2025 REALM Workshop.
46. Anonymous. EFS: Evolutionary Factor Searching for Sparse Portfolio Optimization Using Large Language Models. arXiv 2025.
47. Zhang et al. Enhancing Portfolio Optimization via Heuristic-Guided Inverse Reinforcement Learning with Multi-Objective Reward and Graph-Based Policy Learning (SmartFolio). IJCAI 2025.
