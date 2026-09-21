# 第 9 页：One Model、POMDP 与训练主链

## 1. 这一页回答什么问题

这一页暂时不深入 RL 算法，而是先回答三个更基础的问题：

1. 为什么 Creation Memory 是一个 **POMDP** 问题？
2. 为什么 Memory Update 和 Suggestion Generation 应该由 **同一个模型**完成？
3. 这个 One Model 最终怎样从强模型标注一路训练、强化并蒸馏到线上小模型？

一句话概括：

> 用一份定长 Memory 持续估计用户的隐含兴趣，再让同一个模型既写这份 Memory、又读取它生成创作建议，从而让 Memory 直接为最终业务效果负责。

## 2. 为什么是 POMDP

真实的用户兴趣无法被直接看到。系统每天只能观察用户发来的图片、视频、文本和操作行为，再根据这些局部信号推断用户接下来可能想创作什么。

页面中的技术映射如下：

| POMDP 概念 | 本项目中的含义 |
| --- | --- |
| 隐状态 | 用户真实、持续变化的创作兴趣 |
| 观察 | profile、旧 Memory、当日消息与行为 |
| Belief State | 定长、结构化的 Creation Memory |
| Memory 动作 | 判断是否更新，并覆盖式生成新 Memory |
| Suggestion 动作 | 读取 Memory，生成 1–8 组 `Title + Desc` |
| 奖励 | 精排分、LLM Judge、多样性/安全性和下游 API 质量 |

其中最关键的是：**Memory 不是完整历史，也不是普通摘要，而是在固定容量下保存“对未来创作最有用的信息”的状态。**

## 3. 两阶段在线调用流程

### Stage 01：Write / Memory Update

    profile + 旧 Memory + 当日消息
                    ↓
            update / no_update
                    ↓
            完整结构化新 Memory

主要职责是：

- 区分创作兴趣、操作偏好、无效噪声和风险内容。
- 合并同义兴趣，避免近义条目重复累积。
- 按固定槽位记录兴趣、操作偏好、动机、规避项、可推荐方向、预算与状态。
- 根据跨日复现、频率、最近性和画像一致性更新兴趣强度。
- 谨慎删除旧兴趣，保证长期状态稳定且可追溯。

### Stage 02：Read / Suggestion Generation

    profile + Memory + context + recent titles + mode
                             ↓
                  候选生成、去重与选量
                             ↓
                     1–8 组 Title + Desc

主要职责是：

- 融合画像与真实行为，冲突时优先使用真实行为。
- 从 Memory、profile、操作偏好、探索候选等来源构建候选池。
- 按质量动态决定输出数量，不为了凑数生成重复提案。
- 控制节日、历史标题、文生图/图生图模式和内容安全约束。
- 把抽象兴趣写成具体、可执行、可直接投放的视觉创作描述。

## 4. 为什么必须是 One Model

这里的“One Model”不代表把两个步骤合并成一次调用。它仍然是两次调用，但两次调用共享参数、数据分布和最终目标。页面给出三条理由。

### 4.1 终端信用：Memory 要为最终创作效果负责

如果把 Writer 和 Reader 分开训练，Writer 往往只能学习“什么是好摘要”。但一份语言流畅的摘要，不一定保留了生成创作建议真正需要的画风、题材、新鲜度和可 Prompt 化细节。

One Model 让同一策略同时优化 Memory 写入与最终生成：

$$
\nabla_\theta J=\mathbb E\left[\sum_{t=1}^{T}A_t\nabla_\theta\log\pi_\theta(m_t\mid m_{t-1},x_t)+A_{gen}\nabla_\theta\log\pi_\theta(y\mid m_T)\right]
$$

因此，最终 `Title / Desc` 的效果可以反过来定义“什么值得写进 Memory”。

### 4.2 协议一致：写入方和读取方使用同一种语言

两个独立模型像两个陌生人传纸条：Writer 可能形成自己的编码习惯，而 Reader 未必真正理解或使用这些信号。

共享参数相当于模型给自己写便签，使 Memory 的编码和解码习惯天然一致。它不能保证内容一定正确，但能显著减少跨模型的协议错配。页面中的 Same Model Cross-Play 实验也用于验证这一判断。

### 4.3 长程稳定：单轮小错会被后续更新放大

每次 Memory 都是在上一版基础上覆盖式重写。如果单轮存在遗漏或误写，错误会改变下一轮输入，并继续传播。页面用下面的简化模型说明复利风险：

$$
I_T \approx I_0(1-\varepsilon)^T
$$

- $I_T$：经过 $T$ 轮后仍可使用的信息。
- $\varepsilon$：每轮近似错配率。
- 若 $\varepsilon=5\%$，10 轮后约剩 60%，20 轮后约剩 36%。

这只是解释风险的简化模型，不是严格理论定理。它强调的是：**长程 Memory 的问题不能只看单轮质量，必须在训练中显式处理跨轮信用与状态污染。**

## 5. 实际训练主链

    Gemini 3 Pro 标注
            ↓
    Qwen3-32B SFT 冷启动
            ↓
    Two-Stage RL：Short Horizon → Long Horizon
            ↓
    Qwen3-32B RL Teacher
            ↓
    Qwen3-8B OPD 在线模型

### 5.1 SFT 冷启动

- Gemini 3 Pro 生成结构化候选答案，再通过 Judge 拒绝采样清洗。
- 使用真实分布输入和严格输出契约。
- 当前展示的数据量为 `76,320 train / 2,400 validation`。
- 训练共 2,385 个 optimizer steps，Train / Validation Loss 同步收敛。
- Loss 收敛只说明冷启动稳定，不等于业务效果已经最优。

### 5.2 两阶段 RL

| 阶段 | 轨迹范围 | 作用 |
| --- | --- | --- |
| Stage I：Short Horizon | `<20 turn`，平均 4.8 turn | 先学会稳定完成 `Memory Update → Proposal → Reward` 闭环，降低初期方差和资源压力。 |
| Stage II：Long Horizon | `>30 turn`，平均 38.3 turn | 再面对早期信息遗忘、状态污染和长期偏好保留问题。 |

这里按照真实轨迹长度分为两个阶段，而不是机械地使用固定的 4→8→12 长度课程。

### 5.3 OPD 蒸馏到 8B

- 8B Student 使用自己的策略进行 on-policy 采样。
- 32B RL Teacher 只在 Student 实际访问到的状态上提供逐 token 分布监督。
- 相比稀疏的轨迹奖励，每个 token 都可以获得训练信号。
- Qwen3-32B 与 Qwen3-8B 词表一致，页面配置使用 `top-K=64`。

需要注意：页面明确把 OPD 的最终收益视为**仍需验证的问题**。计划中的四方对照 `8B-SFT / 8B-Direct-RL / 8B-OPD / 8B-SFT→OPD` 尚未完成，遗忘与能力保持测试也仍需补齐。

## 6. 数据、奖励与训练基础设施

### Data

- 覆盖 30 万真实用户：20 万用户的 3 个月数据、10 万用户的 6 个月数据。
- profile 将基础画像、长期兴趣和 7 天短期兴趣压缩到 512 token。
- 进行 NFKC、全半角、零宽字符、重复片段等清洗，并把异常写入审计 JSONL。
- SFT 从 1 万 UID 采样，使用 10 类硬过滤、跨样本去重与严格 train/validation 隔离。
- RL 数据独立构造，过滤组内 advantage 差异过小的样本，并持续审计 reward hacking。

### Reward

奖励来自三个真实业务目标：

1. **Ranker**：固定 Top-K，评价候选的校准效用，避免靠多生成候选刷最高分。
2. **LLM Judge**：评价多样性、新颖度、安全性和侵权风险。
3. **Downstream API**：让生成 Prompt 接受真实下游承接质量的评分。

聚合时先对每路 Reward 单独计算 Advantage，再按业务权重加权，避免不同量纲和方差互相污染。

### Infra

- 训练栈：slime + Ray + Megatron-LM + SGLang。
- 硬件：`32 × A100 80GB`。
- 并行：`TP=8`、`DP=4`、`PP=1`、`CP=1`。
- 序列长度：8192 tokens，动态 microbatch。
- 优化器：Adam，学习率从 `1e-6` cosine decay 到 `1e-7`，3% warmup。

## 7. 这一页最终想让观众记住什么

> One Model 的价值不只是少部署一个模型，而是让“写 Memory、读 Memory、看最终效果”成为同一条可优化的闭环。

它同时解决三件事：

- **目标一致**：Memory 直接为最终创作结果服务。
- **协议一致**：Writer 与 Reader 共享编码、解码习惯。
- **长程一致**：多轮状态更新能接受跨步信用，而不是只优化单步摘要质量。

下一页继续回答：这条长程闭环具体应该怎样分配 RL 信用。

