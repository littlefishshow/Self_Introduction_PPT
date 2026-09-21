# 第 9–10 页内容总结

> 内容依据当前 PPT 主页面、演讲者备注以及可点击展开的补充面板整理。页码以页面底部实际显示的 **9 / 10** 为准。

## 两页共同讲述的故事

这两页围绕同一个问题展开：**怎样让一个模型在多轮用户行为中持续维护 Creation Memory，并让这份 Memory 真正帮助下游生成个性化创作建议。**

整体逻辑是：

    第 9 页：先定义任务与训练主链
    为什么这是 POMDP？为什么 Memory Writer 和 Suggestion Generator 要用同一个模型？
                             ↓
    第 10 页：再解决最关键的 RL 难题
    长轨迹中，最终效果应该怎样准确地归因给每一次 Memory 更新？

## 一页一结论

| 页码 | 页面主题 | 核心结论 |
| --- | --- | --- |
| 第 9 页 | One Model Suggestion Generation | Memory 是不可见用户兴趣的有限容量状态；同一模型同时负责写入和读取，才能让终端回报、读写协议和长程状态形成闭环。 |
| 第 10 页 | Random-Trajectory-Window RL | 单步训练看不到未来，整轨迹训练又太贵且信用混乱；随机窗口在固定成本下兼顾真实长程状态、密集奖励与因果信用。 |

## 核心名词

- **Memory**：对用户创作兴趣的定长、结构化表示，是模型在多轮交互中的状态载体。
- **Suggestion**：模型读取 profile、Memory 等信息后生成的 1–8 组个性化 `Title + Desc`。
- **POMDP**：真实用户兴趣不可直接观测，模型只能根据每天的新消息持续更新对用户兴趣的估计。
- **One Model**：Memory 更新与 Suggestion 生成是两次独立调用，但共享同一套模型参数和最终业务目标。
- **信用分配**：判断最终奖励应该归因给哪些 Memory 更新或 Suggestion 生成动作。
- **OPD**：由在线小模型自己采样，RL 后的大模型 Teacher 在这些真实访问到的 token 状态上提供逐 token 蒸馏信号。
- **GMPO**：以几何平均聚合 token importance ratio，降低长输出中异常 token 对整条更新的影响。

## 建议讲述顺序

1. 用户真实兴趣看不见，Memory 是模型对它的持续估计，因此任务天然是 POMDP。
2. Memory 不是为了“摘要得漂亮”，而是为了让最终创作建议更好。
3. 所以写 Memory 和读 Memory 应共享模型参数，并接受同一终端目标的训练。
4. 训练从高质量标注和 SFT 冷启动，再进入短程、长程 RL，最后蒸馏到线上 8B 模型。
5. 真正困难的是长程信用分配：单步方案太短，整轨迹方案太粗且太贵。
6. 随机窗口方案从真实长轨迹抽取局部窗口，在同一起点并行 rollout，再分别给 Memory 与 Suggestion 分配合理的优势。

## 详细文档

- [第 9 页：One Model、POMDP 与训练主链](./09-One-Model-POMDP与必要性.md)
- [第 10 页：Random-Trajectory-Window RL](./10-Random-Trajectory-Window-RL.md)

## 内容来源

- [当前演示文稿](../../index.html)
- 第 9 页主图：[任务结构图](../../assets/images/fig1_task_structure.png)
- 第 10 页方案图：[Plan 1](../../assets/plan1_grpo_ppt.html) · [Plan 2](../../assets/plan2_agentic_rl_ppt.html) · [Plan 3](../../assets/plan3_reinforce_ppt.html)

