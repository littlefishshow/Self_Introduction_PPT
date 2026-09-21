# 第 10 页：Random-Trajectory-Window RL

## 1. 这一页回答什么问题

第 9 页说明了为什么要用 One Model；第 10 页进一步解决训练中的核心难题：

> 一次早期 Memory 更新会影响后面多轮状态和最终 Suggestion。最终效果到底应该怎样归因给每一步，同时把训练成本控制在可承受范围内？

页面依次比较三种方案：

    Plan 1：只看单步       → 成本低，但没有未来信用
    Plan 2：看完整长轨迹   → 有全局结果，但信用混乱且成本过高
    Plan 3：随机局部窗口   → 固定成本下学习长程状态，是推荐方案

## 2. Plan 1：单步 GRPO / Local

做法：从同一个 Memory 状态 $M_i$ 出发，并行采样 $G$ 组“更新后 Memory + Suggestion + Reward”，再计算组间相对优势：

$$
A^{(g)}=\frac{r^{(g)}-\mu_r}{\sigma_r}
$$

优点是简单、便宜，也容易获得组内基线。

但它有两个根本问题：

- Memory 更新轨迹没有真正进入训练过程，无法观察这次写入对后续状态的影响。
- 如果更新后的 Memory 不是由正在训练的策略持续 rollout 得到，就不是真正的 on-policy 长程优化。

所以 Plan 1 只能回答“眼前这一步哪一个更好”，不能回答“这次更新会不会让未来越来越好”。

## 3. Plan 2：完整长轨迹 Agentic RL / Global

做法：从 $M_0$ 开始执行多轮 Memory 更新，直到 $M_T$，最后生成 Suggestion 并得到一个终端奖励，再把这个奖励广播给整条轨迹。

它确实让完整轨迹进入了训练，但问题也很明显：

- 多次 Memory update 共用同一个终端信号，无法判断究竟是哪一步产生了正面或负面影响。
- 在推荐与内容质量这类非可验证任务中，奖励噪声比数学题唯一答案更大，信用混淆更严重。
- 完整长轨迹的 rollout 和反向传播成本很高，页面指出即使 32 张卡也难以负担。

所以 Plan 2 是“看得足够远，但账算不清，也太贵”。

## 4. Plan 3：随机轨迹窗口 / Windowed

推荐方案不直接训练整条长轨迹，而是从真实长轨迹中随机截取一个固定长度窗口。它包含五个步骤。

### 4.1 随机抽取长度为 K 的局部窗口

从真实轨迹中随机选择起点 $M_i$，截取：

$$
M_i \rightarrow M_{i+1} \rightarrow \cdots \rightarrow M_{i+K}
$$

随机起点会覆盖轨迹早期、中期和晚期，包括已经变长、变脏甚至部分受污染的真实 Memory 状态。窗口长度固定后，训练成本不再随用户累计天数线性增长。

### 4.2 从同一起点并行 rollout N 条短轨迹

从相同 $M_i$ 出发并行采样 $N$ 条长度为 $K$ 的轨迹。每个位置都执行：

    Update Memory → Generate Suggestion → Reward

这让窗口中的每一步都拥有真实下游奖励，而不是只在终点得到一个稀疏信号。

### 4.3 只在相同位置比较，计算组间优势

第 $i+k$ 个位置的第 $n$ 条轨迹优势为：

$$
A_{i+k}^{(n)}=\frac{r_{i+k}^{(n)}-\mu_{i+k}}{\sigma_{i+k}}
$$

同一起点、同一用户、同一天、同一位置的样本彼此比较，可以对冲用户差异、日期差异和任务难度等策略外噪声；组均值同时提供了不需要额外 critic 的基线。

### 4.4 给两类动作分配不同的信用

Memory Update 会改变后续状态，因此需要承担窗口内未来结果：

$$
\widetilde A_{i+k}^{\mathcal M,(n)}=\sum_{j=k}^{K}\gamma^{j-k}A_{i+j}^{(n)}
$$

Suggestion 只决定当前位置的输出，因此只承担当前位置的奖励：

$$
\widetilde A_{i+k}^{\mathcal S,(n)}=A_{i+k}^{(n)}
$$

直观理解：

- 写 Memory 像修改后续所有步骤都会读取的“共享笔记”，应该为未来后果负责。
- 生成 Suggestion 像回答当前这一道题，只需要为当前答案负责。

### 4.5 用 GMPO 稳定长输出训练

Memory Update 往往包含上千 token。若使用普通算术平均，少数异常 token 的 importance ratio 可能带偏整条梯度。页面采用几何平均形式：

$$
r_i^{geo}=\exp\left(\frac{1}{|o_i|}\sum_t\log\rho_{i,t}\right)
$$

再将对应的 Memory 或 Suggestion 优势代入 clipped policy objective：

$$
\mathcal J=\mathbb E\left[\operatorname{clip}(r_i^{geo})\hat A_i\right]
$$

这里的目标是降低极端 token ratio 和输出长度带来的方差，使长输出更新更稳定。GMPO 解决的是 ratio 与长度稳定性，不负责消除多路 Reward 之间的目标冲突。

## 5. 为什么 Plan 3 更合理

页面的展开说明给出五个设计理由：

1. **Random Window**：从真实长轨迹随机取窗口，覆盖不同阶段的真实状态，同时让成本只与 $K$ 有关。
2. **Real Dense Reward**：每一步的 Memory 都立即被 Suggestion 消费并获得业务奖励，信号密度约提高到原来的 $K$ 倍。
3. **Same Start, Same Position**：同起点、同位置比较，免费获得组内基线并抵消大量环境噪声。
4. **Causal Credit**：Memory 承担未来折扣后果，Suggestion 只承担即时后果，更符合两类动作的真实因果范围。
5. **GMPO**：对长输出中的异常 token ratio 更鲁棒，降低训练塌陷风险。

## 6. 页面展示的三组实验信号

### 6.1 Long-Horizon Recall

随着 Memory 更新轮数增加，RL 与 OPD 的记忆保持优于 Gemini 3 Pro、SFT 和 Baseline。它支持的结论是：只在短轨迹上表现良好，不代表模型已经具备稳定的长程状态能力。

### 6.2 Reward × Chunk

当前图中 `4096` 的 Reward 最高，`2048` 次之，`1024` 最低。说明单步可见的信息容量会影响训练上限；窗口或 chunk 太小可能无法提供足够的决策上下文。

### 6.3 Same Model Cross-Play

Memory Updater 和 Suggestion Generator 的交叉组合中，`RL → RL` 得分最高，为 `0.86`，跨模型组合普遍下降。这与第 9 页的判断一致：同模型读写能减少协议错配。

## 7. 关键变量

| 符号 | 含义 |
| --- | --- |
| $M_i$ | 长轨迹中第 $i$ 个 Memory 状态 |
| $K$ | 随机窗口长度 |
| $N=G$ | 从同一起点采样的并行短轨迹数 |
| $\tau$ | 在原始长轨迹上重复随机抽取窗口的次数 |
| $r_{i+k}^{(n)}$ | 第 $n$ 条轨迹在窗口第 $k$ 个位置的即时奖励 |
| $A_{i+k}^{(n)}$ | 同一位置跨 $N$ 条轨迹计算的组间优势 |
| $\gamma$ | 未来优势的折扣系数 |
| $\rho_{i,t}$ | 新旧策略在第 $t$ 个 token 上的概率比 |
| $o_i$ | 一次 Memory Update 或 Suggestion 的输出 token 序列 |

## 8. 这一页最终想让观众记住什么

> Random-Trajectory-Window RL 不是简单截短长轨迹，而是在真实长程状态上做局部、同起点、同位置的并行实验，再按照动作的因果范围分配信用。

它在三个目标之间取得折中：

- 比单步 GRPO 看得更远。
- 比完整 Agentic RL 更容易定位每一步的贡献。
- 通过固定窗口、密集奖励和 GMPO，把算力成本与训练稳定性控制在可实现范围内。

