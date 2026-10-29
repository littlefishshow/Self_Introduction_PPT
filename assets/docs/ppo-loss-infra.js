window.__DOC_PREVIEW_MARKDOWN__ ||= {};
window.__DOC_PREVIEW_MARKDOWN__["ppo-loss-infra"] = String.raw`# 4.3.3 PPO 损失函数与 Infra 工程优化

## 4.3.3 PPO 损失函数

训练采用 PPO-Clip 目标，叠加 TIS（轨迹级重要性采样）、KL3 正则和 entropy 奖励。

### PPO-Clip 目标

$$L^{CLIP}(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{old}}}\left[\frac{1}{|\tau|}\sum_{t \in \tau}\omega_\tau(\theta)\cdot\min\left(r_t(\theta)\,A_t,\ \operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,A_t\right)\right]$$

其中 $$r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$$ 是新旧策略的概率比。

### TIS（Trajectory Importance Sampling）

由于 rollout 与 train 之间有一轮延迟，对整条轨迹的重要性采样比做修正。轨迹级权重为：

$$\omega_\tau(\theta) = \operatorname{clip}\left(\frac{\pi_{old}(\tau)}{\pi_{rollout}(\tau)},0.5,2\right)$$

即整条轨迹所有 token 概率比的几何平均，用来修正因策略延迟一轮带来的 off-policy 偏差。

### KL3 惩罚

$$L^{KL}(\theta) = -\beta_{KL}\,\mathbb{E}_t\left[D_{KL}(\pi_\theta(\cdot|s_t)\ \|\ \pi_{ref}(\cdot|s_t))\right]$$

### Entropy 奖励

$$L^{ENT}(\theta) = +\beta_H\,\mathbb{E}_t\left[H[\pi_\theta(\cdot|s_t)]\right]$$

### 总损失

$$L(\theta) = L^{CLIP}(\theta) + L^{KL}(\theta) + L^{ENT}(\theta)$$

---

## 4.3.3.1 Loss 归约

一条 UID 的每个位置最多有一个 Memory action 和一个 Suggestion action。V2 不放大 action advantage，样本归约分母为：

$$L = \frac{1}{B}\sum_{u=1}^{B} \frac{1}{2L_u}\sum_{t=1}^{L_u}\sum_{a\in\{M,S\}} \frac{1}{N_{\text{token},u,t,a}}\sum_{j=1}^{N_{\text{token},u,t,a}} A_{u,t,a,j}\,\log\pi_\theta(a_{u,t,a,j}\mid s_{u,t})$$

求均值的顺序（从内到外）：

1. **第 1 层：按 token 求均值。** 对单个 action 的所有 policy token 求平均，消除“输出长就梯度大”的影响。
2. **第 2 层：按 action 分配权重。** 每个位置的 Memory + Suggestion 两个 action 各分 $1/(2L_u)$ 的权重，整条 UID 的总梯度贡献固定。
3. **第 3 层：按 batch 求均值。** 跨所有 UID 求平均。

分母 $d_{u,t,a}=2L_uN_{\text{token},u,t,a}$ 是完整公式的简写，它合并了“按 token 均值”和“按 action 分配权重”。

- 失败或缺失 action 的质量不转移给其他 action。
- 有真实 token 的失败 action 保留，只接受本地 penalty。
- 没有 policy token 的 action 使用 zero-mask。
- 外部 reward 最终失败时，失败位置及依赖后缀 zero-mask。
- 一个 UID 不影响其他 UID；整 batch 没有非零可训练 advantage 时跳过更新。

---

## 4.3.4 Infra 工程优化

### 4.3.4.1 Reward / Train 异步流水线

$$\text{generate}(n)\to\text{offload}\to\text{reward\_and\_convert}(n)\parallel\text{train}(n-1)\to\text{同步新权重}\to\text{generate}(n+1)$$

- reward 计算和 train 步骤异步重叠，但 rollout generation 与这两者不重叠。生成完才能算 reward，train 使用上一批次的数据。get reward 和 train 所需时间几乎完全一致，因此异步训练能够节省大量时间。
- 这会带来一轮 off-policy：train(n−1) 使用 generate(n−1) 时旧权重采集的数据，而 generate(n) 已使用新权重采样。train(n−1) 完成后立即同步新权重，再开始 generate(n+1)，所以每步只有一个批次存在一轮延迟。TIS 正是用来修正这一轮 off-policy 偏差。

### 4.3.4.2 数据集课程学习

训练数据按轨迹长度递进组织，轨迹长度上限依次设置为：

$$8 \to 15 \to 30$$

先用较短轨迹建立稳定的 Memory 更新与 Suggestion 生成闭环，再逐步扩大到更长轨迹，降低训练初期的方差和资源压力，并让策略逐阶段适应更长的状态依赖与信用跨度。

### 4.3.4.3 75% 轨迹提前停止

**问题：** rollout 时，长轨迹会阻塞短轨迹。短轨迹早已生成完，但必须等待最长轨迹完成才能开始 reward 和 train，GPU 利用率因此被拖低。

**方案：** 当前 batch 中 75% 的轨迹 rollout 完成时即停止生成，开始 reward_and_convert 和 train。剩余 25% 的轨迹不丢弃，在下一个 batch 中继续生成完成。

**收益：**

- 显著缩短每轮 wall-clock time，尤其适用于轨迹长度方差大的场景。
- 被截断的轨迹不会浪费，它们会在下一批次续完，相当于增加 batch 的有效样本流。
- 代价是截断位置附近的 future return 不完整，但这些位置本就 near-terminal，对 credit 分配影响有限。
`;
