# memory_RL4 当前 Reward 与 Advantage 计算

> 核对日期：2026-10-12  
> 适用实现：`memory_RL4/v3` paired-prestate K=4 Stage 2  
> 当前正式入口：`v3/scripts/run_rl4_stage2_h2_uid8_from_ckpt125.sh`  
> 说明：本文区分 reward 服务输出的原始 reward、进入训练的 advantage，以及本地 penalty；三者不能混为一条 reward 曲线。

## 1. 结论先行

当前 `memory_RL4` 同时训练两类动作：

1. **Suggestion 动作**：比较同一 Memory state 下的 K=4 个 Suggestion，逐 reward head 做组内中心化和尺度归一化，再以 `Funnel/Safety/Diversity = 0.90/0.09/0.01` 聚合。
2. **Memory update 动作**：比较更新后 Memory 与同一 block 更新前 prestate 的后续 Suggestion reward；使用同 request、同 replica seed 的 K=4 配对差分，并经过 absolute floor、SNR、符号一致性、LOO 和 top/bottom 门控。

核心区别是：

```text
Suggestion advantage = 同状态下四个 Suggestion 之间的相对偏好
Memory advantage     = updated Memory 相对 prestate Memory 的后续收益差
```

当前训练的 nominal objective 为：

$$
R_{quality}=0.90\,U_{funnel}+0.09\,J_s+0.01\,J_d
$$

但训练不会直接把这个绝对值当 advantage。Suggestion 使用组内标准化；Memory 使用 paired-prestate 差分、置信门和 previous-rollout EMA scale。

## 2. 当前配置快照

| 项目 | 当前值 | 含义 |
|---|---:|---|
| Suggestion replicas | K=4 | 同一 state 生成四次 |
| Memory branches | 4 | 同一 block prestate 出发的四条更新轨迹 |
| Prestate replicas | K=4 | measurement-only baseline |
| Reward weights | `0.90/0.09/0.01` | Funnel / Safety / Diversity |
| Memory minimum paired replicas | 4 | 必须 4/4 exact pairs |
| Memory paired SNR | `>=2.0` | 差分均值至少为两个标准误 |
| Sign agreement | `>=0.75` | 至少 3/4 replica 同方向 |
| LOO sign agreement | `>=0.75` | 删除任一 replica 后方向仍稳定 |
| Minimum valid branches | 3 | 少于三条可比较 branch 时整 block 不给 quality credit |
| Memory branch selection | top/bottom only | 中间两条 branch quality-mask |
| Memory absolute floors | Funnel `0.004`; Safety/Diversity `0.02` | 先过滤，再 soft shrink |
| Suggestion deadzone | `0.4 × effective_std` | 只作用于 Suggestion quality component |
| Memory deadzone | 关闭 | paired path 使用 absolute floor，不再叠加 `0.4×std` |
| Component clip | `[-1,1]` | 每个 head 归一化后 |
| Final clip | `[-1.5,1.5]` | quality 与本地 penalty 合成后 |
| Safety mode | `soft_only` | unsafe candidate 不把 business heads 整体清零 |
| Diversity mode | `hybrid_v3`, group-only | 不使用 recent-title novelty |

配置入口见 `v3/scripts/run_stage2_paired_prestate_from_ckpt125.py`；不可兼容的配置由 `v3/run_contract.py` 和 launcher preflight fail closed。

## 3. Reward 服务的原始 heads

Reward 服务先给每条 Suggestion 输出五类可观测量：

| Head | 字段 | 当前用途 |
|---|---|---|
| Click | `U_click` | 诊断；当前正式 objective 不直接加权 |
| Download | `U_download` | 诊断；当前正式 objective 不直接加权 |
| Funnel | `U_funnel` | 主业务 head，权重 0.90 |
| Safety | `J_s` | 候选安全通过率，权重 0.09 |
| Diversity | `J_d_suggestion` / `J_d_memory` | 组内多样性，权重 0.01 |

### 3.1 Fine-rank 平滑校准

每个 candidate `i` 的原始业务 head 预测 `p_{i,h}` 使用当前 `smooth_v1` calibration：

$$
c_{i,h}=\frac{p_{i,h}}{p_{i,h}+s_h}
$$

当前 scale 为：

```text
click            0.00126504898
active_download  0.04747009277
active_public    0.00152540207
active_share     0.00621795654
```

所有去重后的 candidate 都进入均值；不是只取 top-4。低质量 filler 因此会真实拉低整条 Suggestion 的 reward，而不会被排序截断隐藏。重复 description 由独立的本地 penalty 处理。

### 3.2 Funnel reward

对 candidate `i`：

$$
V_i=0.50c_{i,download}+0.30c_{i,public}+0.20c_{i,share}
$$

$$
F_i=c_{i,click}\cdot V_i
$$

整条 Suggestion 的 Funnel reward：

$$
U_{funnel}=\frac{1}{N}\sum_{i=1}^{N}F_i
$$

乘法在 candidate 内先完成，再跨 candidates 求均值。这样一个标题的高 Click 不能借用另一 candidate 的高下游价值。

### 3.3 Safety reward

每个 candidate 的独立安全服务输出 `s_i∈{0,1}`：

$$
J_s=\frac{1}{N}\sum_i s_i
$$

当前是 `soft_only`：即使某个 candidate 不安全，`U_funnel` 等 business heads 仍保留，由较低的 `J_s` 提供负向相对信号。`safety_min`、`unsafe_candidate_count` 仍写入 audit，但不会触发全样本 hard gate。

### 3.4 Diversity reward

当前 `hybrid_v3` 结合分词/Jaccard、embedding 和语义轴保护，计算同一 Suggestion 内 candidates 的 group diversity：

```text
J_d_suggestion = D_group
J_d_memory     = D_group
D_recent       = unavailable
```

当前 RL4 rollout 不维护 recent-title state，Suggestion prompt 只输入 `profile + current Memory`。reward 请求仍保留 `recent_titles=[]`，但它只是 wire schema 的空兼容字段；`D_recent` 固定 unavailable。单 candidate 没有可比较 pair 时，Diversity 同样标记为 unavailable，不伪造为 0 或 1。

共享 scorer `memory_RL/reward/diversity_v3.py` 中的 `D_group_only_v2_recent_prompt_retained` 常量和 V15 注释属于历史命名，不能用来判断当前 RL4 rollout；当前 run contract 明确锁定 `recent_titles_prompt=removed_v1` 和 `recent_titles_reward=empty_compat_field_disabled_v1`。

### 3.5 缺失 head 的处理

每个 head 有独立 validity。缺失 observation 的规则是：

- 不把缺失值填成 0；
- Suggestion 逐 head 独立计算 group component；某 head 观测不足时该 component 不参与，但固定 nominal weights 不重归一；
- Memory 以 Funnel 为 quality anchor；Funnel 不可用时 quality unavailable，Safety/Diversity 缺失时才在剩余可用 heads 上重归一权重；
- reward infrastructure failure 超过 run contract 阈值时终止训练，而不是把服务故障当模型负 reward。

## 4. Suggestion advantage

对同一 state 下的 K=4 Suggestion，逐 head `h` 收集有效 raw reward `R_{i,h}`。至少三条有效 observation 后计算：

$$
\mu_h=\operatorname{mean}_i R_{i,h},\qquad
\sigma_h=\operatorname{sample\_std}_i R_{i,h}
$$

有效尺度为：

$$
e_h=\max(\sigma_h,\ noiseFloor_h,\ EMAStd_{h,previous})
$$

先中心化：

$$
x_{i,h}=R_{i,h}-\mu_h
$$

当前 Suggestion deadzone 为 `0.4×effective_std`：

$$
Z_{i,h}=
\begin{cases}
0,&0<|x_{i,h}|<0.4e_h\\
x_{i,h}/(e_h+10^{-6}),&\text{otherwise}
\end{cases}
$$

quality advantage：

$$
A^{S,quality}_i=
\operatorname{clip}
\left(
0.90Z_{i,funnel}+0.09Z_{i,J_s}+0.01Z_{i,J_d},
-1,1
\right)
$$

注意：这是 centered group preference，所以一个组内的 signed mean 接近 0 是正常现象；判断信号强度应看 `mean_abs`、非零率和逐 head coverage。

### 4.1 Suggestion 本地 penalty

本地格式错误采用 penalty-only：一旦存在本地 penalty，remote quality 置 0，避免高业务 reward 抵消协议错误。

| 条件 | 默认 penalty |
|---|---:|
| 不可解析输出 | `-0.3` |
| proposal 数超过 8 | `-0.3` |
| 首个 candidate 格式错误 | 从 `-0.1` 起 |
| 更多错误 candidate | 按错误 candidate 比例线性加重，最大 `-0.3` |
| 同 candidate 额外规则错误 | 每项追加 `-0.02` |
| action-global wrapper/schema 错误 | 每项追加 `-0.1` |
| description 重复 | `-0.2` |

最终：

$$
A^S_i=
\begin{cases}
\operatorname{clip}(P_{format}+P_{duplicate},-1.5,1.5),&P_{local}\neq0\\
A^{S,quality}_i,&\text{otherwise}
\end{cases}
$$

## 5. Memory paired-prestate advantage

### 5.1 采样拓扑

每个 block 从不可变的 `S0=start_memory` 出发：

```text
S0 prestate measurement-only ─────────────── K=4 reward samples

S0 ── Memory branch 0 ── terminal state 0 ─ K=4 reward samples
S0 ── Memory branch 1 ── terminal state 1 ─ K=4 reward samples
S0 ── Memory branch 2 ── terminal state 2 ─ K=4 reward samples
S0 ── Memory branch 3 ── terminal state 3 ─ K=4 reward samples
```

prestate K=4 在同一 block 的四个 branches 间共享；下一个 block 和下一个 rollout 都重新生成。prestate samples 是 measurement-only，`loss_mask=0`，不会训练 Suggestion policy。

### 5.2 配对差分

对 branch `b`、replica `r`、head `h`：

$$
d_{b,r,h}=R(updated_b,r,h)-R(prestate,r,h)
$$

$$
\Delta_{b,h}=\frac{1}{4}\sum_{r=1}^{4}d_{b,r,h}
$$

$$
SE_{b,h}=\frac{sampleStd_r(d_{b,r,h})}{\sqrt{4}},\qquad
SNR_{b,h}=\frac{|\Delta_{b,h}|}{SE_{b,h}}
$$

updated/prestate 必须匹配：replica index 和 seed、UID / episode / block / logical request、block start state hash、generation policy version 与 reward snapshot fingerprint。任一 identity 不一致，该 replica 不允许伪装成 paired evidence。

### 5.3 置信门

置信门不是单一阈值，而是四层串联的 fail-closed 判定：

```text
pair identity 完整
  -> 单 head paired credit 通过
  -> Funnel 选出 block 的 top/bottom
  -> Funnel anchor + 各辅助 head 独立可用性
```

任何一层失败都不会把缺失或不确定质量填成 0 reward；对应 quality component 会变成 unavailable。只有 Memory 自己的本地 penalty 可以绕过 quality gate 继续训练。

#### 5.3.1 第一层：exact pair identity

对每个 `replica_id=r`，updated 和 prestate 两边都必须存在，并且以下字段完全相同：

- `suggestion_replica_index`；
- `suggestion_replica_seed`；
- `memory_pair_id`；
- `memory_logical_request_hash`；
- `memory_block_start_state_hash`；
- `generation_policy_version`；
- `reward_version_snapshot_sha256`。

只有交集中的合法 replicas 才进入 `d_{b,r,h}`。当前要求 `count=4`；少一条、seed 错位、policy/reward snapshot 不同，都会使该 head 的 `reward_confident=False`。

这里的 `reward_confident` 只表示“四条 observation 可以合法配对”，尚不表示差分足够大或足够稳定。

#### 5.3.2 第二层：逐 head paired credit

$$
direction_{b,h}=sign(\Delta_{b,h})
$$

$$
signAgreement_{b,h}=\frac{1}{4}\sum_{r=1}^{4}\mathbf{1}[sign(d_{b,r,h})=direction_{b,h}]
$$

零差分的符号为 0；当整体方向非零时，它不会被算作同方向。

LOO 检查会依次删除一个 replica，得到四个三样本均值：

$$
\Delta^{(-r)}_{b,h}=\frac{1}{3}\sum_{q\neq r}d_{b,q,h}
$$

$$
LOOAgreement_{b,h}=\frac{1}{4}\sum_{r=1}^{4}\mathbf{1}[sign(\Delta^{(-r)}_{b,h})=direction_{b,h}]
$$

最终 `credit_passed=True` 必须同时满足：

1. `reward_confident=True`，即 4/4 exact pairs 且无 identity failure；
2. `|Δ| >= absolute_floor[h]`；
3. `SNR=|Δ|/SE >= 2.0`；
4. `signAgreement >= 0.75`，即至少 3/4 原始差分同方向；
5. `LOOAgreement >= 0.75`，即至少 3/4 删一子集仍保持方向。

当前 absolute floors：

| Head | Floor | 当前是否进入正式 objective |
|---|---:|---|
| Funnel | `0.004` | 是，且是 anchor |
| Safety `J_s` | `0.02` | 是 |
| Diversity `J_d` | `0.02` | 是 |
| Click | `0.004` | 否，仅保留兼容配置 |
| Download | `0.004` | 否，仅保留兼容配置 |

当 `SE<=1e-12` 时，非零 `Δ` 的 SNR 视为 `+∞`，而 `Δ≈0` 的 SNR 为 0。这样完全一致的非零 paired deltas 可以通过，但四个全零差分不能借助除零通过。

一个仅用于说明的 K=4 例子：

```text
d = [+0.010, +0.012, +0.008, +0.011]
Delta = +0.01025
sample_std = 0.001708
SE = 0.000854
SNR = 12.0
sign agreement = 4/4
LOO agreement = 4/4
```

对 Funnel 而言它超过 `0.004` floor，因此逐 head paired credit 可以通过。

#### 5.3.3 第三层：Funnel extreme branch gate

逐 head 通过仍不等于 branch 会训练。对一个 block，代码先排除轨迹不完整、drop 或 transition fallback 的 branch，再取 Funnel `reward_confident=True` 的 branches：

$$
eligible_b=validTrajectory_b\land rewardConfident_{b,funnel}
$$

至少需要 3 条 eligible branches；否则整个 block 返回 `insufficient_valid_branches`。在 eligible branches 中按 Funnel 的原始 paired mean `Δ_{b,funnel}` 冻结：

$$
b_{top}=\arg\max_b\Delta_{b,funnel},\qquad
b_{bottom}=\arg\min_b\Delta_{b,funnel}
$$

extreme 排序只要求 Funnel observations 可配对，即 `reward_confident`；排序后 top/bottom 自己仍必须通过 floor、SNR、sign 和 LOO 的 `credit_passed`。中间 branches 统一返回 `ambiguous_middle_branch`。当前没有额外的 top-vs-second 或 bottom-vs-second margin gate。

top/bottom 的最终符号仍由各自相对 prestate 的 `Δ` 决定，而不是被强制成一正一负，因此允许四条 update 都有益时两个 extreme 都为正，或四条都受损时两个 extreme 都为负。

#### 5.3.4 第四层：Funnel anchor 与辅助 heads

Funnel branch gate 必须先通过。随后每个 active head 再独立检查自己的 `credit_passed`：

```text
head_gate[h]
  = funnel_branch_gate
    AND paired_stats[h].credit_passed
    AND branch in {top, bottom}
```

因此：

- Funnel 失败：整个 Memory quality unavailable，即使 Safety/Diversity 看起来稳定也不能单独训练；
- Funnel 通过、Safety 失败：保留 Funnel，跳过 Safety；
- Funnel 通过、Diversity 失败或不适用：保留其他 heads，跳过 Diversity；
- local transition/protocol failure：即使 quality gate 通过，该 Memory step 仍切换到 penalty-only。

主要 gate reason 与含义：

| Reason | 含义 | 训练结果 |
|---|---|---|
| `pair_identity_mismatch` | seed/request/state/policy/reward identity 不一致 | quality unavailable |
| `insufficient_matched_replicas` | 少于 4 个 exact pairs | quality unavailable |
| `below_absolute_floor` | paired mean 小于绝对噪声门槛 | quality unavailable |
| `low_paired_snr` | 均值相对标准误不够大 | quality unavailable |
| `low_sign_agreement` | 四个原始差分方向冲突 | quality unavailable |
| `low_loo_agreement` | 删除一个 replica 后方向不稳 | quality unavailable |
| `insufficient_valid_branches` | block 内不足三条可排序 branch | 整个 block 无 quality credit |
| `ambiguous_middle_branch` | 不是 top/bottom | 该 branch 无 quality credit |
| `invalid_memory_trajectory` | step 缺失、drop 或 transition fallback | quality unavailable；本地错误另行 penalty-only |

### 5.4 Soft shrink 与尺度归一化

5.3 只决定哪些 `(branch, head)` 有资格进入 quality。5.4 再把通过的 paired mean 转换到稳定的训练尺度。该过程只有缩放，不做跨 UID 或跨 batch 的均值中心化，因此 0 始终表示“相对 prestate 没有改善”。

### 5.4.1 Absolute-floor soft shrink

通过 gate 后不直接使用原始 `Δ`，而是先减去该 head 的 absolute floor：

$$
\widetilde\Delta_{b,h}=\operatorname{sign}(\Delta_{b,h})\max(|\Delta_{b,h}|-floor_h,0)
$$

它保留原符号，并使阈值处连续：

```text
Delta = +floor       -> shrunk = 0
Delta 略高于 +floor  -> 从 0 平滑增长
Delta = -floor       -> shrunk = 0
Delta 略低于 -floor  -> 从 0 向负方向平滑增长
```

### 5.4.2 Previous-rollout EMA scale

rollout `t` 只读取 `baseline_state_after_{t-1}.json` 中的：

```text
memory_paired_delta_std_ema[h]
memory_paired_delta_std_ema_updates[h]
```

当前 rollout 的 paired deltas 不会改变自己使用的分母。有效 scale 为：

$$
scale_h=\max(EMAStd_{h,t-1},stdFloor_h)
$$

当前 std floors：

| Head | `stdFloor_h` |
|---|---:|
| Funnel | `0.008` |
| Safety `J_s` | `0.105` |
| Diversity `J_d` | `0.129` |
| Click | `0.011` |
| Download | `0.010` |

rollout 0 没有历史 state，直接从这些 floors 初始化。rollout `t>0` 默认必须找到精确的 `t-1` state，并校验 EMA beta、state version 和 run-contract fingerprint；缺失时默认 fail closed，只有显式启用 baseline reset override 才允许重置。

rollout `t` 完成后，为下一 rollout 更新：

$$
EMAStd_{h,t}=\max\left(stdFloor_h,\beta EMAStd_{h,t-1}+(1-\beta)BatchStd_{h,t}\right)
$$

其中当前 `β=0.9`。`BatchStd` 使用当前 rollout 中轨迹有效、identity 完整且 `reward_confident=True` 的 branch paired means；不只使用 top/bottom，也不要求已经通过 floor/SNR。每个 head 少于两个 observation 或 batch std 近零时，不更新该 head EMA。

这种时序有两个目的：当前 batch 不能通过改变自己的分母来放大或压小 advantage；strict resume 时只要恢复精确的上一 rollout state，就能复现相同尺度。

### 5.4.3 Component normalization 与 clip

soft-shrunk component 除以 previous-rollout scale；当前 multiplier 为 1：

component normalization：

$$
Z^M_{b,h}=\operatorname{clip}\left(\frac{\widetilde\Delta_{b,h}}{scale_h},-1,1\right)
$$

Memory paired path 不再叠加 `0.4×std` deadzone。若 Funnel 可用但辅助 head 缺失，则只在可用 heads 上重归一权重；无 Funnel 时 quality unavailable。

### 5.4.4 可用 head 的权重重归一

base weights 为：

```text
w_funnel = 0.90
w_Js     = 0.09
w_Jd     = 0.01
```

令 `H_available` 为同时通过 Funnel branch gate 和自身 head gate 的集合。Funnel 必须属于该集合，否则整个 quality unavailable。有效权重为：

$$
w'_h=\frac{w_h}{\sum_{q\in H_{available}}w_q}
$$

| Available heads | 有效权重 |
|---|---|
| Funnel + Safety + Diversity | `0.90 / 0.09 / 0.01` |
| Funnel + Safety | `0.9091 / 0.0909` |
| Funnel + Diversity | `0.9890 / 0.0110` |
| 只有 Funnel | `1.0` |
| 只有 Safety/Diversity、无 Funnel | quality unavailable |

最终 block quality：

$$
A^{M,quality}_b=\operatorname{clip}\left(\sum_{h\in available}w'_hZ^M_{b,h},-1,1\right)
$$

缺失辅助 head 的重归一只发生在 Memory path。Suggestion path 仍使用固定 nominal weights，不因某个 head 缺失而放大其余 heads。

### 5.4.5 数值例子

审计中的一个 Funnel branch 有：

```text
paired mean Delta_funnel = +0.0100948
absolute floor           =  0.0040000
shrunk delta             = +0.0060948
previous-rollout scale   =  0.00902678
```

$$
Z^M_{funnel}=clip\left(\frac{0.0100948-0.004}{0.00902678},-1,1\right)\approx +0.67519
$$

如果该 branch 只有 Funnel 可用，则重归一后 `w'_funnel=1`，所以 `A^{M,quality}≈+0.67519`。还要区分两个审计量：`memory_delta_quality_advantage` 是 paired normalized quality；`raw_terminal_reward` 是 terminal state 的绝对 reward head 均值按有效权重聚合，只用于记录，不是 Memory update 的因果 advantage。

### 5.5 Sequence credit 与本地 penalty

一个 block 只测量 terminal state，因此识别的是整段 Memory sequence 的收益，不是某一步的边际收益。block 内每个有效 Memory step 得到同一个 quality advantage，loss reducer 再按真实 H 使用一次 `1/H`。

| 条件 | 计算 |
|---|---:|
| Memory 长度 `<=1024` tokens | `0` |
| 超过 1024 tokens | `-min(0.5, 0.0005×over_tokens)` |
| transition fallback / 非法状态转移 | `-0.2` |
| 可恢复 JSON wrapper 协议违规 | `-0.1` |

严重 transition / protocol failure 使用 penalty-only，不允许 remote quality 抵消本地错误。

对单个 Memory step：

$$
A^M=
\begin{cases}
clip(P_{len}+P_{transition}+P_{protocol},-1.5,1.5),&\text{transition/protocol failure}\\
clip(A^{M,quality}+P_{len},-1.5,1.5),&\text{quality available}\\
clip(P_{len},-1.5,1.5),&\text{only length penalty}\\
0\ \text{and loss-mask},&\text{no quality and no local penalty}
\end{cases}
$$

## 6. `raw_reward`、`reward` 与 `advantage` 的区别

| 字段/概念 | 含义 |
|---|---|
| `U_funnel/J_s/J_d_*` | reward 服务的原始绝对观测 |
| `v3_weighted_raw_reward` | Suggestion 当前有效 heads 的绝对加权和 |
| `memory_branch_mean_reward` | branch terminal K=4 的逐 head 均值 |
| `memory_previous_state_mean_reward` | prestate K=4 的逐 head 均值 |
| `memory_delta_components` | `terminal - prestate` 的 paired mean |
| `memory_delta_normalized` | shrink 后除 previous-rollout EMA scale 的逐 head 值 |
| `suggestion_advantage` | same-state group preference 加本地 penalty 后的训练信号 |
| `memory_advantage` | paired-prestate sequence return 加本地 penalty 后的训练信号 |

绝对 raw reward 上升不等价于 advantage 为正；组内 centered Suggestion advantage 均值接近 0 也不表示 reward 没有改善。

## 7. 为什么改成现在的设计

| 旧问题 | 当前修改 | 原因 |
|---|---|---|
| updated branches 相对排名会强制分赢家/输家 | 显式 prestate baseline | 让 0 表示“不更新”，支持 all-good / all-bad |
| 相邻位置 delta 同时改变 request/date/position | same request、same seed paired delta | 尽量只隔离 Memory state 变化 |
| K=4 方差大，中间排名不稳定 | SNR、sign、LOO、top/bottom only | 只训练可分辨的极端偏好 |
| 小差异过阈值就获得大梯度 | absolute floor 后 soft shrink | 阈值处连续增长 |
| 当前 batch 自己决定尺度 | previous-rollout EMA | 避免自归一反馈并支持 resume |
| Click 高但下游价值弱 | candidate-aligned Funnel product | 同一 candidate 同时具备吸引力与下游价值 |
| safety false positive 清零业务信号 | candidate-mean Safety + soft-only | 保留业务信号，同时维持 Safety 压力 |
| recent novelty 随轨迹增长产生位置偏置 | Diversity group-only | 避免后期机械变差 |
| head 缺失填 0 | 独立 validity + 可用 head 重归一 | 缺测是 unavailable，不是假想低 reward |
| 高业务 reward 抵消非法状态 | penalty-only | 强制协议和状态机正确性 |
| block return 被每步重复后放大总信用 | 每步写完整 return，loss 仅做一次 `1/H` | 保持 token 训练稳定，同时守恒 sequence credit |

## 8. 当前活跃任务的运行时偏差

当前活跃 run 名义配置为 H=2，但审计 rollout 0–25 时，RolloutManager 未收到 `MEMORY_RL4_V28_BLOCK_HORIZON`，使用了 `rollout.py` 默认 H=4。原因是 Ray 环境注入白名单遗漏 `MEMORY_RL4_`。

这不改变 raw reward heads、paired delta 或 gate 公式，但会改变 credit unit：

```text
名义设计：两步 Memory sequence -> 一个 terminal return
实际运行：四步 Memory sequence -> 一个 terminal return
```

因此当前 run 的 `memory_advantage` 只能解释为 H4 sequence-level reward，不能解释成单次 memory update 的边际收益，也不能作为 H2 实验结果。

## 9. 已知限制

1. fixed K=4；adaptive K=8 尚未实现。
2. Safety / Diversity absolute floor 仍需 frozen-snapshot test-retest 标定。
3. same seed 没有完全消除生成/打分非确定性。
4. top/bottom extreme 尚无独立 ranking-margin gate。
5. sequence return 无法区分 block 内具体哪一步产生收益。
6. held-out checkpoint gain 尚需 paired causal evaluation。

## 10. 实现索引

- Raw reward contract：`memory_RL/reward/contract.py`
- Fine-rank/Funnel 公式：`memory_RL/reward/reward_client.py`
- Diversity：`memory_RL/reward/diversity_v3.py`
- Suggestion 与 Memory advantage：`memory_RL4/v3/advantage.py`
- Paired gate：`memory_RL4/v3/confidence.py`
- Prestate/terminal 采样：`memory_RL4/v3/rollout.py`
- Run contract：`memory_RL4/v3/run_contract.py`
- 当前 Stage 2 参数：`memory_RL4/v3/scripts/run_stage2_paired_prestate_from_ckpt125.py`
