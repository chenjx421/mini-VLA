# 强化学习与从头实现 PPO

## 1. 为什么项目里既有 BC 又有 RL

行为克隆有专家动作，训练稳定、样本效率高，但会遇到分布偏移。强化学习只需要奖励，可以探索
专家未覆盖状态并直接优化任务回报，但样本昂贵、对奖励敏感。

本项目先在 reach 任务用 privileged state 验证 PPO 实现。这样把“RL 算法是否正确”与“视觉
表征是否学好”分开。随后才适合研究视觉 PPO、BC 初始化或 residual RL。

## 2. Actor-Critic

Actor 给出策略分布：

\[
a_t\sim\pi_\theta(a_t\mid s_t)
\]

Critic 估计状态价值：

\[
V_\phi(s_t)\approx
\mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\right]
\]

连续动作 actor 输出高斯分布参数，再通过 `tanh` 映射到 `[-1,1]`。变换后 log probability 必须
包含 Jacobian 修正，否则 PPO ratio 不正确。

## 3. Advantage

Advantage 表示某动作相对当前状态平均水平有多好：

\[
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
\]

若 \(A>0\)，增加该动作概率；若 \(A<0\)，降低概率。

## 4. TD residual 与 GAE

一步 TD residual：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

GAE：

\[
\hat A_t=\delta_t+
\gamma\lambda\delta_{t+1}+
(\gamma\lambda)^2\delta_{t+2}+\cdots
\]

递推：

\[
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}
\]

\(\lambda\) 控制偏差和方差：

- 小 \(\lambda\)：更依赖 critic，方差低、偏差可能高；
- 大 \(\lambda\)：更接近 Monte Carlo，偏差低、方差高。

## 5. terminated 与 truncated 的 bootstrap

真实 terminal 后没有未来价值：

\[
V_{\text{bootstrap}}=0
\]

time-limit truncation 只是采样被截断，状态本身仍有价值：

\[
V_{\text{bootstrap}}=V(s_{T})
\]

但无论哪种 episode end，都不能让 GAE 穿过 reset 串到下一条 episode。本项目分别维护
bootstrap value 和 `episode_ends`，测试覆盖这一点。

## 6. PPO clipped objective

新旧策略概率比：

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
\]

目标：

\[
L^{CLIP}=
\mathbb{E}\left[
\min(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
)
\right]
\]

直觉：

- 好动作可以提高概率，但一次不能提高太多；
- 坏动作可以降低概率，但一次不能降低太多；
- clip 是近似 trust region，减少一次更新把策略推坏的风险。

代码最小化负目标，所以 `policy_loss` 前有负号。

## 7. Value clipping

critic 也可能一步变化太大。本项目比较：

\[
(V_\phi-R)^2
\]

和基于旧 value 限幅后的误差，取更大者。这是保守更新。

## 8. Entropy、KL 和梯度裁剪

- entropy bonus：防止策略过早变得确定，维持探索；
- approximate KL：监控新旧策略变化，超过阈值提前停止 epoch；
- gradient clipping：限制异常 batch 的梯度范数。

这些不是“让成功率自动变高”的开关。entropy 太大策略一直随机，太小则可能早熟；KL 阈值太小
几乎不学习，太大则失去约束。

## 9. Rollout 与更新

默认配置：

```text
8 parallel envs
x 256 rollout steps
= 2048 transitions per update
```

每轮将 2048 条 transition 打乱成 256 大小的 minibatch，重复最多 8 个 epoch。总训练步数
约 100k。

on-policy 的含义是：这些数据由当前或非常接近当前的策略采样，不能像 DQN 那样无限复用旧 replay
buffer。

## 10. Reward shaping

reach 任务可用距离势函数：

\[
\Phi(s)=-\|p_{ee}-p_{target}\|
\]

进步奖励：

\[
r_t^{progress}=\Phi(s_{t+1})-\Phi(s_t)
\]

它比每步直接给负距离更强调“这一步有没有进步”。成功再给稀疏 bonus。

常见 reward hacking：

- 只靠近但不抓；
- 反复进入退出阈值刷奖励；
- 用碰撞推动物体绕过夹取；
- 初始化就在目标区；
- 超时行为比尝试更划算。

因此奖励曲线和任务成功率必须同时看。

## 11. 当前 baseline 应怎样解读

三次训练都使用 100,352 个实际环境步，每个 policy 再用 20 条独立 episode final evaluation：

| Seed | Success | Mean return |
| ---: | ---: | ---: |
| 1 | 15/20 | 4.058 |
| 2 | 4/20 | 1.222 |
| 3 | 16/20 | 4.268 |

seed-level success 为：

\[
58.3\%\pm33.3\%\quad(\text{mean}\pm\text{sample std}, n=3)
\]

把三次 policy 的 eval episode 合并是 35/60，Wilson 95% CI 45.7%-69.9%。但 pooled interval 把
policy 训练随机性条件化了，所以不能用它替代 seed-level std。三 seed 结果能证明：

- PPO 数据流、GAE 和 clipped update 可以工作；
- MuJoCo 环境能被在线优化；
- seed 1/3 明显学习，训练不是只跑通不学习；
- 100k-step 配置存在显著训练 seed 敏感性。

它还不能证明：

- PPO 已稳定收敛；seed 2 只有 20%；
- pick-place RL 已解决；
- RGB policy 已解决；
- 比成熟库实现更强。

简历应写三 seed mean/std，不应挑 seed 3 的 80%。这也是为什么“多 seed”不是为了让表格更
完整，而是为了暴露优化不稳定。

## 12. 下一步 RL 路线

1. 增加训练步数并检查 seed 2 是收敛更慢还是落入坏局部最优。
2. curriculum：reach -> pick -> pick-place。
3. BC 初始化 actor，减少随机探索。
4. residual RL：最终动作 = VLA 动作 + 小残差。
5. 用 domain randomization 做鲁棒性评测。
6. 对 sparse reward、dense reward 和 progress reward 做消融。

## 13. 必答题

1. PPO 为什么是 on-policy？
2. clipped objective 在 advantage 正负时分别做什么？
3. GAE 中 \(\lambda\) 的作用是什么？
4. timeout 为什么仍需 value bootstrap？
5. entropy 和 KL 分别解决什么问题？
6. 为什么先用 state PPO，而不是一开始就用 RGB？
7. 高 return 但低 success 说明什么？
8. 为什么 pooled 35/60 的 Wilson interval 不能替代三 training-seed 标准差？
9. seed 2 失败时，怎样区分训练预算不足、奖励设计和实现 bug？
