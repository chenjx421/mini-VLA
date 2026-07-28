# EmbodiedVLA-RL 工程日志

> 记录规则：保留失败、超时和错误假设；区分观察与推断；每个结论绑定代码、JSON、图像或命令。
> 本日志会随项目继续更新，日期使用 Asia/Shanghai。

## 2026-07-27：重新定义项目

### 背景

目标不是修补已有项目，而是重新设计一个能同时证明仿真、算法、强化学习、VLA、ROS2 和
SLAM 能力的简历项目。机器是 Intel i5-13420H、16 GB 内存、无 NVIDIA GPU，因此所有核心
实验必须能在 CPU 上复现。

### 关键决策

1. 使用 MuJoCo 搭建关节级 SO-ARM100 操作环境，而不是用 2D 玩具环境。
2. 主任务采用“语言指定颜色方块和左右目标区”的抓取放置，强迫策略使用视觉和语言。
3. VLA 保持约百万参数，从头训练，以便完整讲清 token、融合、动作头和部署。
4. PPO 用独立 state-based reach 任务验证在线强化学习能力，不冒充 RGB-VLA。
5. SLAM 放在独立移动传感平台，因为固定桌面机械臂不需要定位建图。
6. `contact` 与 `contact_assisted` 分开报告，不把辅助抓取包装成严格接触成功。

### 为什么这样做

- CPU 约束排除了从头训练大型视觉语言 backbone，但不妨碍研究完整闭环。
- 独立的 manipulation、RL 和 SLAM 链比把关键词强行串成一个不合理系统更容易解释。
- 小模型可以看到每一个张量和 loss，面试中不依赖“调用了某个大库”。

## 2026-07-27：MuJoCo 环境、IK 与专家

### 实现

- 5D 动作：`[dx, dy, dz, wrist, jaw]`。
- 50 Hz 控制循环，末端位移通过阻尼最小二乘 IK 转为关节目标。
- 观测包含 RGB、depth、12D proprioception、语言 token 和调试用 privileged state。
- 场景包含红/绿/蓝三个方块及左右两个目标区。
- 专家采用 approach、descend、close、lift、transport、release 等显式阶段。

相关代码：

- `embodied_vla/envs/so_arm_pick_place.py`
- `embodied_vla/control/ik.py`
- `embodied_vla/experts/pick_place.py`

### 难题 1：一开始就成功的假轨迹

**现象：** 某些 episode 在 1 step 内成功，不符合抓取放置过程。

**假设：** 方块初始化时可能已经落入目标区；旧成功条件只看最终位置。

**验证：** 检查初始化位置和成功判据，发现二者确实允许“未抓、未抬、已在目标区”。

**修复：**

- 初始化采样排除目标区；
- 成功必须经历 lift 后再 place；
- 回归测试检查 terminal 与 success 语义。

**面试重点：** 仿真能跑不代表任务定义正确。先验证奖励和终止条件，否则模型可能优化漏洞。

### 难题 2：抓取稳定性与实验口径

纯摩擦接触在低成本模型和简单控制器下容易滑落。项目没有直接 teleport 方块，而是新增
`contact_assisted`：只有双指已经接触且夹爪关闭后才施加稳定约束。严格 `contact` 模式仍保留，
所有结果必须写明模式。

## 2026-07-27：专家数据集与审计

### 正式数据

- 路径：`datasets/so_arm_pick_place_v2_120_dr`
- 120 个成功 episode，18,972 帧；
- 3 种颜色 x 2 个目标方向，每类 20 条；
- 开启 domain randomization；
- 收集时有 1 次 timeout 被拒绝；
- 数据 fingerprint：
  `42c36938e5157cf9e188413e5d4cb76cb85b0f551752853a8e8d18a7c77914b3`

证据：`datasets/so_arm_pick_place_v2_120_dr/statistics.json`

### 难题 3：相邻帧泄漏

随机按帧切 train/validation 会把同一条轨迹的相邻图像放到两边，验证指标虚高。修复为按完整
episode 切分，并在 6 个任务组内分层，得到 102 个训练 episode 和 18 个验证 episode。

### 难题 4：随机读取反复解压

**现象：** DataLoader shuffle 后吞吐明显低。

**原因：** 小 episode cache 导致压缩 NPZ 被反复打开和解压。

**修复：** episode cache 大小可配置；在 16 GB 机器上正式数据可常驻内存。

### 已知限制

立方体没有姿态要求，wrist expert action 恒为 0。总体 MAE 会被这一维美化，因此必须报告每个
动作维度，并把带姿态物体列为后续扩展。

## 2026-07-27：从头实现 PPO

实现了 tanh Gaussian Actor-Critic、GAE、policy/value clipping、entropy、KL 监控、
gradient clipping 和并行 rollout。特别检查 terminal 与 timeout 的 bootstrap 语义。

### 已验证结果

- 运行：`outputs/ppo_reach_seed1_clean_20260727`
- 100,352 environment steps；
- 独立 20 episode 评测：15/20 成功；
- mean return：4.0578；
- mean length：61.85。

证据复制到：`results/ppo_reach`

### 难题 5：多实验并发让吞吐崩溃

seed 1 约 660 SPS；同时启动 seed 2/3 后降到约 10 SPS。实验在 4,096 steps 主动停止，避免把
资源争用下的异常速度当成算法问题。后续多 seed 改为顺序运行。

**教训：** benchmark 必须记录系统负载和并发任务；“代码慢”与“机器被抢占”是两类问题。

## 2026-07-27 至 2026-07-28：Tiny-VLA

### 架构

- 1,153,169 个参数；
- 64 个视觉 patch token、16 个语言 token、proprio token 和 task token；
- Transformer encoder + action-query decoder；
- 8 步连续 action chunk；
- 7 类专家阶段辅助头；
- target/goal cross-attention grounding；
- deterministic regression 与 Flow Matching 两种动作头。

### 训练可靠性

两次中断暴露出恢复训练的需要。随后保存 optimizer、scheduler、Torch RNG 和 DataLoader
generator 状态，并用 PID 锁、配置、seed、数据 fingerprint 和 metrics 末尾 epoch 验证恢复。

### 正式 deterministic 训练

- 运行：`outputs/tiny_vla_det_seed1_v2`
- 15 epochs；
- validation total loss：0.09449；
- validation action MAE：0.06443；
- phase accuracy：0.95139；
- grounding L2：0.03511。

这些是 episode-disjoint 离线结果，不等于闭环成功。

## 2026-07-28：离线很好，闭环却失败

### 现象

checkpoint 在 2 个 unseen-seed 闭环 episode 中均运行到 300 steps timeout，0/2 成功。
grounding L2 约 0.0348，目标与目标区热力图位置合理，但机械臂没有完成抓取。

证据：

- `outputs/tiny_vla_det_seed1_v2_failure_visual/episode_000.gif`
- `outputs/tiny_vla_det_seed1_v2_failure_trace/episode_000_trace.jsonl`

### 第一处分叉

step 0：

```text
model  = [-0.012,  0.165, 0.438, 0.010, 0.182]
expert = [ 1.000, -0.656, 0.478, 0.000, 0.150]
MAE ~= 0.38
```

模型随后在仍距方块约 0.06 m 时提前闭夹。约 step 50 时模型阶段头已预测 lift，但没有双指接触、
没有 grasp、也没有 lift；同一状态上的诊断专家始终处于 approach。

### 排除的假设

- 不是主要的视觉 grounding 失败：目标/目标区坐标误差较低。
- 不是推理太慢导致控制错过：修复线程配置后单步延迟满足当前重规划。
- 不是阶段标签完全不会：离线 phase accuracy 较高。

### 当时的推断

更符合 behavior cloning 的 covariate shift 和 phase shortcut：模型在专家状态上学到轨迹时间模式，
一旦自己的动作造成偏离，便会沿错误阶段继续推进。

## 2026-07-28：CPU 线程反向加速

### 现象

最初 2 episode 闭环运行超过 10 分钟。对 batch size 1 forward 做线程消融：

| Torch threads | 平均 forward |
| ---: | ---: |
| 1 | 约 19.55 ms |
| 2 | 约 1,230 ms，且不稳定 |
| 4 | 约 3,027 ms |
| 8 | 约 3,447 ms |

### 原因判断

小 batch Transformer 在该 Windows CPU 和当前系统负载下出现线程 oversubscription；线程调度
开销远大于矩阵计算收益。

### 修复

- 所有闭环和离线评测默认 `torch_threads=1`；
- 调用结束恢复原线程数；
- 正式报告 mean/p50/p95 延迟；
- 训练也新增显式 `torch_threads`，默认 1。

该优化把 batch-1 forward 从约 3.45 s 降到约 19.55 ms，约 176 倍。这个数字只适用于当前机器、
模型和输入，不外推为通用 PyTorch 结论。

## 2026-07-28：把平均 MAE 拆开

### 新工具

新增 `evla-eval-vla-offline`，输出：

- 全局 chunk MAE 与首动作 MAE；
- 按真实专家阶段分组；
- 按轨迹进度分组；
- 每个动作维度；
- 初始状态；
- phase confusion matrix；
- `summary.json`、`samples.jsonl` 和诊断图。

### 正式验证集结果

- 18 个 episode，2,777 帧；
- 全局首动作 MAE：0.05682；
- 初始状态首动作 MAE：0.33566；
- 初始 `dx` MAE：0.85173；
- 初始 `dy` MAE：0.76822；
- 初始 phase accuracy：100%；
- 初始 grounding L2：0.03470。

训练集 102 个初始帧的首动作 MAE 同样为 0.30565，因此不是 validation 泛化偶然。

### 结论

全局平均数被大量轨迹中段的小动作稀释。模型知道“当前是 approach”，也能找到目标，但对每条
轨迹最开始的大幅、方向相关动作学成了条件均值。闭环执行的恰好是这些关键动作。

## 2026-07-28：设计 DAgger 纠错

### 为什么不是规则硬修

可以写一个 phase-gated controller 强制 approach，但那会把模型失败藏在规则里，也不能证明
VLA 自己学会恢复。选择 DAgger：让 learner 访问状态，特权专家只提供标签。

### 实现细节

1. learner 与 expert 按概率混合执行，专家混合概率记为 beta；
2. 每个 learner-visited 状态都查询 expert action；
3. 保存 model action、expert action、executed action 和是否由 expert 执行；
4. correction 样本只把 action chunk 第一步标为 valid；
5. 不伪造“如果当时连续执行 expert，未来 7 步会是什么”的反事实标签；
6. 原演示与 correction 用 `ConcatDataset` 混合；
7. action loss 先按样本内有效步归一化，避免 1-step correction 天然比 8-step demo 少 8 倍权重；
8. 新增 validation 前 10 步 MAE，并保存 `best_early_action.pt`。

### 第 1 轮采集

- 路径：`datasets/dagger_v1_seed30000_beta050`
- checkpoint：原 deterministic `best_action.pt`
- beta：0.5；
- 24 个均衡任务 episode；
- 最多 240 steps；
- domain randomization：开启；
- 4,225 个 correction states；
- 混合控制成功：19/24；
- 旧模型对专家平均 MAE：0.10063；
- phase counts：`[1710, 439, 292, 666, 706, 383, 29]`；
- fingerprint：
  `8966d5fd6809f67600113c162a327550396f3a50596345ad726123ccb6fc0434`。

所有 7 个训练阶段均有覆盖，approach 状态得到最多补充，符合本轮目标。

## 2026-07-28：DAgger 微调运行时问题

### 实验配置

- 原训练样本：16,195；
- correction：4,225，重复 2 次；
- 从旧 `best_action.pt` 初始化新 run；
- learning rate：1e-4；
- batch size：64；
- Torch threads：1；
- 先跑 1 epoch 冒烟。

### 当前观察

在机器同时运行 8-worker OCR 任务时，shell 等待 15 分钟后超时，但训练进程仍正常占用一个
CPU core，run lock PID 有效。没有删除 lock 或启动第二个同目录训练器；继续监控后，epoch 在
1,026 s 时正常结束。

### 结果

- DAgger states 的首动作 MAE：0.10063 -> 0.08165；
- DAgger approach MAE：0.13592 -> 0.07735；
- validation 初始 MAE：0.33566 -> 0.33232，改善很小；
- validation 前 10 步 MAE：0.09714 -> 0.10619，反而变差；
- 初始动作预测的 `dx/dy` 方差仍远小于专家。

### 结论

朴素 DAgger 确实拟合了 learner-visited states，但没有解决精确初始方向，且牺牲了部分原专家
轨迹 early-state 表现。不能仅凭 correction loss 下降就宣称闭环改善。

## 2026-07-28：关键状态采样与空间动作瓶颈消融

### 假设

1. 初始/恢复状态在逐帧 BC 中权重太低；
2. grounding 辅助头虽然会定位，但坐标没有显式进入 action decoder；
3. 8 x 8 heatmap 可能缺少控制需要的亚 patch 精度。

### 共同设置

- 从原始 `best_action.pt` 初始化；
- 原数据 + 第 1 轮 DAgger；
- initial-state weight 25；
- 前 10 步 weight 5；
- correction weight 2；
- 每个 epoch 固定采样 6,000 states；
- seed 21，1 epoch；
- 除目标模块外其余配置相同。

### A/B 结果

| 模型 | 初始 MAE | 前 10 步 MAE | 初始 dx 相关系数 | 初始 dy 相关系数 |
| --- | ---: | ---: | ---: | ---: |
| 原 baseline | 0.33566 | 0.09714 | -0.17 | -0.13 |
| A：仅关键状态采样 | 0.32393 | 0.09460 | 0.32 | -0.11 |
| B：采样 + grounding-conditioned action | 0.31086 | 0.09682 | 0.49 | 0.26 |

B 将模型自己预测的 target/goal 坐标与 proprio 投影为 action-query residual。动作头不读取真值
坐标，因此没有 inference-time privileged leakage。新增 residual 最后一层零初始化，更新前行为
与旧模型一致。

### B 的开发集闭环

在 seed 33000-33005 的 6 个开发 episode 上仍为 0/6 success。3 个 episode 的末端最小距离降到
约 4.9-6.8 cm，但没有双指接触，phase 仍在错误状态下循环。该结果说明方向改善尚不足以完成
抓取，不能进入 final test。

证据：`outputs/ablation_grounded_action_e1_closed_loop_dev`

GitHub 可见证据：

- `docs/assets/vla_baseline_offline_diagnostics.png`
- `docs/assets/vla_grounded_failure_attention.png`
- `docs/assets/vla_grounded_failure.gif`
- `docs/assets/dagger_v1_mixed_rollout.gif`

### 用简单模型检验 grounding 假设

用 102 个训练 episode 的真值 target pixel 拟合线性回归，在 18 个 validation 初始帧上预测
expert action：

- overall MAE：0.05884；
- dx MAE：0.21334；
- dy MAE：0.08086。

因此固定相机下的 2D target pixel 足以提供大部分初始方向信息。换成旧 VLA 的预测 grounding
后总体 MAE 变为 0.22316，说明主要问题转移到 grounding 精度，而不是相机几何不可辨识。

### 轴向 grounding 诊断

validation 初始帧 target：

- 真值 x std 0.09488，预测 x std 0.10992，相关系数 0.95；
- 真值 y std 0.02762，预测 y std 0.00006，相关系数约 0.35。

8 x 8 attention 将 y 方向变化几乎压在同一 patch 行。为此新增连续坐标精修头：使用
role-specific grounded visual feature 和 coarse soft-argmax 坐标，预测限制在一个 patch 内的
残差；热力图仍保留用于监督和可视化。

### C：亚 patch 精修

1 epoch 后：

- grounding L2：约 0.034 -> 0.02102；
- target-y prediction std：0.00006 -> 0.01702；
- target-y 与真值相关系数：约 0.35 -> 0.50；
- 初始 action MAE：0.31207；
- 前 10 步 action MAE：0.09411。

坐标精度明显改善，但动作头只训练一轮，dy action 相关性尚未稳定。当前正式候选从 C checkpoint
继续 4 epochs；使用独立 run directory 和新 seed 22，仍只使用开发/validation 指标选模。

### C 后续 4-epoch 候选

运行：`outputs/tiny_vla_grounded_refined_dagger_v1_stage2_seed22`

- 额外 4 epochs，每轮 weighted replacement 采样 6,000 states；
- 参数量：1,189,139；
- validation first-action MAE：0.06202；
- validation initial MAE：0.21481；
- validation 前 10 步 MAE：0.07485；
- grounding L2：0.01972；
- 全验证首动作相关系数：dx 0.975、dy 0.933、dz 0.964、jaw 0.952；
- 推理 latency：p50 8.53 ms、p95 13.21 ms。

### 12-episode 开发闭环仍失败

seed 34000-34011，6 个任务各 2 条，结果 0/12。没有 episode 形成双指接触。

新的第一处分叉与旧模型不同：

- 多数策略保持 approach，不再普遍提前预测 lift/release；
- 最小 target distance 平均约 0.083 m；
- expert 的 approach waypoint 本来就在方块上方 0.085 m，所以该距离表示“到达上方附近”，
  不是完全没有接近；
- 但策略无法把 waypoint 的 xy/z 误差压进 6-8 mm 的专家转阶段阈值；
- 12 条中只有 1 条 diagnostic expert 进入 descend，该条 learned phase 仍停在 approach。

**决策：** 不直接扩大 final test。使用改进策略采 DAgger 第 2 轮，将 beta 从 0.5 降到 0.2，
重点采集 learner 在 waypoint 邻域和阶段边界造成的状态，再与 v1 一起训练。

### DAgger 第 2 轮

- 路径：`datasets/dagger_v2_seed35000_beta020`
- 24 个平衡 episode，最多 240 steps；
- beta：0.2；
- 5,239 个 correction states；
- 混合控制成功：8/24；
- phase counts：`[3501, 755, 264, 279, 274, 150, 16]`；
- learner/expert MAE：0.17606；
- fingerprint：
  `a52603e8c15167b5e2942a5abdf1577fef328db6d0a2d3c379eabca2f1450174`。

beta 降低后，成功率从 v1 的 19/24 降到 8/24，但这不是策略最终成功率；混合 rollout 的目的
是让 learner 更多地制造当前会遇到的状态。approach 占 3,501 帧，符合 waypoint 邻域纠错目标。

GitHub 预览：`docs/assets/dagger_v2_waypoint_corrections.gif`

### Waypoint 卡住的动作级证据

对 dev episode 002 逐步检查：

- x waypoint error：约 -50 mm -> -4 mm；
- y waypoint error：约 -23 mm -> -10 mm 后停滞；
- z waypoint error：约 14 mm -> 9 mm 后停滞；
- 停滞时 model y action 约 -0.03，expert 约 -0.34；
- 300 步只有少量符号变化，不是高频来回振荡。

因此更像 BC 对小残差动作的幅值衰减，而不是 IK 发散。优先让 DAgger v2 学这些残差；若 stage 3
仍失败，再在开发 seeds 上显式消融 Cartesian action gain，并保持环境成功阈值不变。

### Stage 3 配置

从 stage 2 `best_early_action.pt` 初始化，聚合原 demo、DAgger v1 和 v2。每轮 weighted
replacement 采样 8,000 states，correction weight 2，learning rate 1.5e-4，训练 4 epochs。
输出目录：`outputs/tiny_vla_dagger_v2_stage3_seed23`。

### Stage 3 结果：离线继续改善，闭环仍失败

Stage 3 第 4 个 epoch 在本轮各项动作指标上最好：

- validation 全局首动作 MAE：0.06683；
- validation initial MAE：0.20135；
- validation 前 10 步 MAE：0.08737；
- grounding L2：0.02136；
- 全局首动作相关系数：dx 0.971、dy 0.924、dz 0.962、jaw 0.952。

与 stage 2 相比，initial MAE 从 0.21481 改善到 0.20135，但 early MAE 从 0.07485 退化到
0.08737。不能只按 run 的新旧顺序选模型。

在同一组 seed 25000-25011 上做 12-episode、六任务均衡、execution horizon 1 的闭环评估：

- success：0/12；
- 双指接触：0/12；
- mean minimum target distance：0.08282 m；
- on-policy expert MAE：0.48248；
- latency：p50 7.32 ms，p95 13.94 ms。

这再次证明 episode-disjoint offline validation 不是闭环成功的充分条件。证据目录：
`outputs/tiny_vla_dagger_v2_stage3_seed23_closed_loop_dev12_gain1`。

### Cartesian action gain 消融

保持 checkpoint、12 个 seed、任务顺序、execution horizon 和成功条件不变，仅把执行的 dx/dy/dz
乘以 2 后裁剪：

| Gain | Success | Mean return | Mean minimum target distance | Contact episodes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0/12 | -0.0791 | 0.08282 m | 0 |
| 2 | 0/12 | -0.2688 | 0.09884 m | 0 |

增益 2 同时降低回报并增大最近距离，说明问题不是统一的动作幅值不足；策略在部分状态的方向也错，
放大动作只会放大错误。停止继续试 gain 4，避免把开发集调参伪装成算法进步。

### 状态表示诊断：控制空间与本体感觉不对齐

策略输出的是 Cartesian delta `[dx, dy, dz, wrist, jaw]`，但 12D proprio 只有 6 个关节位置和
6 个关节速度。模型若想知道“手相对目标在哪里”，必须同时完成：

1. 从关节角隐式学习正运动学；
2. 从 64 x 64 RGB 估计目标位置；
3. 对齐世界坐标与相机坐标；
4. 在 learner 偏离状态上输出小残差修正。

对约 1.19M 参数、120 条演示的 CPU 小模型而言，这个中间问题没有必要全部隐式解决。末端 XYZ
可由关节编码器和机器人 URDF/MuJoCo model 通过 forward kinematics 得到，不是 target/goal
privileged truth。因此新增可选 15D proprio：

```text
[6 joint positions, 6 joint velocities, normalized ee_x, ee_y, ee_z]
```

旧数据无需重采：正式 NPZ 的 `state[12:15]` 已保存与在线环境同尺度的 normalized end-effector
XYZ。训练/验证、DAgger 和闭环环境使用同一拼接函数。

### 向后兼容的权重扩展

从 12D checkpoint 初始化 15D 模型时：

- `proprio_projection.0.weight` 的旧 12 列原样复制，新 XYZ 三列置零；
- grounding-action projection 的旧 proprio 12 列原样复制；
- 旧 target/goal coordinate 四列移动到新输入的最后四列；
- 新 XYZ 三列置零。

因此扩展刚完成时，任意 XYZ 输入都不影响输出。回归测试验证 action chunk 和 grounding
coordinates 完全一致。Stage 4 的真实 validation epoch 0 也复现 Stage 3：

- total：0.09675903 -> 0.09675903；
- action MAE：0.07257417 -> 0.07257417；
- early MAE：0.08737309 -> 0.08737309；
- initial MAE：0.20135146 -> 0.20135146。

Stage 4 采用与 Stage 3 相同 seed、数据、采样权重、学习率和 4 epochs，只改变 proprio
维度。输出目录：`outputs/tiny_vla_cartesian_proprio_stage4_seed23`。

### Stage 4 结果：空间接近改善，阶段与对准仍失败

4 epochs 后按不同指标得到两个候选：

| Checkpoint | Global first MAE | Initial MAE | Approach MAE | Initial dy correlation |
| --- | ---: | ---: | ---: | ---: |
| `best_initial_action.pt` | 0.06859 | 0.17546 | 0.07985 | 0.409 |
| `best_action.pt` | 0.06514 | 0.18187 | 0.07273 | 0.168 |

Stage 3 的 initial dy correlation 只有 0.025，因此 15D proprio 确实让动作开始响应末端位置。两份
checkpoint 均在相同 seed 25000-25011 上做闭环，结果都为 0/12，且无双指接触：

| Checkpoint | Mean return | Mean waypoint distance | Expert left approach |
| --- | ---: | ---: | ---: |
| best initial | -0.0301 | 0.02328 m | 3/12 |
| best action | -0.0018 | 0.02501 m | 3/12 |

这不是“改造完全无效”：原 Stage 3 的诊断主要停在 approach waypoint 外；15D 模型已有 3 条
轨迹进入同状态 expert 的 descend 条件。但它仍不能稳定完成最后的 xy 对准和阶段动作切换。

动作级证据来自 best-action episode 003：

- step 19：expert 从 approach 转为 descend，要求 z action -1；
- model 到 step 43 也预测 descend，z action 随后接近 -1；
- 但 model x action 长时间保持正值，同状态 expert x action 已变为负值；
- step 60 左右模型在没有双指接触时进入 close-gripper；
- 最终没有接触，说明下降前的横向世界坐标修正仍错。

**结论：** 末端世界位姿解决了“手在哪里”的一半问题，target pixel 到 Cartesian action 的相机
几何仍由小模型隐式学习。下一轮增加可量化的 target/goal 3D world-grounding 辅助头，并将其
预测而非真值输入动作 residual。

### Stage 5：geometry-aligned action

新增三个可选模块：

1. 从 target/goal grounded visual features 预测两个 3D 世界坐标；
2. 将预测 3D 坐标和 15D proprio 投影到 action query residual；
3. 将预测 phase probability 投影到另一个 action query residual。

world labels 只用于训练监督，闭环 evaluator 从模型输出读取预测，不向 action path 注入 MuJoCo
真值。两个 action residual 的最后一层均为零初始化。真实 epoch 0 复现 Stage 4 的 action
MAE 0.071975、initial MAE 0.181866；随机 world head 的初始 3D L2 为 0.253 m。输出目录：
`outputs/tiny_vla_geometry_aligned_stage5_seed23`。

### Stage 5 结果：学会了近似几何，但精度不够直接闭环

最佳 initial checkpoint 在 validation 上达到：

- initial action MAE：0.16886；
- 全局 first-action MAE：0.06652；
- phase accuracy：94.63%；
- 3D world-grounding L2：0.03965 m；
- initial-state 3D world-grounding L2：0.03355 m。

相同开发 seeds 25000-25011 上直接执行 VLA 第一动作仍为 0/12，无双指接触。模型的 3-4 cm
世界坐标误差远大于下降阶段约 6-13 mm 的控制窗口。结论不是“3D 辅助头没学到”，而是
“学到的量级不足以替代精确相机几何”。保留 direct VLA 失败结果，不把后续 hybrid 成功冒充
端到端策略成功。

### 从端到端失败转向可审计的分层 VLA

新增 hybrid policy，并明确三层边界：

1. learned：Tiny-VLA 根据 RGB 和语言预测 target/goal 像素；
2. calibrated：固定相机射线与桌面平面求交，把像素变成世界坐标；
3. engineered：Cartesian visual servo、阶段状态机和接触反馈。

控制器只收到末端位置、双指接触、辅助抓取状态以及模型估计坐标；传入控制器的字典不包含
MuJoCo target/goal 真值。真值只写入 trace 用于事后评估。使用未做后处理的模型像素时，12 个
开发 episode 首次得到 3/12 成功，三条均完成接触、抓取、抬升、运输和放置。

最初报告的 target world error 为 40.35 mm，后来发现统计口径错误：成功后方块已被抬起，
但 evaluator 仍把预测像素投到桌面高度，导致成功轨迹反而被惩罚。修复后只在抓取前统计 target
平面误差，并单列 descend/close 误差；goal 平面误差可全程统计。这次修正没有改变任何动作，
只修复测量定义。

### 只用训练集拟合的像素校正

假设：模型 grounding 存在稳定像素偏差，相机反投影会把几像素误差放大到厘米级。实现
`evla-fit-grounding-calibration`：

- 参数只在 102 个 train episodes 上拟合；
- 18 个 validation episodes 只用于 identity / affine 模型选择；
- 闭环开发和最终测试 scene 的 target/goal 标签不参与拟合；
- calibration JSON 保存 checkpoint SHA256、dataset fingerprint、矩阵、划分和 runtime；
- target 只使用 approach、descend、close 三个抓取前阶段；
- 输出 validation 误差图和 13 mm 抓取窗口覆盖率。

全局 affine 在 validation 上把 target 世界 XY 平均误差从 31.56 mm 降到 27.83 mm，
13 mm 窗口覆盖率从 14.8% 提到 17.9%；goal 从 2.01 mm 降到 1.17 mm。同一开发集仅加入
该校正后从 3/12 提升到 7/12。

又尝试加入 target color / goal side 的语言条件线性特征。它在 validation 上进一步把 target
误差降到 27.19 mm、13 mm 覆盖率升到 19.9%，但闭环从最终候选的 10/12 降到 8/12。因此拒绝
语言条件版本。离线代理指标更好不等于闭环控制更好。

### 接触失败后的局部搜索

原 controller 关闭夹爪 35 步仍无接触后，会回到同一个有偏视觉落点重试。新增不读取物体真值
的局部搜索：第 0 次使用原估计，后续按 18 mm 网格搜索 `-x, +x, -y, +y` 和对角方向；每次
下降前锁定当前估计与 offset，失败后解锁。仅搜索、保持 300 步预算时达到 9/12。

开发集消融：

| 配置 | Success | 关键结论 |
| --- | ---: | --- |
| learned pixel + camera geometry | 3/12 | 首次真实闭环成功 |
| + train-only global calibration | 7/12 | 系统像素偏差可校正 |
| + contact search，未标定 | 9/12 | 重复同一落点是主要故障 |
| calibration + search，300 steps | 9/12 | 两种修复有重叠 |
| calibration + search，400 steps | 10/12 | 晚抓取有足够运输时间 |
| 同上，close timeout 18 -> 8 | 6/12 | 机械闭合和接触建立存在延迟 |
| language-conditioned calibration + search | 8/12 | 离线更好但闭环更差，拒绝 |

8-step close timeout 版本有 8 条 episode 建立接触、8 条抓取，但只有 6 条抬升和成功，说明
“发出闭合命令”与“关节完成闭合并形成稳定接触”不是同一个时刻。最终开发配置锁定为全局
train-only calibration、18 mm 搜索、18-step close timeout、400-step episode budget。

### Final Test v1：开发集 83.3% 没有复现在 60 条未见种子上

冻结配置后，使用此前未参与训练、标定或调参的 seeds 50000-50059 做六任务均衡测试，每个
`color -> side` 10 条。运行前声明不再根据中间结果改参数；60 条全部完成、锁文件正常移除后才
读取 summary。

- success：30/60 = 50.0%；
- Wilson 95% CI：37.7%-62.3%；
- 双指接触 / 抓取 / 抬升：32/60；
- 抓取后的最终放置：30/32；
- raw / calibrated pixel L2：0.02540 / 0.02348；
- pre-grasp target world XY error：35.02 mm；
- descend/close target world XY error：31.66 mm；
- goal world XY error：0.90 mm；
- inference latency：p50 9.21 ms，p95 20.27 ms；
- 总运行时间：702.4 s，19,850 次 CPU 推理。

任务分项为 red-left 6/10、red-right 6/10、green-left 5/10、green-right 4/10、blue-left
3/10、blue-right 6/10。30 个失败中 28 个从未形成双指接触，另外 2 个在较晚抓取后因 400 步
预算结束停在 transport。结论非常明确：一旦抓稳，搬运链成功率为 93.8%；当前主要瓶颈仍是
target grounding，不是 IK、抬升或目标区定位。

开发集 10/12 与 final 30/60 的差距说明 12 条开发样本方差很大。简历只能写 30/60 final，
10/12 必须标为 dev。该 final-v1 从现在起视为已消费测试集，不用于 Stage 6 超参数选择。

下一步采用新的协议：

1. Stage 6 只使用原 train/validation 数据训练高分辨率 grounding；
2. 用另一组开发 seeds 做闭环选择；
3. 配置锁定后用第三组未见 seeds 做 final-v2；
4. final-v1 永久保留，不能被覆盖或改名隐藏。

### Stage 6：冻结主干的高分辨率语言 grounding

Final-v1 的 30 个失败里有 28 个从未形成双指接触，因此新增高分辨率分支，而不是继续改运输
状态机。结构为：

1. 原 8 x 8 Transformer attention 负责根据语言选择 target/goal 语义；
2. 轻量 CNN 从原 RGB 产生 16 x 16 spatial features；
3. coarse grounded feature 投影成 query，与高分辨率 key 做点积 attention；
4. 16 x 16 soft-argmax 坐标通过一个标量门控与旧坐标融合。

门控初始化为 0。从 Stage 5 初始化时，action chunk 和 grounding coordinates 逐元素不变，
回归测试覆盖该合同。第一次 256-sample smoke 让全模型一起更新，action MAE 从 0.0746 退化到
0.0877，说明随机新分支不应拖动已收敛主干。正式 Stage 6 冻结 Stage 5，仅训练 stem/key/query
和门控，共 145,601 个可训练参数；冻结主干保持 eval mode，避免 dropout 造成随机漂移。

正式配置：seed 24、8 epochs、每轮 12,000 mixed demo/DAgger states、batch 64、learning rate
3e-4。validation heatmap NLL 从接近均匀分布的 5.546 降到约 0.24，合并 grounding L2 从
0.02139 降到约 0.01939，action MAE 保持约 0.0744。target-only、抓取前 validation：

| Checkpoint | Target pixel L2 | Target world XY | Within 13 mm |
| --- | ---: | ---: | ---: |
| Stage 5 | 0.04040 | 31.56 mm | 14.8% |
| Stage 6 epoch 4 | 0.03513 | 27.78 mm | 18.1% |
| Stage 6 epoch 7 | 0.03457 | 27.28 mm | 17.7% |
| Stage 6 epoch 8 | 0.03456 | 27.27 mm | 18.2% |

Stage 6 epoch 8 的 train-only calibration 在 validation 上进一步达到 24.30 mm 和 23.3%
within-13-mm。用全新 seeds 26000-26023 做配对开发评估：

| Pipeline | Success | Contact | Pre-grasp error | p50 / p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Stage 5 + global calibration | 11/24 | 16/24 | 29.76 mm | 9.27 / 16.74 ms |
| Stage 6 + train-only calibration | 14/24 | 16/24 | 26.27 mm | 18.47 / 26.79 ms |

Stage 6 提升 3 条最终成功并把平均 episode 从 341 步降到 319 步，但推理延迟约翻倍；50 Hz
控制周期为 20 ms，因此 p50 满足、p95 偶尔超期。此项是精度/实时性的真实权衡，不删除。

训练过程中还暴露了 checkpoint 选择缺口：训练器原来只保存 best total/action/early/initial，
没有 best grounding。epoch 6/7 通过人工复制完整 `last.pt` 保留，随后训练器新增
`best_grounding.pt` 自动留档，防止以后再次覆盖关键候选。

### Final Test v2：接触增加，但提升尚未达到统计显著

锁定 Stage 6 epoch 8、对应 train-only calibration 和原 18 mm recovery search 后，使用全新
seeds 60000-60059 做第二组 60-episode 六任务均衡测试：

| Metric | Final-v1 Stage 5 | Final-v2 Stage 6 |
| --- | ---: | ---: |
| Success | 30/60 (50.0%) | 34/60 (56.7%) |
| Wilson 95% CI | 37.7%-62.3% | 44.1%-68.4% |
| Contact / grasp / lift | 32/60 | 39/60 |
| Pre-grasp target XY error | 35.02 mm | 27.06 mm |
| Descend/close target XY error | 31.66 mm | 25.17 mm |
| Mean episode length | 330.8 | 312.1 |
| p50 / p95 latency | 9.21 / 20.27 ms | 18.17 / 48.27 ms |

Stage 6 比 v1 多成功 4 条，并多产生 7 条有效抓取，和感知误差下降方向一致。但两组 Wilson 区间
明显重叠，seed 集也不同，不能声称统计显著。抓取后的失败从 2 条增到 5 条：高分辨率推理更慢，
一些搜索后晚抓取的 episode 在 transport 阶段耗尽 400-step 总预算。当前系统从单一“看不准”
瓶颈变成“感知更准但实时性/时间预算受限”的多目标权衡。

简历推荐写 final-v2 34/60，同时注明 CPU p50 18.17 ms 和 hybrid policy 边界；final-v1、开发集
和 direct VLA 0/12 都保留在 registry，不能只展示最高开发结果。

### 冻结后的 domain-randomized robustness test

在不修改 Stage 6 checkpoint、calibration、18 mm search、18-step close timeout 和 400-step
budget 的前提下，使用新 seeds 70000-70029 开启环境 domain randomization。六个任务各 5 条：

- success：18/30 = 60.0%；
- Wilson 95% CI：42.3%-75.4%；
- contact / grasp：23/30；
- lift：22/30；
- pre-grasp target XY error：25.88 mm；
- descend/close target XY error：24.77 mm；
- 12 个失败中，7 个从未接触，1 个接触后未抬升，4 个抬升后在 400 步预算内未完成；
- blue-right 为 1/5，但每任务只有 5 条，不能据此断言颜色或目标侧系统性失效。

18/30 数值高于 clean final-v2 的 34/60，但两个 Wilson 区间大幅重叠，样本也来自不同 seed，
因此结论只能是“在这 30 条随机化场景中没有观察到明显崩溃”，不能写“domain randomization
提升成功率”。

这次运行由隐藏后台进程启动，同时前台在复制文件、运行小型测试和编辑文档。模型仍设置一个
Torch thread，但 host load 不受控；记录到 p50/p95 43.44/104.92 ms，明显慢于前台 clean run 的
18.17/48.27 ms。由于存在并发负载混杂，这组 latency 只描述该次运行，不能归因于 domain
randomization。以后做性能对照应前台独占运行、预热、固定电源模式并记录系统负载。

### 记录工具本身的 Windows 问题

新增自动 runtime metadata 后，`platform.platform()` 在 Windows pytest 退出阶段触发
`0x80070006` invalid handle。测试断言虽通过，但进程退出不干净。原因是该函数内部可能再启动
系统查询子进程；改为分别读取 `system/release/version/machine`，避免额外进程。可复现工具本身
也必须经过测试，不能假设“只写 JSON 的代码不会影响实验”。

### Calibration 保存了 SHA，但 evaluator 原来没有执行合同

将最终 checkpoint 和 calibration 提升到仓库根目录后做可移植性审计，发现 calibration JSON
保存了 `checkpoint_sha256`，但 `AffinePixelGroundingCalibration.load()` 只验证 schema 和矩阵
shape；hybrid evaluator 没有比较实际 `.pt`。如果用户把 Stage 5 权重和 Stage 6 calibration
混用，程序仍会运行并产生难以解释的结果。

新增 `verify_checkpoint()`：有 SHA 字段时流式计算实际权重 hash，不一致立即抛错；旧测试构造的
无 SHA calibration 仍可用于单元测试和向后兼容。hybrid evaluator 在创建环境前执行验证。回归
测试先用匹配字节通过，再修改 checkpoint 内容并断言拒绝。提升后的
`checkpoints/tiny_vla_stage6_highres.pt` 实际 SHA 与 JSON 中
`47bef658...f4476` 完全一致。

这次修复的教训是：metadata 被写进文件不等于约束已经生效。审计可复现性时要沿着读取路径确认
它真的被验证，而不只检查输出里“有这个字段”。

### ROS2 arm bridge：消息能发布不等于接口已经验证

在 WSL2 Ubuntu 24.04 / ROS2 Jazzy 中重新执行：

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

包可以干净构建，但第一次把多个步骤拼成一条 Windows -> WSL 命令时失败。为让 WSL 找到项目，
当时手工覆盖了 `PYTHONPATH`；这也覆盖了 ROS2 安装环境中的 Python metadata 路径，导致
`ros2cli` 无法发现入口点。结论是：ROS2 overlay 应由 `setup.bash` 管理，不能为了导入本项目
随意替换整条 `PYTHONPATH`。

第二次 arm probe 实际已经成功，但外层 PowerShell 在 Bash 收到命令前展开了 `$!`，用于记录
launch PID 的变量变成空值。probe 退出后 bridge 仍在后台运行。确认命令行后只清理了本次启动的
WSL PID，没有按进程名批量终止其他程序。这个问题说明跨 shell 自动化必须明确“变量由哪一层
shell 展开”，不能依赖长行内命令。

随后把流程固化为 `scripts/run_ros_arm_probe_wsl.sh`：

1. 在 Bash 文件内部启动 launch 并立即保存 `$!`；
2. 用 `trap` 在成功、失败或中断时发送 `SIGINT`，超时后再 `SIGTERM`；
3. source ROS2 和 workspace overlay 后才启用 `set -u`；
4. 启动 5 秒后运行独立 `arm_probe`，把消息转成数组并执行语义检查。

脚本第一次运行仍失败：在 source `/opt/ros/jazzy/setup.bash` 之前启用 `set -u`，ROS 的 setup
脚本读取尚未定义的 `AMENT_TRACE_SETUP_FILES` 时被 Bash 当作错误。将开头改为
`set -eo pipefail`，完成两个 setup 后再 `set -u`，问题消失。最终 clean run 正常退出，确认
没有遗留本次 arm ROS 进程。

`outputs/ros_arm_probe_wsl_jazzy_clean/summary.json` 的 11/11 检查通过：

- 六个关节名称、位置数量和有限值；
- RGB 为 `rgb8`、深度为 `32FC1`，均为 128 x 128；
- 深度包含有限正值，CameraInfo 焦距为正；
- 自然语言任务与结构化 target/goal metadata 非空且合法；
- TF 中存在 `world -> camera_color_optical_frame`；
- 共观察到 12 对 robot/camera/object TF。

这项验证的作用是证明 MuJoCo 与 ROS2 消息边界真实跑通，不是证明真机部署。RGB/depth 图、
summary 和可重复脚本一并保存，面试时可从“构建成功、话题存在、消息语义正确、进程可清理”
四层解释验证方法。

第一次 depth 证据虽然数值正确，但灰度图视觉上几乎全白。原因是场景远平面和近处机械臂的动态
范围很大，单通道预览不利于人眼分辨。将 2%-98% 分位数归一化后映射成 near-warm/far-cool RGB，
原始 ROS `32FC1` 数据和检查完全不变。

最初把 `depth_to_rgb` 放进现有 `visualization.py`，单元测试直接导入时暴露循环：
`visualization -> experts -> env -> control -> experts`。ROS node 若采用该导入也会启动失败。颜色
映射不应依赖策略和环境，因此移到只依赖 NumPy 的 `image_utils.py`，测试 shape、dtype、远近颜色
顺序和全无效输入。重新 `colcon build` 后 clean probe 再次 11/11 通过，脚本退出无残留进程，
新伪彩图提升到 `docs/assets/ros2_arm_depth.png`。

### Expert grasp-mode 配对基准：辅助约束的影响有多大

此前 README 中存在没有被提升到 `results/` 的 99/100 旧数字。为避免继续引用不可追溯结果，
重新用 seeds 10000-10099、六任务轮换、300-step budget 做配对基准，只改变 `grasp_mode`：

| Mode | Success | Wilson 95% CI | Mean length |
| --- | ---: | ---: | ---: |
| `contact_assisted` | 98/100 | 93.0%-99.4% | 164.19 |
| strict `contact` | 38/100 | 29.1%-47.8% | 260.98 |

辅助模式的两次失败都停在 transport。strict contact 的 62 次失败分布为 approach 9、
descend_grasp 16、close_gripper 8、lift 17、transport 11、descend_release 1。六任务 strict
success 在 29.4%-43.8% 之间，没有单一颜色完全失效。说明 expert 的运动学路径基本合理，但
真实摩擦抓取、抬升稳定性和运输滑落仍是主要难点；辅助等式约束把这个物理瓶颈大幅简化。

运行过程还有一个可复现性问题：第一次 100-episode assisted 命令的工具等待上限为 120 秒，
外层命令返回 timeout 时已经写到第 49 条。Windows venv launcher 的子 Python 仍在运行，run
directory 保持 active lock。没有删除目录或启动第二个 writer，而是检查精确命令行 PID，并等待
原进程写出 100 条和 summary。最终结果完整、没有重复 episode。这与“训练程序自身崩溃”不同，
也说明外层工具 timeout 不等于子进程一定终止。

旧 99/100 已被正式 98/100 替换。简历若提辅助 expert，必须同时给出 strict 38/100 或明确
`contact_assisted`，不能只展示较高数字。

### PPO reach 三 seed：单 seed 75% 掩盖了训练不稳定

固定 state-based reach、100,000 配置步（实际 rollout 对齐后 100,352）、8 envs、256 rollout、
同一 PPO 超参数和每 seed 20 条独立 final evaluation，补齐 seed 2/3：

| Seed | Final success | Mean return | Mean length |
| ---: | ---: | ---: | ---: |
| 1 | 15/20 (75%) | 4.058 | 61.85 |
| 2 | 4/20 (20%) | 1.222 | 90.05 |
| 3 | 16/20 (80%) | 4.268 | 51.45 |

seed-level success mean 为 58.3%，sample std 为 33.3 个百分点；pooled episode 为 35/60，
Wilson 95% CI 45.7%-69.9%。主结论必须使用 seed mean/std，因为 60 条 episode 条件于三个已训练
policy，并不等价于 60 个独立训练 seed。较大的标准差说明 100k steps 下 PPO 对初始化和采样
轨迹敏感，下一步应增加训练预算、调 reward/entropy 或报告 learning curve AUC，而不是挑 seed 3。

运行时间也暴露了主机负载/热状态差异：seed 1 约 156 s，seed 2 约 724 s，seed 3 约 1272 s；
训练配置相同，steps/s 却约 662、160、最终累计约 79。seed 2 的外层等待 timeout 后，子 Python
继续持有 active lock，最终正常写出 summary；seed 3 因此改用 stdout/stderr 重定向的隐藏进程
并监控 metrics。性能差异不影响 episode success 的配置语义，但说明正式 wall-clock benchmark
需要独占机器、固定电源/温度并记录 Torch threads。

聚合时还发现 schema 演进问题：seed 1 的旧环境 JSON 没有
`include_end_effector_position_in_proprio`，seed 2/3 明确写 `false`。它们语义相同，但原聚合器
按字典全等会拒绝。现在只为这个有明确旧默认值的字段做规范化；显式 `true` 仍由测试确认拒绝，
避免为了凑三 seed 悄悄忽略真实配置差异。

### 发布前可移植性与测试审计

使用将提交到 Git 的根目录文件，而不是 `outputs/`：

```powershell
evla-eval-vla-hybrid `
  --checkpoint checkpoints\tiny_vla_stage6_highres.pt `
  --grounding-calibration checkpoints\tiny_vla_stage6_calibration.json `
  --output-dir outputs\packaged_checkpoint_smoke_seed60000 `
  --episodes 1 --seed 60000 --video-episodes 0 `
  --max-episode-steps 400 --torch-threads 1 `
  --recovery-search-radius-m 0.018 --close-retry-steps 18
```

SHA 合同通过，episode 115 steps 成功。这是 packaging smoke，不是新的独立成功率证据，因为
seed 60000 已属于 final-v2。

第一次从仓库根运行 `pytest -q` 时，测试尚未收集就出现三个 `WinError 1920`。pytest 递归进入
WSL `colcon --symlink-install` 的 Linux symlink，Windows 无法 stat。Ruff 已排除 build/install/log，
但 pytest 没有 test path 配置。给 `pyproject.toml` 增加 `testpaths = ["tests"]` 后，完整 50 tests
通过。

随后清理两类 warning：

- expert 1-episode smoke 对另外五个空任务执行 `np.mean([])`；现在空任务明确写 `success_rate:
  null`，测试确认，不再生成 NaN；
- Transformer `norm_first=True` 本来就禁用 nested-tensor fast path；构造器显式设置
  `enable_nested_tensor=False`，不改参数或 state dict，10 个模型/evaluator 测试通过。

发布审计还包括：

- 21 个 Markdown 文件的本地链接全部可解析；
- 157 个文本文件未发现 token、API key、password 或 private-key 模式；
- 无单文件超过 50 MiB；
- WSL2 ROS2 Jazzy `colcon build --symlink-install` 成功；
- artifact manifest 对 checkpoint、三份数据、results 和 docs assets 逐文件计算 SHA256。

## 仍在进行的工作

- deterministic / Flow Matching、action horizon 和 execution horizon 消融；
- 最终 claim ledger、图像、Git commit 和 GitHub 发布。

这些条目完成前不得在简历中写成已完成结果。
