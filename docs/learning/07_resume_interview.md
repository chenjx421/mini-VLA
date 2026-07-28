# 简历与面试答辩

## 1. 项目一句话

> 从零实现了一个 CPU 可复现的具身智能研究栈：在 MuJoCo 中搭建 SO-ARM100 语言条件抓取放置
> 环境，完成专家示范采集、PPO、Tiny-VLA action chunk、高分辨率语言 grounding 与混合闭环，
> 并通过 ROS2 桥接相机、关节和 LiDAR 数据，使用 slam_toolbox 验证移动平台建图。

这句话只描述范围，不塞未经验证的结果。

## 2. 推荐项目名

**EmbodiedVLA-RL: SO-ARM100 MuJoCo Manipulation and ROS2 SLAM**

中文可写：

**基于 MuJoCo、Tiny-VLA 与 PPO 的具身操作及 ROS2 SLAM 系统**

## 3. 简历版本

下面数字均能在 `results/` 中找到 JSON/JSONL 证据，面试时不要脱离作用域：

```text
具身操作与 Mini-VLA 仿真研究项目                         个人项目
Python / PyTorch / MuJoCo / Gymnasium / ROS2 Jazzy / slam_toolbox

- 基于官方 SO-ARM100 关节模型构建语言条件抓取放置环境，实现 5D 笛卡尔增量动作、
  阻尼最小二乘 IK、MuJoCo 接触动力学、RGB-D/本体状态观测及光照与物理参数随机化。
- 从头实现 PPO 的 tanh Gaussian actor、GAE、策略/价值裁剪、KL 早停和多环境 rollout；
  在 state-based reach 上完成 3 seeds x 100,352 环境步训练，独立评测成功率
  58.3% +/- 33.3%（mean +/- sample std），量化并保留初始化敏感性。
- 采集并审计 120 条平衡专家轨迹，共 18,972 帧、6 种颜色-目标组合；按 episode 分层切分，
  保存 SHA256 数据指纹、归一化统计和逐动作维度指标，避免相邻帧数据泄漏。
- 从零实现 1.39M 参数 Tiny-VLA，将图像 patch、语言 token 与 15D proprioception 融合，
  通过 Transformer decoder 预测 8 步连续 action chunk；新增冻结主干的 16x16 语言 grounding
  分支，将 validation target 世界坐标误差由 31.56 mm 降至 27.27 mm。
- 构建 VLA-grounded hybrid policy：仅由 RGB+语言预测 target/goal，结合 train-only 标定、
  相机反投影、Cartesian visual servo 与接触恢复；在六任务均衡的 60 条未见仿真轨迹上完成
  34/60，Wilson 95% CI 44.1%-68.4%，CPU 推理 p50 18.17 ms。
- 开发 ROS2 arm/mobile bridge，发布 JointState、RGB-D、CameraInfo、LaserScan、Odometry 和 TF；
  arm probe 的消息/数值/TF 11/11 检查通过；接入 slam_toolbox 生成 5 cm 分辨率占据栅格，
  在强合成漂移下记录 odom 1.59 m、SLAM 0.84 m 的末端位置误差。
```

若简历版面只能放四条，保留 MuJoCo/控制、PPO、Tiny-VLA hybrid final、ROS2/SLAM 四条，把数据
审计合并进 VLA 条目。PPO 必须同时写较大的 seed 标准差，不能只写 pooled 35/60。

## 4. 30 秒介绍

> 我想做一个不是只调库的 VLA 项目，所以把感知、策略、控制和仿真闭环都实现了。场景里同时有
> 三种颜色方块和左右目标区，语言真正决定操作对象。模型把 64 个视觉 patch、语言和关节状态
> 送进小型 Transformer，研究直接预测 8 步 action chunk；直接策略闭环失败后，我定位到厘米级
> grounding 误差，改成冻结主干的高分辨率语言定位和可审计几何控制，在 60 条未见轨迹上做到
> 34 次成功。我同时实现 PPO，并用 ROS2 把机械臂传感器和独立移动 SLAM 平台接到标准
> topic/TF。项目的重点是成功和失败都有 seed、checkpoint、逐步轨迹与明确边界。

## 5. 两分钟介绍

按四层讲：

1. **任务**：为什么相同场景配不同语言，避免模型忽略文本。
2. **仿真控制**：5D action、DLS-IK、50 Hz 控制、contact 与 contact-assisted 口径。
3. **算法**：PPO、episode-disjoint BC、Tiny-VLA token、action chunk、grounding、flow matching。
4. **证据**：数据审计、direct 失败、hybrid final、Wilson 区间、ROS probe/map、已知限制。

结尾主动说限制：

> 当前结果是 MuJoCo 仿真，contact-assisted 抓取与严格接触分开报告；立方体任务不需要 wrist
> 旋转；34/60 是 VLA grounding 驱动的 hybrid policy，不是 direct action decoder 成功率；
> Stage 6 的 CPU p95 也超过 20 ms，下一步会做模型蒸馏、异步推理和真机标定。

主动说清边界通常比等面试官拆穿更有说服力。

## 6. 高频追问

### “这就是行为克隆，为什么叫 VLA？”

因为 policy 的条件输入同时包含视觉、语言和机器人状态，输出可执行动作序列，语言反事实会改变
目标 grounding 和动作。训练方法确实主要是 behavior cloning，VLA 描述模型输入输出，不等于
必须使用某一种优化算法。

### “为什么不用大 VLM？”

目标是完整理解和在 CPU 上复现训练闭环。1.39M 参数模型能暴露 tokenization、fusion、action
decoder、grounding 和部署逻辑。大 VLM 是后续替换 backbone 的扩展，当前项目不把小模型冒充
foundation model。

### “为什么 action chunk？”

它能建模动作时间相关性、降低重规划开销，并避免每步独立回归的抖动。预测 horizon 和执行
horizon 分离，执行太多会开环漂移，所以需要消融。

### “Flow Matching 比回归好在哪里？”

确定性回归倾向于条件均值，多模态动作可能被平均。Flow Matching 学习从噪声到动作分布的速度
场，可产生不同合理轨迹，但采样要多次 forward，CPU 延迟更高。是否更好由同数据闭环结果决定。

### “PPO 里最容易写错什么？”

- tanh 后 log probability 的 Jacobian；
- terminal 与 timeout 的 bootstrap；
- GAE 穿过 reset；
- ratio 使用错误动作或新旧 log prob；
- value clipping；
- 更新时重复使用变化后的 old policy 数据。

仓库对 GAE 终止语义和 update shape 有单元测试。

### “PPO 为什么一个 seed 80%，另一个只有 20%？”

三次使用相同超参数和 100,352 steps，但初始网络、采样轨迹和早期探索不同。PPO 是 on-policy，
早期到达的状态会改变后续训练分布；100k 预算下 seed 2 没有追上另外两条曲线。当前证据不能断言
具体是局部最优还是训练预算不足，所以我报告 58.3% mean 和 33.3% sample std，不挑 80%。
下一步会延长 seed 2、比较 learning curve AUC，并做 entropy/reward 消融。

### “为什么验证集不能随机抽帧？”

相邻帧几乎相同，随机抽帧会让同一 episode 同时出现在 train 和 validation，产生严重泄漏。这里按
完整 episode 划分，并在 6 种任务内分层。

### “grounding attention 是可解释性吗？”

它是受监督的空间诊断量，能显示目标/目标区定位，但 attention 本身不是因果解释。还需要同图像
换语言的 counterfactual test 和遮挡/扰动实验。

### “34/60 是不是端到端 VLA 成功率？”

不是。Stage 5 direct action policy 在 12 条开发轨迹中是 0/12。34/60 对应
VLA-grounded hybrid policy：模型根据 RGB 和语言输出 target/goal pixel，train-only calibration
与相机几何恢复世界坐标，visual servo 和接触状态机输出 5D action。控制路径不读取仿真目标真值。
我保留 direct 失败，是因为它揭示了小数据 VLA 在毫米级操作上的空间精度瓶颈。

### “为什么不直接把真值坐标给控制器？”

那会把视觉语言任务退化成 privileged state control，无法证明 grounding 能力。真值只在训练
auxiliary label 和评测 trace 中出现；部署 action path 使用模型像素、相机标定、合法 proprio
和接触反馈。summary 的 `policy_boundary` 会机器可读地记录这条边界。

### “高分辨率分支真的有效吗？”

验证集 target 世界误差从 31.56 mm 降到 27.27 mm；新的 24 条配对开发集 success 从 11/24
增到 14/24；独立 final 的接触从 32/60 增到 39/60。但 final success 仅从 30/60 到 34/60，
Wilson 区间重叠，且 p95 延迟从 20.27 ms 增到 48.27 ms，所以我只说观察到 accuracy/latency
trade-off，不说显著提升。

### “SLAM 和机械臂任务有什么关系？”

固定桌面机械臂不需要 SLAM，所以我没有强行把它放进控制链。我另建移动传感平台，通过标准 ROS2
LaserScan、Odometry 和 TF 接入 slam_toolbox，学习系统集成、定位和建图；未来移动操作平台可以
把两条链合并。

### “contact-assisted 是作弊吗？”

它是显式标注的实验简化。辅助约束只有双指接触且夹爪关闭后才启用，不是直接 teleport。它适合
先隔离研究 VLA，但不能代表严格接触或真机成功，所以另有 `contact` 口径和 sim-to-real 限制。

### “项目里你遇到的真实 bug？”

优先讲下面五个中的两个，并准备代码和日志路径：

1. 初始物体偶尔落在目标区，旧成功条件产生 1-step 假成功。修复为初始化排除目标区，并要求先
   lift 再 place，加入回归测试。
2. 随机 DataLoader 只缓存 8 个 NPZ，导致反复解压。改为可配置 episode cache，正式数据在
   16 GB 内存机器上常驻。
3. 断电后 stale lock 与 checkpoint 无法续训。加入 PID 检查、配置/数据指纹验证及
   optimizer/scheduler/RNG 状态保存。
4. 成功后方块离开桌面，旧 evaluator 仍按桌面平面反投影，导致成功轨迹定位误差反而更大。
   将指标拆成 pre-grasp、descend/close 和 goal，策略动作不变。
5. 跨 PowerShell/WSL 启动 ROS2 时 `$!` 被外层展开，probe 成功却留下 launch 进程。改为带
   `trap` 的 Bash 脚本，并把 `set -u` 放到 source ROS overlay 之后。

这类回答比背模型结构更能证明你真的做过。

## 7. 白板题

必须能独立写出：

1. DLS-IK：
   \[
   \Delta q=J^\top(JJ^\top+\lambda^2I)^{-1}\Delta x
   \]
2. GAE：
   \[
   \hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}
   \]
3. PPO clipped objective；
4. 图像 patch token 数量；
5. Flow Matching 插值和速度目标；
6. `map -> odom -> base_link -> laser` TF 树；
7. pixel ray 与桌面平面的求交公式；
8. ridge affine calibration 的闭式解和数据划分。

## 8. 不能写的内容

在没有对应证据前，不要写：

- “部署到 SO101 真机”；
- “实现端到端导航抓取”；
- “达到 SOTA”；
- “训练 VLA foundation model”；
- “严格接触抓取成功率等于 assisted 成功率”；
- “SLAM 精度提升 xx%”但不写轨迹、时间对齐和测试条件；
- 单 seed 结果包装成稳定结论。
- “VLA 闭环成功率 56.7%”却不写 hybrid policy 边界；
- “高分辨率模型显著提升”但没有显著性检验且置信区间重叠。

## 9. 把项目真正变成你的

至少亲手完成并提交：

1. 有朝向物体和 wrist 控制；
2. 一个 language counterfactual evaluator；
3. 一次 PPO 奖励消融；
4. 一次 execution horizon 消融；
5. 一个你自己定位并修复的失败案例；
6. 10 分钟无稿录屏讲解。

能修改、预测结果、解释失败，才算拥有这个项目。

## 10. 用工程日志准备追问

完整过程记录在 `docs/engineering/project_journal.md`。准备面试时不要只背最终数字，而要从日志中
任选一个问题，按“现象、假设、证据、修复、结果、限制”无稿讲清楚。已提炼但仍需亲自验证的
STAR 素材在 `docs/engineering/interview_stories.md`。
