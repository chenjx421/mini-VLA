# 面试复盘素材

这里只提炼已经在 `project_journal.md` 中留下证据的问题。先亲自跑命令和读代码，再使用这些回答。

## 故事 1：离线指标好但闭环 0/2

### Situation

Tiny-VLA 的 episode-disjoint validation action MAE 约 0.064、phase accuracy 约 95%，但 unseen
MuJoCo 闭环 2/2 timeout。

### Task

找出第一处决策错误，判断是视觉定位、控制频率、阶段预测还是行为克隆分布偏移。

### Action

1. 给闭环 evaluator 加入不执行的 privileged expert，只做同状态动作对照；
2. 保存逐 step 的模型动作、专家动作、阶段、末端位置、目标位置和接触状态；
3. 做第一处分叉分析；
4. 新增按初始状态、阶段和轨迹进度拆分的离线 evaluator；
5. 发现初始首动作 MAE 约 0.336，而全局只有约 0.057；
6. 实现 DAgger learner-state relabeling，而不是用规则覆盖模型动作。

### Result

定位到关键状态欠采样与 covariate shift。第 1 轮得到 4,225 个 correction states，完整覆盖 7 个
阶段。继续经过 15D Cartesian proprio、3D grounding 和高分辨率 grounding 后，direct action
仍为 0/12，因此没有把后续 hybrid success 冒充端到端成功；最终将 learned perception 与可审计
几何控制分层，得到 34/60 final-v2。

### 追问准备

- 为什么 grounding 正确仍会动作错？
- 为什么 phase accuracy 100% 仍会过早闭夹？
- 为什么 correction chunk 只监督第 1 步？
- DAgger 的 beta 怎样退火？

## 故事 2：推理慢 176 倍不是模型太大

### Situation

约 1.15M 参数模型评测 2 个 episode 超过 10 分钟。

### Task

把环境渲染、MuJoCo step、模型 forward 和 Python 开销拆开，找到部署瓶颈。

### Action

固定 batch size 1 和输入，消融 Torch threads 1/2/4/8，记录 forward latency。发现 1 thread
约 19.55 ms，8 threads 约 3.45 s。将 evaluator 和 trainer 的线程数做成显式配置并恢复调用前
状态。

### Result

当前机器上的 batch-1 forward 约加速 176 倍，闭环评测恢复到可用速度。结论严格限定为当前
Windows CPU、模型和系统负载。

### 追问准备

- 为什么小矩阵多线程可能更慢？
- intra-op threads 和进程并发有什么区别？
- 为什么报告 p50/p95 而不只报告平均值？

## 故事 3：任务定义漏洞造成假成功

### Situation

专家偶尔 1 step 完成抓取放置。

### Task

判断是 expert 太强、物理引擎异常，还是初始化与终止条件存在漏洞。

### Action

检查 reset 分布、方块和目标区位置以及 success 条件，确认方块可能初始化在目标区。修改采样排除
目标区，并要求任务先 lift 再 place，加入 terminal 回归测试。

### Result

消除 1-step 假成功，正式数据只接受完整成功轨迹。

### 追问准备

- 为什么先检查 reward/termination 再调算法？
- 怎样定义“抓取成功”而不是“位置偶然正确”？

## 故事 4：断电与可恢复训练

### Situation

CPU 训练时间长，断电或中断会留下 lock；简单重跑可能覆盖 metrics 或重复 epoch。

### Task

实现可验证、不会双写的恢复机制。

### Action

保存 model、optimizer、scheduler、Torch RNG 和 DataLoader generator；lock 记录 PID；恢复前检查
PID 是否仍存活，并验证 seed、模型配置、数据 fingerprint 以及 metrics 最后 epoch。

### Result

15-epoch deterministic run 经中断后可继续，且不会把两个 trainer 写进同一 run directory。

### 追问准备

- 只保存模型权重为什么不能严格续训？
- stale lock 与 active lock 怎样区分？
- 为什么 fine-tune initialization 与 resume 是两个不同参数？

## 故事 5：成功轨迹反而让定位指标变差

### Situation

第一版 hybrid policy 得到 3/12 成功，但汇总出的 target world error 约 40 mm，而且部分成功
episode 的误差比失败更大。

### Task

判断模型定位是否真的错误，还是从像素到世界坐标的评测定义不成立。

### Action

逐 step 对齐 phase、方块高度、预测像素、反投影平面和真值位置。发现抓取后方块已经随夹爪抬起，
evaluator 却仍把它的预测像素投到桌面平面；这个几何假设只在抓取前成立。将指标拆成 raw/calibrated
pixel L2、pre-grasp target XY、descend/close target XY 和全程 goal XY。

### Result

动作和成功率完全不变，但指标恢复了物理含义，也使后续 13 mm 抓取窗口分析可信。这个案例说明
评测代码与策略代码具有同等研究风险。

### 追问准备

- 为什么不能简单统计整条轨迹平均误差？
- 反投影为什么需要已知平面或深度？
- 修指标后怎样证明没有顺手改策略？

## 故事 6：离线更好的标定在闭环里更差

### Situation

加入 target color 和 goal side 后，language-conditioned affine calibration 的 validation target
误差和 13 mm 覆盖率都优于 global affine。

### Task

决定是否把它作为最终控制配置，而不是只看离线排序。

### Action

保证 calibration 参数只在 102 个 train episodes 上拟合，18 个 validation episodes 只做模型
选择；随后在相同 12 个开发 seeds 上做配对闭环。global 版本达到 10/12，而离线更好的 language
版本只有 8/12。保留两份结果并拒绝后者。

### Result

避免用代理指标替代任务目标。Stage 6 重新训练后，language-conditioned calibration 才通过新的
24-episode 配对开发评测并进入 final-v2。

### 追问准备

- 为什么离线欧氏误差下降不保证接触成功？
- 校正器为什么不能使用 final scene 标签？
- 固定左右 goal station 会给校正器带来什么捷径？

## 故事 7：开发集 10/12，最终只有 30/60

### Situation

Stage 5 在 12 条开发 episode 上达到 83.3%，看起来已经足够写进简历。

### Task

检验这个数字是否能代表六种任务和未见初始化，而不继续消费同一小开发集。

### Action

冻结 checkpoint、calibration、搜索半径、close timeout 和 400-step budget；预先指定
seeds 50000-50059，每个颜色-目标组合 10 条；完整跑完后才查看 summary，并报告 Wilson 区间和
任务分项。

### Result

final-v1 为 30/60，而不是 10/12。失败中 28/30 未形成双指接触，直接指向 grounding 瓶颈。
简历只采用 final 数字，10/12 永久标为 dev；后续 Stage 6 使用新的 dev 和 final seed 段。

### 追问准备

- 为什么 12 条样本的成功率方差很大？
- Wilson interval 比正态近似好在哪里？
- 测试集为什么“看过一次就消费了”？

## 故事 8：视觉更准，但实时性变差

### Situation

Final-v1 的 30 个失败中有 28 个没有接触，因此需要提高目标定位分辨率；同时系统目标频率是
50 Hz，CPU 每步只有 20 ms。

### Task

在不破坏已训练动作主干的前提下提高空间精度，并量化部署代价。

### Action

新增 16 x 16 CNN key 与 coarse language query 的 cross-attention；用零门控保证初始化输出与
Stage 5 完全相同。第一次全模型 smoke 导致 action MAE 退化，正式训练改为冻结 backbone，只训练
145,601 个新参数。用回归测试验证 epoch 0 action/coordinate 等价，再做配对开发和独立 final。

### Result

Final-v2 target error 35.02 -> 27.06 mm，接触 32 -> 39，成功 30 -> 34；但 p50/p95 从
9.21/20.27 ms 增至 18.17/48.27 ms。Wilson 区间重叠，因此不声称统计显著，并明确记录尾延迟
超过 50 Hz 周期。

### 追问准备

- 零初始化 gate 为什么能保持函数等价？
- 为什么冻结模块还要保持 eval mode？
- p50 合格但 p95 超期会怎样影响控制？

## 故事 9：跨 PowerShell、WSL 与 ROS2 的进程泄漏

### Situation

ROS2 arm probe 已经打印成功，但 launch 进程没有退出；另一次脚本在 source ROS 环境时因未定义
变量直接中止。

### Task

把一次能跑的命令变成失败也能清理、可重复验证的工程脚本。

### Action

定位到外层 PowerShell 提前展开 Bash `$!`，导致 PID 丢失；改为独立 Bash 脚本，在同一 shell
启动、记录 PID 并用 `trap` 清理。随后发现 `set -u` 早于 ROS setup，调整为先 source overlay、
再启用 nounset。移除手工覆盖 `PYTHONPATH`，由 ROS setup 管理 metadata 路径。

### Result

clean run 正常退出且无遗留 arm 进程；独立 probe 对 JointState、RGB-D、CameraInfo、任务 metadata
和 TF 执行 11/11 检查。后续改 depth 伪彩时，测试又发现 `visualization -> experts -> env ->
control -> experts` 循环导入，最终把纯图像函数下沉到无机器人依赖的 `image_utils`，重新构建和
probe 仍通过。这个结果证明的是仿真消息合同，不是真机部署。

### 追问准备

- `$!` 属于哪一层 shell，为什么会被提前展开？
- `trap EXIT INT TERM` 分别覆盖什么？
- ROS2 overlay 与普通 Python `PYTHONPATH` 有什么区别？
