# 简历项目条目

## 推荐名称

**EmbodiedVLA-RL：基于 MuJoCo、Tiny-VLA 与 PPO 的具身操作及 ROS2 SLAM 系统**

技术栈：Python / PyTorch / MuJoCo / Gymnasium / ROS2 Jazzy / slam_toolbox

## 中文简历版

```text
EmbodiedVLA-RL：具身操作、强化学习与 ROS2 SLAM                  个人项目
Python / PyTorch / MuJoCo / Gymnasium / ROS2 Jazzy / slam_toolbox

- 基于官方 SO-ARM100 关节模型搭建语言条件抓取放置环境，实现 5D Cartesian 增量动作、
  阻尼最小二乘 IK、50 Hz 控制、RGB-D/15D 本体观测及 domain randomization；在相同
  100 个 seeds 上对比 strict contact 38% 与 contact-assisted 98%，明确物理简化边界。
- 从零实现 1.39M 参数 Tiny-VLA，以图像 patch、语言 token 和 proprioception 预测 8-step
  action chunk；设计冻结主干的 16x16 language grounding 分支，将 validation target 世界
  坐标误差从 31.56 mm 降至 27.27 mm，并保留 direct policy 0/12 的失败诊断。
- 构建不读取目标真值的 VLA-grounded hybrid policy，结合 train-only calibration、相机
  反投影、visual servo 与接触恢复；在六任务均衡的 60 条未见轨迹上完成 34/60
  （Wilson 95% CI 44.1%-68.4%），30 条随机化场景完成 18/30，CPU p50 18.17 ms。
- 从头实现 tanh Gaussian Actor-Critic、GAE、PPO policy/value clipping、KL early stop 和
  8-env rollout，3 seeds x 100,352 steps 的 reach 成功率为 58.3% +/- 33.3%；开发 ROS2
  arm/mobile bridge，11/11 RGB-D/JointState/TF 检查通过，并用 slam_toolbox 生成 5 cm 地图。
```

简历空间紧张时，删掉第三条中的 domain-randomized 结果和第四条中的具体 ROS topic，不要删
hybrid 边界或 PPO 标准差。

## 30 秒介绍

> 这个项目不是套一个大模型 API，而是让我把具身闭环逐层做清楚。我在 MuJoCo 中搭了
> SO-ARM100 抓取放置环境，从头实现 PPO 和一个 1.39M 参数 Tiny-VLA。直接 action policy
> 离线指标不错但闭环失败，我通过同状态 expert trace 定位到厘米级 grounding 误差，后来加入
> 冻结主干的高分辨率语言定位和可审计几何控制，在 60 条未见轨迹上完成 34 次。系统同时通过
> ROS2 发布机械臂 RGB-D、关节和 TF，并在独立移动平台上用 slam_toolbox 完成建图。所有成功、
> 失败和限制都有 seed、JSONL、checkpoint 和图像证据。

## 证据对照

| 简历表述 | 证据 | 面试时主动说的限制 |
| --- | --- | --- |
| strict 38%、assisted 98% | `results/expert_benchmark/` | assisted 是双指接触后激活的稳定约束 |
| 1.39M Tiny-VLA | `results/vla_stage6/training_summary.json` | 从零训练的小模型，不是 foundation model |
| target 31.56 -> 27.27 mm | Stage 5/6 calibration JSON 与训练曲线 | 仍大于 13 mm 抓取窗口 |
| hybrid 34/60 | `results/vla_final_v2/` | direct Stage 5 是 0/12 dev |
| DR 18/30 | `results/vla_robustness_dr/` | 只有 30 条，CI 与 clean final 重叠 |
| p50 18.17 ms | final-v2 summary | p95 48.27 ms，超过 20 ms 周期 |
| PPO 58.3% +/- 33.3% | `results/ppo_reach/aggregate_summary.json` | state reach，不是 RGB pick-place；seed 方差大 |
| ROS arm 11/11 | `results/ros_arm/summary.json` | MuJoCo bridge，不是真机 SO101 |
| SLAM 5 cm map | `results/ros_slam/summary.json` | 合成 LiDAR；误差是 endpoint，不是完整 ATE |

## 不能改写成

- “端到端 VLA 成功率 56.7%”；
- “VLA 显著提升 6.7 个百分点”；
- “PPO 稳定达到 80%”；
- “完成 SO101 真机部署”；
- “实现导航抓取一体化”；
- “SLAM 定位精度提升 47%”。

这些说法要么混淆 policy 边界，要么忽略统计和实验作用域。正确边界不会让项目变弱，反而说明
你知道机器人研究中什么能由证据支持。

## 投递前个人验收

1. 不看代码画出 direct 和 hybrid 两条数据流；
2. 手写 DLS-IK、GAE、PPO clip 和 ray-plane intersection；
3. 从一条失败 trace 找第一处分叉；
4. 解释为什么 language calibration 曾离线更好、闭环更差；
5. 解释为什么 PPO 要报 seed std；
6. 运行一次 ROS arm probe；
7. 亲手完成一个测试驱动的小改动并提交；
8. 对上表每个数字指出 JSON 路径。

没有完成第 7 项前，这仍主要是一个由工具搭建、你正在接管的项目；完成并能解释后，才逐步变成
你的工程经验。
