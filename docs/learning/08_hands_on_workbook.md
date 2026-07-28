# 四天动手工作簿

这份工作簿不是阅读清单。每一项都要求先写预测、再运行、最后用自己的话解释。答案不要写进本
文件；新建 `notes/my_learning_log.md`，保留你第一次答错的内容和修正过程。面试官问“你遇到什么
问题”时，你自己的错误比背项目历史更有说服力。

## 使用规则

每个实验都写六行：

```text
问题：
运行前预测：
我改动的变量：
实际结果：
为什么一致/不一致：
下一步怎样排除另一个解释：
```

通过标准不是“命令成功”，而是能关掉仓库后讲清输入、变换、输出、证据和限制。

## Day 1：MuJoCo、动作与控制闭环

### 1. 环境 smoke

```powershell
python -m pip install -e ".[dev]"
evla-check-env --episodes 2 --image-size 192 --save-frame outputs\day1_scene.png
```

你要回答：

1. `reset()` 返回的 observation 与 `info` 分别放什么？
2. 为什么 policy 可以读 proprio，但不能读 target world coordinate？
3. 5D action 每一维是什么，范围为什么归一化到 \([-1,1]\)？
4. 50 Hz 对应一个 MuJoCo control step 多长？

代码入口：

- `embodied_vla/envs/so_arm_pick_place.py`
- `embodied_vla/envs/config.py`
- `embodied_vla/proprioception.py`

### 2. 手推 DLS-IK

先独立写出：

\[
\Delta q=J^\top(JJ^\top+\lambda^2I)^{-1}\Delta x
\]

再读 `embodied_vla/control/ik.py`，逐个变量标注 shape。回答：

- \(J\) 为什么不是方阵？
- \(\lambda=0\) 在奇异位形附近会怎样？
- action 是世界系增量还是关节角？
- wrist 和 jaw 为什么不经过同一个 3D Jacobian 求解？

亲手实验：把 damping 乘 10，先预测轨迹会更稳还是更快，再运行 expert GIF 对照。

### 3. 接触与任务终止

读 `embodied_vla/experts/pick_place.py` 和环境 success 条件，画出：

```text
approach -> descend -> close -> lift -> transport -> release -> done
```

必须区分：

- 单指接触与双指接触；
- close command 与 jaw 已闭合；
- grasped 与 lifted；
- `contact` 与 `contact_assisted`。

当天口试：为什么“方块初始就在目标区”会制造假成功，测试应该怎样防止回归？

## Day 2：从数据到 Tiny-VLA

### 1. 数据审计

```powershell
evla-audit-dataset datasets\so_arm_pick_place_v2_120_dr
```

找到并记下：

- episode 数、frame 数、六任务计数；
- train/validation episode ID；
- fingerprint；
- 五个 action 维度的标准差。

解释为什么随机抽 frame 会泄漏，为什么 wrist 恒零会美化总体 MAE。

代码入口：

- `embodied_vla/data/trajectory.py`
- `embodied_vla/data/audit.py`
- `embodied_vla/training/vla_trainer.py`

### 2. 手画模型张量

读 `embodied_vla/models/tiny_vla.py`，不要先看 README 架构图。自己填表：

| Tensor | 你写 shape | 来源 | 进入哪个模块 |
| --- | --- | --- | --- |
| RGB | | | |
| coarse vision tokens | | | |
| language tokens | | | |
| proprio token | | | |
| encoder memory | | | |
| action queries | | | |
| action chunk | | | |
| coarse heatmap | | | |
| high-res heatmap | | | |
| world grounding | | | |

然后回答：

1. patch size 8 为什么产生 64 个 coarse visual tokens？
2. action horizon 8 与 execution horizon 有什么区别？
3. language 是怎样影响“选择红块”而不是只作为装饰 token？
4. high-resolution branch 为什么仍需要 coarse language query？
5. 零 gate 如何保证旧 checkpoint 加载后函数不变？

### 3. 跑最终 checkpoint

先用 6 条 episode 快速复现，不把它当新 final：

```powershell
evla-eval-vla-hybrid `
  --checkpoint checkpoints\tiny_vla_stage6_highres.pt `
  --grounding-calibration checkpoints\tiny_vla_stage6_calibration.json `
  --output-dir outputs\my_first_closed_loop `
  --episodes 6 --seed 60000 --video-episodes 2 `
  --max-episode-steps 400 `
  --recovery-search-radius-m 0.018 `
  --close-retry-steps 18
```

从 `summary.json` 找：

- success；
- contact/grasp/lift funnel；
- pre-grasp 与 descend/close 误差；
- p50/p95 latency；
- `policy_boundary`。

任选一个成功和失败 trace，找第一处不同。不要只说“最后没放进去”。

## Day 3：PPO、ROS2 与 SLAM

### 1. PPO 白板

不看答案写：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

\[
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}
\]

\[
L^{clip}=\mathbb E[
\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]
\]

再读：

- `embodied_vla/algorithms/ppo.py`
- `embodied_vla/training/ppo_trainer.py`
- `embodied_vla/models/state_policy.py`

逐项检查 tanh log-prob correction、timeout bootstrap、GAE reset boundary、value clipping 和
KL early stop。读取 `results/ppo_reach/aggregate_summary.json` 和三份 metrics，解释为什么
15/20、4/20、16/20 应汇总为 58.3% mean、33.3% sample std。明确这是 state reach，不是
RGB pick-place。

### 2. ROS2 arm probe

在 WSL2 ROS2 Jazzy：

```bash
cd /mnt/d/GIT/EmbodiedVLA-RL/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
cd ../
bash scripts/run_ros_arm_probe_wsl.sh outputs/my_ros_arm_probe
```

打开 summary，解释每一个 check。再画 arm TF，至少包括：

```text
world -> Base -> ... -> Fixed_Jaw
world -> camera_color_optical_frame
```

回答为什么 `topic list` 不能替代消息数值检查。

### 3. SLAM 数据流

对照 `docs/assets/slam_trajectory_comparison.png` 画：

```text
MuJoCo LiDAR -> /scan -----------+
                                     -> slam_toolbox -> /map, map->odom
noisy odometry -> /odom, odom->base_link
static TF -> base_link->laser
```

必须能解释 occupancy grid、scan matching、loop closure，以及为什么固定桌面机械臂本身不需要
SLAM。1.59 m 与 0.84 m 是强合成漂移下的 endpoint error，不是完整 ATE，也不是真机精度。

## Day 4：让项目开始属于你

### 1. 选一个亲手改动

投递前建议选最小但可讲清的任务：

- 给 arm probe 增加 RGB 像素方差检查，防止全黑图也通过；
- 给 hybrid summary 增加 `post_grasp_success_rate`；
- 给 report plot 标出 post-grasp failure 数；
- 新增一个 language counterfactual 单元测试。

流程：

1. 先写失败测试；
2. 预测修改会影响哪些文件；
3. 实现；
4. 跑 targeted test；
5. 跑 full test；
6. 在个人日志按 STAR 记录；
7. 用自己的 Git author 提交。

### 2. 三个 3 分钟故事

不用背句子，只按六格讲：

```text
现象 -> 假设 -> 隔离实验 -> 证据 -> 决策 -> 限制
```

第一轮固定讲：

1. offline MAE 好但 closed-loop 失败；
2. 错误的 post-grasp 平面指标；
3. dev 10/12 到 final 30/60。

第二轮由对方随机追问：

- 为什么不继续扩大模型？
- calibration 是否泄漏？
- 为什么 high-res p95 超过 20 ms？
- 为什么 DAgger correction chunk 只监督第一步？
- ROS2 probe 怎样保证进程清理？

### 3. 投递前核对

- 简历每个数字能指到 `results/` 文件；
- 不把 34/60 写成 direct action success；
- 不写 PPO 多 seed；
- 不写真机部署；
- 能解释一个成功 GIF 和一个失败 GIF；
- 能在白板写 DLS、GAE、PPO clip、ray-plane intersection；
- 能说出自己亲手改过的代码和测试。

## 自测评分

每题 0-2 分：0 不会，1 看提示会，2 关掉资料也会。

| 模块 | 题数 | 通过线 |
| --- | ---: | ---: |
| MuJoCo/控制 | 10 | 16/20 |
| 数据/VLA | 15 | 24/30 |
| PPO | 8 | 12/16 |
| ROS2/SLAM | 10 | 16/20 |
| 实验与诚信边界 | 10 | 18/20 |

总分 86 分以上再把项目放到简历靠前位置；未达到时仍可投递，但把它放在“研究中项目”，每天继续
补齐。真正的拥有感来自能预测、修改和解释，而不是仓库在谁的账号下。
