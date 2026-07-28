# EmbodiedVLA-RL 学习路线

这不是“把仓库跑起来就算学会”的路线。完成每个阶段后，你都要能够：

1. 不看答案画出数据流；
2. 指出代码入口并解释关键张量形状；
3. 修改一个变量，先预测结果，再运行实验；
4. 用失败案例说明方法的边界；
5. 在白板上推导核心公式。

建议每天 4 小时，按“概念 45 分钟 + 阅读代码 60 分钟 + 实验 90 分钟 +
复盘 45 分钟”执行。

## 7 月 28-31 日投递前冲刺

距离 8 月开始投递只有 4 天，不能假装四天内掌握全部具身智能。目标是先达到“简历每句话都有
证据、项目主链能讲、三个问题能深入追问”的最低可投递线；投递后继续完成下面 14 天主线。

| 日期 | 4 小时安排 | 当天必须产出 |
| --- | --- | --- |
| 7/28 | 运行 smoke；看 SO-ARM100 场景；画 observation -> VLA -> servo -> IK -> physics | 一张手画系统图；能解释 direct 与 hybrid |
| 7/29 | 读 Tiny-VLA forward；逐张量写 shape；用最终 checkpoint 跑 6 条闭环 | 自己写的 tensor 表；一条成功和一条失败复盘 |
| 7/30 | 手推 DLS、GAE、PPO clip；运行 ROS arm probe；阅读 SLAM 图 | 三个公式手写稿；ROS topic/TF 说明 |
| 7/31 | 亲手改一个小功能并测试；按 STAR 讲三次排障；做两轮模拟面试 | 一次自己的 commit；10 分钟无稿录屏；最终简历条目 |

每天最后 45 分钟必须关掉文档回答：

1. 今天的模块为什么存在？
2. 输入输出张量或消息是什么？
3. 最容易发生什么静默错误？
4. 仓库里哪条证据能支持简历表述？
5. 如果结果变差，下一步如何隔离变量？

投递前优先掌握三个故事：

- 离线 MAE 好但 direct closed loop 失败，怎样找到第一处分叉；
- 成功轨迹反而让旧空间指标变差，怎样修正评测定义；
- dev 10/12 但 final 30/60，怎样冻结配置并诚实报告。

## 14 天主线

| 天 | 主题 | 必做实验 | 通过标准 |
| --- | --- | --- | --- |
| 1 | 具身闭环、MDP/POMDP | 运行环境检查，改变随机种子 | 能区分 observation、state、action、reward |
| 2 | MuJoCo 模型与接触 | 可视化 SO-ARM100，改变摩擦系数 | 能解释 actuator、joint、geom、contact |
| 3 | 坐标系与相机 | 投影方块中心到像素 | 能写出世界坐标到像素坐标的变换链 |
| 4 | 正运动学、Jacobian、DLS-IK | 改变 IK damping | 能推导 DLS 更新式并解释奇异位形 |
| 5 | 专家状态机与示范数据 | 收集 12 条平衡示范并审计 | 能解释为什么按 episode 划分 |
| 6 | 行为克隆与 covariate shift | 训练一个小模型，比较离线与闭环 | 能解释低 MAE 仍可能失败 |
| 7 | Transformer 与多模态融合 | 手画 Tiny-VLA token 序列 | 能逐项说清每个 token 的来源 |
| 8 | Action chunk 与闭环部署 | 比较 execution horizon 1/2/4 | 能解释预测长度和执行长度的区别 |
| 9 | Flow Matching | 与确定性动作头同数据对照 | 能写出插值路径和速度目标 |
| 10 | RL、PPO、GAE | 复现 reach PPO 曲线 | 能推导 clipped objective 和 GAE |
| 11 | 奖励、终止与评测 | 故意制造 reward hacking | 能区分 terminated 和 truncated |
| 12 | ROS2 通信 | 检查 topic、TF、CameraInfo | 能解释 node/topic/message/TF/QoS |
| 13 | LiDAR SLAM | 生成地图和轨迹误差图 | 能解释 occupancy grid、scan matching、loop closure |
| 14 | 综合答辩 | 从零复现实验并做 10 分钟讲解 | 不看 README 回答面试问题 |

## 每天的固定输出

在自己的学习日志里写四段：

```text
今天的闭环：
我改了什么：
结果是否符合预测：
一个仍未解决的问题：
```

不要抄结论。先写预测，再运行命令。实验结果和预测不一致时，学习才真正开始。

## 三次里程碑答辩

### 里程碑 A：仿真与控制

- 从 MuJoCo XML 找到关节、执行器、相机和碰撞体。
- 解释 5 维动作如何经过缩放、DLS-IK 和位置执行器变成机械臂运动。
- 说明 `contact` 与 `contact_assisted` 的不同实验口径。
- 解释为什么“把物体直接传送到目标”不是有效控制。

### 里程碑 B：算法

- 用 POMDP 语言描述 RGB 抓取任务。
- 解释行为克隆、PPO、确定性 action chunk 和 flow matching 的训练目标。
- 说明训练/验证为什么必须按 episode 隔离。
- 同时报离线误差、闭环成功率、推理延迟和失败类型。

### 里程碑 C：系统

- 画出 MuJoCo、VLA policy、ROS2 bridge、TF 和 slam_toolbox 的边界。
- 展示地图、真值轨迹、漂移里程计和 SLAM 校正轨迹。
- 说明固定机械臂为什么不需要 SLAM，以及本项目为什么给 SLAM 单独设计移动平台。
- 对简历中的每个数字指出生成它的脚本和结果文件。

## 必须亲手完成的改动

项目由工具协助搭建，但下面这些改动必须由你亲手完成，才能真正变成你的：

1. 新增一种颜色和对应语言 token，重新收集平衡数据。
2. 修改相机位姿，重新检查目标像素投影。
3. 增加一个有朝向的长方体，让 wrist 动作不再恒为零。
4. 给 PPO 奖励加一个错误项，观察 reward hacking，再修掉。
5. 修改 Tiny-VLA 的 action horizon，并解释参数量为何几乎不变。
6. 给 ROS2 bridge 新增一个诊断 topic。
7. 关闭 loop closure 或增大 odometry noise，对比 SLAM 误差。

完成这些后，你讲的就不再是“别人写的项目”，而是你亲自验证过的系统。
