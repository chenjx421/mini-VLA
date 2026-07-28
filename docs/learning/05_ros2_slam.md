# ROS2 与 SLAM：从仿真传感器到地图

## 1. 为什么把 ROS2 和 SLAM 分开设计

桌面固定机械臂不需要定位整间房。如果强行把 SLAM 塞进 pick-place 控制链，只是在堆关键词。

本项目设计两个合理边界：

- **arm bridge**：机械臂关节、RGB、深度、CameraInfo、TF、任务和 VLA 动作；
- **mobile SLAM bridge**：移动传感平台、2D LiDAR、漂移里程计、TF 和 slam_toolbox。

这样既能学习机械臂系统接口，也能真正学习移动机器人 SLAM。

## 2. ROS2 基本概念

### Node

独立功能进程。这里包括：

- `arm_bridge`
- `mobile_slam_bridge`
- `async_slam_toolbox_node`
- `slam_recorder`

### Topic

异步发布/订阅数据流：

```text
/scan
/odom
/ground_truth_pose
/joint_states
/camera/color/image_raw
/camera/depth/image_raw
/vla/action
/map
```

### Message

topic 的强类型数据结构。例如：

- `sensor_msgs/LaserScan`
- `nav_msgs/Odometry`
- `sensor_msgs/Image`
- `sensor_msgs/CameraInfo`
- `geometry_msgs/Twist`

### Service

请求/响应接口，适合 reset 等一次性命令，不适合高频图像流。

### QoS

ROS2 基于 DDS，QoS 决定 reliability、history、durability 等。传感器通常使用
`qos_profile_sensor_data`，允许低延迟和偶发丢包；地图可能需要可靠和 transient local。

## 3. TF 坐标树

SLAM 常见 TF：

```text
map -> odom -> base_link -> laser
```

- `map`：全局一致坐标，loop closure 后可跳变；
- `odom`：局部连续但会漂移；
- `base_link`：机器人本体；
- `laser`：LiDAR 安装坐标。

- 里程计节点发布 `odom -> base_link`。
- 静态安装关系发布 `base_link -> laser`。
- slam_toolbox 根据 scan matching 发布 `map -> odom`。

不要同时让两个节点发布同一条 TF，否则树会冲突。

## 4. 2D LiDAR

`LaserScan` 的第 \(i\) 束角度：

\[
\theta_i=
\theta_{min}+i\Delta\theta
\]

本项目每帧发 180 束射线。MuJoCo `mj_ray` 从移动平台位置沿各世界方向查询最近碰撞距离，再加入
噪声并裁剪到 `[range_min, range_max]`。

发布消息时必须正确填写：

- `frame_id=laser`
- `angle_min/max/increment`
- `range_min/max`
- `scan_time`
- `ranges`

单位分别是弧度、米和秒。

## 5. Odometry 为什么漂移

里程计通过速度积分：

\[
x_{t+1}=x_t+v_t\cos\theta_t\Delta t
\]

\[
y_{t+1}=y_t+v_t\sin\theta_t\Delta t
\]

\[
\theta_{t+1}=\theta_t+\omega_t\Delta t
\]

每步小偏差都会积累，尤其角度误差会把后续平移投到错误方向。里程计的优点是局部连续、频率高；
缺点是长期漂移。

## 6. Occupancy grid

地图把平面离散成栅格。每格通常表示：

- `-1`：unknown；
- `0`：free；
- `100`：occupied。

贝叶斯更新更常在 log-odds 空间：

\[
l_t(m_i)=l_{t-1}(m_i)
+\operatorname{inverseSensorModel}(z_t,x_t)
-l_0
\]

LiDAR 射线经过的格子趋向 free，末端命中的格子趋向 occupied。

本项目 slam_toolbox 结果分辨率为 0.05 m/格。地图尺寸会随探索范围变化，不应硬编码。

## 7. Scan matching

scan matching 寻找当前 scan 与已有地图或上一帧最匹配的位姿修正：

\[
\hat x_t=
\arg\max_x
\operatorname{score}(z_t,\mathcal M,x)
\]

里程计提供初始猜测，LiDAR 几何提供校正。环境若长走廊、重复结构或特征太少，匹配会退化。

## 8. Pose graph 和 loop closure

pose graph：

- 节点：历史关键帧位姿；
- 边：里程计约束、scan matching 约束和 loop closure 约束。

回到旧地点时，loop closure 增加跨时间约束。图优化重新分配累计误差，使地图闭合。

这也是 `map` 坐标可能跳、`odom` 坐标应连续的原因。控制器通常依赖局部 odom，导航目标依赖 map。

## 9. 项目中的真实验证

在 WSL2 Ubuntu 24.04 + ROS2 Jazzy 中：

1. `mobile_slam_bridge` 发布 `/scan`、`/odom`、`/ground_truth_pose` 和 TF；
2. `slam_toolbox` 订阅 scan 和 TF；
3. 自动巡航绕仓库循环；
4. `/map` 生成非空 occupancy grid；
5. recorder 同时保存真值、里程计和 SLAM pose；
6. 使用 header timestamp 最近邻对齐后计算末端定位误差。

一次 60 秒采集中：

- 地图 123 x 118；
- 分辨率 0.05 m；
- 里程计末端位置误差约 1.59 m；
- SLAM 校正后约 0.84 m。

噪声有意设置得较强，目的是让短时间实验能明显观察漂移与校正。它不是现实硬件精度声明。

## 10. 为什么误差必须按时间戳对齐

`/ground_truth_pose`、`/odom` 和 `/pose` 发布频率不同。直接比较三个数组最后一项，可能是在比较
不同时间的机器人位置。

recorder 保存 ROS header timestamp，并对估计的最终时间在真值轨迹中找最近邻：

\[
i^*=\arg\min_i |t_i^{gt}-t_{final}^{est}|
\]

然后计算：

\[
e=\|p_{i^*}^{gt}-p_{final}^{est}\|_2
\]

更严格的研究还应插值真值、报告整条轨迹 ATE/RPE，而不仅是末端误差。

## 11. Arm bridge 应该会解释什么

arm bridge 发布：

- `JointState`：关节名称、位置和速度；
- RGB `Image`；
- 32-bit float depth `Image`；
- `CameraInfo`：内参矩阵；
- camera/robot TF；
- 任务 metadata。

订阅 `/vla/action` 后，把 5 维动作送入同一个 MuJoCo 环境。这条边界以后可以替换为：

- 本地 PyTorch policy node；
- 独立推理服务；
- 真机 SO101 driver。

消息接口稳定，仿真或真机后端可以替换，是 ROS2 系统设计的关键价值。

## 12. 必答题

1. node、topic、message、service 和 QoS 有什么区别？
2. `map -> odom -> base_link -> laser` 每条 TF 由谁发布？
3. 为什么 odom 连续但长期漂移？
4. occupancy grid 的 `-1/0/100` 是什么？
5. scan matching 与 loop closure 有何不同？
6. 为什么固定机械臂不需要 SLAM？
7. 为什么异步轨迹不能直接比较最后一个数组元素？

## 13. 怎样证明 arm bridge 不是“代码写了但没跑”

只执行 `colcon build` 证明语法、依赖和入口点可构建，不能证明 runtime topic 正确。只执行
`ros2 topic list` 又只能证明名字存在，不能证明消息内容可用。本项目的 `arm_probe` 分四层检查：

1. **连接层**：在 timeout 内收到 JointState、RGB、depth、CameraInfo、task metadata 和 TF；
2. **schema 层**：6 个 joint name 对应 6 个 position，RGB 为 `rgb8`，depth 为 `32FC1`；
3. **数值层**：关节和深度为有限值，深度存在正值，CameraInfo 焦距为正；
4. **语义层**：自然语言任务与结构化 color/side 合法，TF 包含相机坐标系。

一次 clean run 结果：

- 11/11 checks passed；
- RGB/depth 均为 128 x 128；
- 观察到 12 对 robot/camera/object TF；
- 任务为 `grasp the red block and put it into the right bin`；
- 脚本退出后没有遗留本次 launch 进程。

复现：

```bash
cd /path/to/EmbodiedVLA-RL
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
cd ..
bash scripts/run_ros_arm_probe_wsl.sh outputs/ros_arm_probe
```

## 14. 跨 Windows、WSL 和 ROS2 的三个坑

### 不要覆盖 ROS2 的整条 PYTHONPATH

`source /opt/ros/jazzy/setup.bash` 和 workspace `install/setup.bash` 会建立 overlay。手工把
`PYTHONPATH` 设成单个项目路径，可能让 `ros2cli` 找不到 entry-point metadata。若需要让 WSL
导入项目，优先在 WSL Python 中执行 editable install，或在原变量前后追加，不要覆盖。

### `$!` 必须在启动后台进程的 Bash 中读取

`$!` 是 Bash 的“最近后台进程 PID”。若把含 `$!` 的命令放进 PowerShell 双引号长字符串，
外层 shell 可能先处理变量，Bash 最后得到空值。独立 `.sh` 文件能明确展开边界：

```bash
ros2 launch ... &
bridge_pid=$!
trap cleanup EXIT INT TERM
```

### `set -u` 应在 source ROS setup 之后启用

`set -u` 会把读取未定义变量当作错误。ROS setup 脚本可能先读取可选环境变量再设置默认值，所以：

```bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
set -u
```

这不是关闭严格模式，而是把严格模式放在你自己的脚本主体上。

## 15. 新增必答题

1. `colcon build`、topic 存在、收到消息和消息语义正确分别证明什么？
2. 为什么 RGB 与 depth 除了 encoding 还要检查 shape 和有限值？
3. CameraInfo 的焦距为什么必须为正？
4. TF 中有 frame name 为什么还不等于整棵树连通？
5. ROS2 overlay 与普通 Python 包搜索路径有什么关系？
6. `SIGINT`、`SIGTERM` 和 `trap EXIT` 在 launch 清理中各有什么作用？
7. 仿真 arm bridge 通过为什么仍不能写“已部署 SO101 真机”？
