# MuJoCo、运动学与机械臂控制

## 1. MuJoCo XML 里有什么

MuJoCo 的 MJCF 不是“3D 展示文件”，而是动力学模型。阅读顺序：

1. `worldbody`：世界、桌面、机器人 body 树、物体和相机；
2. `joint`：自由度、轴和范围；
3. `geom`：视觉或碰撞几何；
4. `site`：没有质量的标记点，适合末端和目标；
5. `actuator`：控制信号如何作用到关节；
6. `sensor`：可观测物理量；
7. `equality`：焊接等约束。

本项目使用 MuJoCo Menagerie 的 SO-ARM100 资产，并在任务场景中增加桌面、三个方块、两个目标区
和相机。机械臂的 mesh 负责外观，collision geom 和接触参数决定物理交互。

## 2. 正运动学

正运动学回答：

> 已知关节角 \(q\)，末端在哪里？

串联机械臂的变换链：

\[
{}^0T_E(q) = {}^0T_1(q_1){}^1T_2(q_2)\cdots{}^{n-1}T_E(q_n)
\]

齐次变换：

\[
T =
\begin{bmatrix}
R & p\\
0 & 1
\end{bmatrix}
\]

其中 \(R\) 是旋转矩阵，\(p\) 是位置。MuJoCo 在每次 `mj_forward` 或 `mj_step` 后计算 body、site
和 geom 的世界位姿，因此环境可以直接读取末端 site 的位置。

## 3. Jacobian

Jacobian 描述小关节变化如何引起末端变化：

\[
\Delta x \approx J(q)\Delta q
\]

若只控制位置，\(J_p\in\mathbb{R}^{3\times n}\)。MuJoCo 可以计算 site 的平移和旋转 Jacobian。
本项目使用平移 Jacobian，把期望末端位移转换成关节增量。

## 4. 为什么不用普通逆矩阵

机械臂的 Jacobian 通常不是方阵，也可能接近奇异。直接伪逆在小奇异值处会产生极大的关节速度。
阻尼最小二乘 DLS 解：

\[
\Delta q =
J^\top(JJ^\top+\lambda^2I)^{-1}\Delta x
\]

它等价于最小化：

\[
\|J\Delta q-\Delta x\|^2+\lambda^2\|\Delta q\|^2
\]

阻尼 \(\lambda\) 的含义：

- 太小：跟踪更精确，但奇异位形附近动作剧烈；
- 太大：更稳定，但末端响应迟钝。

实验入口在 `embodied_vla/control/ik.py`。你应分别把 damping 设为 `0.005`、`0.04`、`0.2`，
记录末端误差和关节动作。

## 5. 控制链

本项目一帧控制经过：

```text
normalized policy action
    -> physical Cartesian delta
    -> DLS-IK joint delta
    -> clipped joint target
    -> MuJoCo position actuator
    -> 10 physics substeps
    -> new observation
```

控制周期为：

\[
\Delta t_{\text{control}}
= \Delta t_{\text{physics}}\times N_{\text{substeps}}
= 0.002\times10
= 0.02\ \text{s}
\]

即 50 Hz。VLA 不需要每个 2 ms 直接推理，低层执行器在中间维持关节目标。

## 6. 接触和抓取

MuJoCo 接触由 collision geom、摩擦、solver 和时间步共同决定。稳定夹取要求：

- 两个手指形成合理接触；
- 法向力足够；
- 切向摩擦能抵抗重力和加速度；
- 控制命令不要穿透或抖动；
- 时间步和 solver 足够稳定。

本项目提供两种口径：

### `contact`

完全依赖接触和摩擦。它更接近物理，但低成本夹爪 mesh、接触参数和专家控制器会让成功率较低。

### `contact_assisted`

先要求左右手指都接触目标且夹爪关闭，然后激活 weld 约束帮助稳定抓取。它不是“无条件吸住”，
但仍比真实接触简单，适合先研究策略、语言和 action chunk。

实验报告必须分开写，不能把 assisted 成功率冒充严格接触成功率。

### 配对实验说明了什么

在完全相同的 seeds 10000-10099、任务轮换和 300-step budget 下，waypoint expert 得到：

| Mode | Success | Wilson 95% CI |
| --- | ---: | ---: |
| strict `contact` | 38/100 | 29.1%-47.8% |
| `contact_assisted` | 98/100 | 93.0%-99.4% |

60 个百分点的差值不是策略网络能力，而主要来自 grasp dynamics 的简化。strict contact 失败集中
在 descend、lift 和 transport，分别对应对准、摩擦稳定性和运输滑落。它同时说明两件事：

1. 仿真 strict grasp 并非完全不可行，38 条在纯接触下完成；
2. assisted 结果适合隔离研究视觉语言策略，但不能作为真机抓取性能代理。

配对 seed 很重要。如果两个模式使用不同初始化，就无法确定差异来自 grasp mode 还是场景难度。

## 7. 专家不是 teleport

专家控制器是 waypoint state machine：

```text
PREGRASP -> DESCEND -> CLOSE -> LIFT -> TRANSFER -> LOWER -> RELEASE
```

每一阶段根据真实末端、目标物和夹爪状态产生动作。物体只由接触或经过接触门控的辅助约束移动，
不直接修改物体位置。

状态机的价值：

- 产生结构化示范；
- 暴露 phase 辅助监督；
- 便于定位失败阶段；
- 为模型提供可解释 baseline。

它的局限：

- 路径和阈值由人工设计；
- 不会自然恢复所有异常；
- 对复杂障碍和物体朝向泛化有限。

## 8. 相机投影

世界点到像素要经过：

\[
p_c = {}^cT_w p_w
\]

\[
\tilde{u}=Kp_c,\qquad
u=\tilde{u}_x/\tilde{u}_z,\quad
v=\tilde{u}_y/\tilde{u}_z
\]

其中 \(K\) 是相机内参。MuJoCo 相机给出外参和视场角，环境据此计算内参，并把目标方块和目标区
中心投影成归一化像素坐标。这些坐标监督 Tiny-VLA 的 grounding head。

注意：

- 点在相机后方时无效；
- 点超出画面时无效；
- 中心点可见不代表整个物体无挡；
- MuJoCo、OpenCV 和图像数组的轴方向可能不同。

`pixel_valid` 正是为了不对无效投影强行计算损失。

## 9. Domain randomization

训练数据可以随机化光照、材质、摩擦和初始位置。目的不是“越随机越好”，而是让真实变化落在
训练分布覆盖范围内。

- 过弱：模型记住背景和精确坐标。
- 过强：任务规律被噪声淹没，训练变难。

正确实验应固定测试组：

- in-distribution；
- unseen random seeds；
- 更强视觉随机化；
- 更强物理随机化。

## 10. 你应该能回答

1. MuJoCo 的 `joint`、`geom`、`actuator` 和 `site` 分别是什么？
2. 为什么策略输出末端增量后还需要 IK？
3. DLS 比普通伪逆稳定在哪里？
4. 50 Hz 控制频率如何由配置得到？
5. `contact_assisted` 的门控条件是什么，为什么必须单独报告？
6. 目标世界坐标如何变成 grounding label？
