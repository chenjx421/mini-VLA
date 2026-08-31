# MuJoCo 两天实操入口

这部分课程只针对 MuJoCo 使用，使用本项目的 SO-ARM100 抓取放置场景作为练习环境。
它不重复讲 Tiny-VLA、PPO 或 ROS2。

## 环境

项目要求 Python 3.10 或更高版本。安装完整依赖：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

如果只想运行 MuJoCo 示例，也可以安装最小依赖：

```bash
python -m pip install mujoco gymnasium numpy pillow imageio pyyaml
```

## 第一课：加载、推进和渲染

```bash
python mujoco_course/lesson00_smoke.py
```

这个脚本会：

1. 从 `embodied_vla/assets/so_arm100/pick_place.xml` 加载 MJCF；
2. 创建 `MjModel`（固定模型）和 `MjData`（实时状态）；
3. 读取 `grip_site` 的世界坐标；
4. 调用 `mj_step` 推进 100 个物理步；
5. 用 `Renderer` 保存 `outputs/mujoco_lesson00.png`。

正常输出应包含：

```text
nq=27, nv=24, nu=6
physics timestep=0.0020 s
simulated time=0.200 s
```

## 两天学习顺序

### Day 1：模型和状态

- 阅读 `embodied_vla/assets/so_arm100/pick_place.xml`；
- 找到 `worldbody`、关节、碰撞 geom、`grip_site`、执行器和相机；
- 在 Python 中检查 `model.nq`、`model.nv`、`model.nu`；
- 对比 `mj_forward`、`mj_step`、`qpos`、`qvel`、`ctrl`；
- 修改关节控制并观察末端位姿和渲染结果；
- 用 `data.contact` 检查手指与方块的接触。

### Day 2：控制和调试

- 阅读 `embodied_vla/control/ik.py` 的 DLS-IK；
- 理解笛卡尔位移到关节增量的转换；
- 阅读 `embodied_vla/envs/so_arm_pick_place.py` 的 `_apply_action`；
- 记录 physics timestep、substeps 和控制频率；
- 比较不同 damping 对末端误差和关节动作的影响；
- 用 `embodied_vla/experts/pick_place.py` 跟踪 approach、descend、close、lift、transport、release 阶段。

## 重要坐标和频率

本项目的策略动作先表示末端笛卡尔增量，再经过 DLS-IK 变成关节目标，最后交给 position actuator。
MuJoCo 每次物理步长为 `0.002 s`；环境每个控制步执行 10 个 physics substeps，所以控制周期为
`0.002 × 10 = 0.02 s`，即 50 Hz。

## 常见渲染错误

如果看到 `Image width ... > framebuffer width 256`，说明请求的离屏图像超过 XML 的默认 framebuffer。
降低 `Renderer` 的宽高，或在 XML 中增加：

```xml
<visual>
  <global offwidth="640" offheight="480" />
</visual>
```
