# 示范数据、行为克隆与 Tiny-VLA

## 1. 一条示范包含什么

每个 episode 保存为一个压缩 NPZ，时间维长度为 \(T\)：

| 字段 | 典型形状 | 含义 |
| --- | --- | --- |
| `rgb` | `[T, 64, 64, 3]` | 前视相机 |
| `proprio` | `[T, 12]` | 关节位置和速度 |
| `state` | `[T, 37]` | 仅调试/奖励使用的 privileged state |
| `language` | `[16]` | 整条 episode 的指令 token |
| `action` | `[T, 5]` | 归一化连续动作 |
| `phase` | `[T]` | 专家状态机阶段 |
| `target_pixel` | `[T, 2]` | 目标方块归一化像素 |
| `goal_pixel` | `[T, 2]` | 目标区归一化像素 |
| `pixel_valid` | `[T, 2]` | 投影标签是否有效 |
| `terminated/truncated` | `[T]` | 终止语义 |

语言是 episode-level，动作和图像是 step-level。读取样本时把语言复制到相应时间步，不代表每帧
都存一份重复文本。

## 2. 为什么任务必须平衡

任务是 3 种颜色乘 2 个目标侧，共 6 类。如果大多数数据都是 `red->left`，模型可以通过先验
猜测任务，忽略语言。

正式数据集每类 20 个成功 episode，共：

- 120 episodes；
- 18,972 steps；
- 只有 1 次被拒绝的超时轨迹；
- episode 长度中位数 147.5。

`audit_dataset.py` 还检查：

- 数组形状和有限值；
- 动作是否越界；
- 最后一帧是否真的终止；
- 是否出现异常短的“成功”；
- 六种任务覆盖；
- SHA256 数据指纹；
- RGB、proprio、action 和 state 统计量。

数据指纹的价值是：checkpoint 可以证明自己训练在具体哪份数据上，而不是同名但内容不同的目录。

## 3. 为什么按 episode 划分

相邻帧高度相似。若随机按帧划分，训练集可能含 `episode_010` 的第 50 帧，验证集含同一轨迹的
第 51 帧，验证指标会虚高。

本项目先按任务分组，再在每组内划分完整 episode：

```text
red-left:   train episodes | validation episodes
red-right:  train episodes | validation episodes
...
```

这保证：

- train/validation 没有轨迹泄漏；
- 验证集仍覆盖全部语言任务；
- 各任务比例一致。

## 4. Action chunk

单步策略输出：

\[
\hat a_t = \pi(o_t,l)
\]

Action chunk 策略一次预测未来 \(H\) 步：

\[
\hat A_t =
[\hat a_t,\hat a_{t+1},\ldots,\hat a_{t+H-1}]
\]

本项目默认 \(H=8\)，输出形状 `[B, 8, 5]`。episode 尾部不足 8 步时，用最后动作填充，但
`action_mask` 只让真实步参与损失，避免 padding 污染训练。

### Prediction horizon 与 execution horizon

- prediction horizon：模型一次预测多少步；
- execution horizon：部署时真正执行其中多少步后重新观察。

例如预测 8 步，只执行 2 步，再拍新图重规划。执行得少更稳健但推理更频繁，执行得多更快但更
容易开环漂移。

## 5. Tiny-VLA token 化

输入图像 `64 x 64`，patch size 为 8，因此得到 \(8\times8=64\) 个视觉 token。

```text
[TASK]                 1 token
[PROPRIO]              1 token
language              16 tokens
image patches         64 tokens
--------------------------------
encoder input         82 tokens, each 128 dimensions
```

每种 token 都加：

- 内容 embedding；
- 位置 embedding；
- modality embedding。

语言 padding 通过 attention mask 屏蔽。Transformer encoder 让视觉、语言和本体状态相互
注意，输出共享 multimodal memory。

## 6. 为什么有 task token

`[TASK]` 类似一个可学习的汇总槽位。经过 self-attention 后，它聚合当前任务特征，用于：

- phase classification；
- 给 action query 提供任务条件；
- 给 grounding query 提供任务条件。

这不是必须的唯一结构，但它让“全局任务摘要”和“空间视觉 token”有清晰分工。

## 7. Action decoder

模型有 8 个可学习 action query，每个对应 chunk 中一个未来位置。Transformer decoder 让
这些 query cross-attend multimodal memory，最后映射成 5 维动作并经过 `tanh` 限制到
`[-1, 1]`。

确定性动作损失采用 Smooth L1：

\[
\mathcal{L}_{action}
= \frac{1}{\sum m}
\sum_{t=0}^{H-1}
m_t\operatorname{SmoothL1}(\hat a_t,a_t)
\]

Smooth L1 在小误差处像平方损失，在大误差处像绝对值损失，比纯 MSE 更不容易被异常动作支配。

## 8. Phase 辅助头

phase head 从 task token 预测 7 个真实受监督阶段。数据中不同阶段频次差异较大，因此使用
平方根逆频率权重：

\[
w_c \propto \sqrt{\frac{N}{N_c}}
\]

辅助任务的作用不是部署时接管控制，而是迫使共享表示编码“当前处于接近、下降、闭合还是搬运”。

注意：辅助准确率高不保证动作成功，它只能提供可诊断信号。

## 9. Grounding 辅助头

两个 query 分别表示 target 和 goal，它们只 cross-attend 64 个视觉 patch。归一化 attention
得到 \(8\times8\) 热力图 \(p_i\)。

预测坐标是 patch 中心的期望：

\[
\hat c = \sum_i p_i c_i
\]

训练同时使用：

- 坐标 Smooth L1；
- 正确 patch 的负对数似然。

热力图能回答“模型把目标方块和目标区看在哪里”。但 attention 只是可审计信号，不等于因果
解释。真正验证语言使用仍需反事实任务测试。

## 10. 总损失

\[
\mathcal{L}
= \lambda_a\mathcal{L}_{action}
+\lambda_p\mathcal{L}_{phase}
+\lambda_c\mathcal{L}_{coord}
+\lambda_h\mathcal{L}_{heatmap}
\]

默认权重：

```text
action = 1.0
phase = 0.25
coordinate = 0.5
heatmap = 0.25
```

权重决定各梯度相对规模，不应只看数字大小拍脑袋。正确做法是观察各 loss、梯度和最终闭环结果，
再做消融。

## 11. Flow Matching 动作头

确定性回归会把多种合理动作平均。Flow Matching 从噪声逐步生成动作，更适合多峰分布。

令真实 action chunk 为 \(x_0\)，高斯噪声为 \(x_1\)。线性路径：

\[
x_t=(1-t)x_0+tx_1,\qquad t\sim U(0,1)
\]

真实速度恒为：

\[
v^*(x_t,t)=x_1-x_0
\]

模型根据 noisy action、time embedding 和 multimodal memory 预测速度，最小化：

\[
\mathcal{L}_{flow}=\|v_\theta(x_t,t,o,l)-v^*\|^2
\]

推理从 \(t=1\) 的噪声开始，用 Euler 积分反向走到 \(t=0\)：

\[
x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t,o,l)
\]

本项目默认 8 步采样。它比确定性头推理慢约 8 倍，但能生成不同动作模式。是否值得必须由同数据、
同 seed 的闭环实验回答。

## 12. 离线指标为什么会骗人

平均动作 MAE 可能被以下因素美化：

- 大量保持或慢速动作；
- 某一维在数据中恒为零；
- 相邻帧泄漏；
- 预测平均动作；
- 尾部 padding 未 mask。

当前数据中 wrist 动作恒为零，因为立方体不要求调整朝向。因此总 MAE 会被这一维拉低。仓库同时
报告每个动作维度 MAE，并把“增加有朝向物体”列为扩展，不隐藏限制。

最终必须报告：

- validation action MAE，分维度；
- phase accuracy；
- grounding L2；
- unseen-seed closed-loop success；
- execution horizon；
- 推理延迟；
- 典型失败轨迹。

## 13. 从预训练到后训练的概念映射

这个 CPU 项目从头训练小模型，不是假装训练大 foundation model。但完整 VLA 常见流程可以映射：

| 大模型流程 | 本项目对应 |
| --- | --- |
| 视觉/语言 backbone 预训练 | 当前从零初始化，后续可换预训练 encoder |
| 多机器人数据预训练 | 平衡 MuJoCo expert dataset |
| 任务 SFT / behavior cloning | action chunk + auxiliary losses |
| 后训练 | 可加入 DAgger、失败恢复数据、residual RL |
| 部署 | receding-horizon closed loop / ROS policy boundary |

你应明确说“实现并研究了 VLA 的核心结构和训练闭环”，不要把 1M 参数模型说成通用基础模型。

## 14. 必答题

1. 为什么 64 x 64 图像得到 64 个 patch token？
2. action horizon 和 execution horizon 有什么区别？
3. 为什么训练/验证按 episode 划分？
4. 语言反事实评测如何构造？
5. grounding heatmap 能证明模型真的因果使用视觉吗？
6. Flow Matching 的 \(x_t\) 和速度目标是什么？
7. 离线 MAE 很低但闭环失败可能有哪些原因？

## 15. Behavior Cloning 的分布偏移

行为克隆训练的数据来自专家分布：

\[
(o_t,a_t)\sim d_{\pi_E},\qquad
\min_\theta \mathbb{E}_{d_{\pi_E}}
\left[\ell(\pi_\theta(o_t),a_t)\right]
\]

部署时状态却由学习策略自己产生：

\[
o_t\sim d_{\pi_\theta}
\]

即使单步犯一个小错，下一帧也可能落到专家数据没覆盖的位置；新的错误继续累积，这就是
covariate shift。序列长度为 \(T\) 时，朴素 BC 的最坏情况累计代价可能从单步误差的一次量级
放大到 \(O(T^2\epsilon)\)。这里最重要的不是背复杂度，而是理解：训练时“看老师走过的路”，
部署时“走自己造成的路”，二者不是同一个状态分布。

本项目的具体表现是：

- validation 全局首动作 MAE 约 0.057；
- episode 初始首动作 MAE 约 0.336；
- 模型第一步偏离后，phase 继续推进；
- 没有接触却预测 lift、transport 和 release。

因此 phase accuracy 高、grounding 看起来正确，都不能替代 on-policy 闭环检查。

## 16. DAgger 怎样修复

DAgger 的核心循环：

1. 用当前 learner 在环境中访问状态；
2. 在这些状态上查询 expert label；
3. 将新样本聚合到训练集；
4. 重新训练 learner；
5. 逐轮降低 expert 执行概率 \(\beta_i\)。

混合策略可以写为：

\[
\pi_i =
\begin{cases}
\pi_E, & \text{probability } \beta_i\\
\pi_\theta, & \text{probability } 1-\beta_i
\end{cases}
\]

注意“谁执行动作”和“谁提供监督”是两件事：无论该步由谁执行，都查询 expert action 作为标签。

本项目第 1 轮使用 \(\beta=0.5\)，让专家帮助轨迹进入抓取后阶段，同时保留足够 learner
偏离状态。24 个 episode 得到 4,225 个 correction states，并覆盖 7 个阶段。

### 为什么 correction 只监督 chunk 第一步

在 learner 状态 \(o_t\) 上，我们只问到了专家当前动作 \(a_t^E\)。如果下一步实际执行 learner
动作，那么数据中的 \(a_{t+1}^E\) 是“另一个状态上的专家动作”，不是“从 \(o_t\) 连续执行专家
后得到的第二步”。因此把相邻 8 个 correction label 当成严格 counterfactual action chunk
并不成立。

仓库把 correction chunk 的 mask 设为：

```text
[True, False, False, False, False, False, False, False]
```

这比填充未来动作更保守，也更容易在面试中解释。

### 为什么 action loss 要先按样本归一化

原 demo 每个样本最多有 8 个有效动作，DAgger correction 只有 1 个。如果直接对所有有效时间步
求总平均，一条 correction 的权重天然约为 demo 的八分之一。修复后的损失先在每个样本内平均，
再对 batch 平均：

\[
\mathcal{L}_{action}
=\frac{1}{B}\sum_{b=1}^{B}
\frac{\sum_t m_{bt}\ell_{bt}}
{\max(1,\sum_t m_{bt})}
\]

## 17. 关键状态采样

全局逐帧均匀采样并不等于“任务上公平”。一条 150-step 轨迹只有 1 个初始帧，但第一步方向错误
就可能让后面 149 步全部失去意义。

本项目的 weighted sampler 分别配置：

- `initial_state_weight`；
- `early_state_weight` 和 `early_window_steps`；
- `correction_sample_weight`；
- `samples_per_epoch`。

采样权重只改变训练分布，不改变 validation 分布。正式比较必须保持相同 validation episode，
并同时看：

- global MAE；
- initial MAE；
- early-window MAE；
- 模型动作标准差；
- 模型与专家动作相关系数；
- closed-loop success。

如果 MAE 略降但预测标准差仍接近 0，模型可能仍在输出条件均值。

## 18. 从 grounding 到 action 的空间瓶颈

原模型的 grounding 是辅助输出，action decoder 理论上可以从视觉 memory 中自己提取位置，但
没有结构保证它会使用已经学到的坐标。新增的可选路径为：

```text
target/goal attention
        |
        v
coarse soft-argmax coordinates
        |
        +--> sub-patch coordinate refiner
        |
concat with proprio
        |
grounding-action projection
        |
add to action queries
```

动作头使用的是模型预测坐标，不是真值坐标，所以部署时没有 privileged leakage。

### 为什么需要亚 patch 精修

8 x 8 heatmap 的一个 patch 宽度是 \(1/8=0.125\)。初始 target-y 的真实标准差只有约 0.028，
小于一个 patch。旧 soft-argmax 把 y 变化压成几乎常数。

精修头根据 role-specific grounded visual feature 和 coarse coordinate 预测残差：

\[
\hat c_{\text{refined}}
=\operatorname{clip}
\left(
\hat c_{\text{coarse}}
+\frac{1}{G}\tanh f_\theta(z,\hat c_{\text{coarse}}),
0,1
\right)
\]

其中 \(G=8\)，所以残差最多一个 patch。最后一层零初始化，确保刚加载旧 checkpoint 时
\(\hat c_{\text{refined}}=\hat c_{\text{coarse}}\)，不会先随机破坏已有策略。

1 epoch 消融中，validation grounding L2 从约 0.034 降到 0.021，target-y 预测标准差从
约 0.00006 增到 0.017。这个结果只证明坐标精度改善；是否提升任务成功仍要由闭环评测决定。

## 19. 新增必答题

1. expert distribution 与 learner distribution 为什么不同？
2. DAgger 中 beta 表示什么，为什么逐轮下降？
3. 为什么 correction action chunk 只监督第一步？
4. 为什么全局 MAE 低但预测标准差接近零仍是坏模型？
5. weighted sampler 会怎样影响 train metric，为什么 validation 不能加同样权重？
6. grounding-conditioned action 是否使用了真值坐标？
7. 为什么坐标精修残差要限制在一个 patch？
8. 零初始化 residual 最后一层有什么作用？

## 20. 为什么 Cartesian action 需要 Cartesian proprio

本项目的动作前三维是末端执行器在世界坐标中的增量。控制器随后用阻尼最小二乘 IK 把它转成
关节目标：

\[
\Delta q = J^\top(JJ^\top+\lambda^2I)^{-1}\Delta x
\]

原 12D proprio 只有：

\[
p_{12}=[q_1,\ldots,q_6,\dot q_1,\ldots,\dot q_6]
\]

理论上神经网络可以学习正运动学 \(x_{ee}=f_{\text{FK}}(q)\)，但这会消耗数据和模型容量。
而真实机器人本来就可以根据编码器、URDF 和 TF2 求出末端位姿。因此 15D 版本显式加入：

\[
p_{15}=[p_{12},x_{ee}/0.5,y_{ee}/0.5,z_{ee}/0.5]
\]

这里的 `0.5` 是归一化尺度，不是目标坐标。必须区分：

- **合法本体感觉**：关节编码器、关节速度、由正运动学算出的末端位姿；
- **privileged label**：方块世界坐标、目标区世界坐标、仿真器内部接触真值；
- **视觉预测**：模型从 RGB 预测的 target/goal pixel。

15D 模型仍不知道方块的世界坐标。它只知道自己的手在哪里，方块和目标区仍需通过图像与语言
grounding 获得。这种设计在机器人学习中叫 state/action representation alignment：输入状态应
提供产生所选动作空间所需的机器人自身状态。

### 为什么升级 checkpoint 要移动四列权重

grounding-action MLP 的旧输入顺序是：

```text
[12D proprio, target_x, target_y, goal_x, goal_y]
```

升级后的顺序是：

```text
[12D proprio, ee_x, ee_y, ee_z, target_x, target_y, goal_x, goal_y]
```

如果只在权重矩阵末尾补三个零，旧 coordinate 权重会错误地作用在 ee 坐标上。正确做法是：

1. 复制旧 proprio 12 列；
2. 新 ee 三列置零；
3. 把旧 coordinate 四列移动到最后四列。

项目同时用单元测试和真实 validation epoch 0 检查升级前后输出一致。这个细节很适合作为面试中
“如何保证模型迁移没有静默错误”的例子。

## 21. 本阶段必答题

1. 末端位姿为什么不是 privileged information？
2. 正运动学和逆运动学分别解决什么问题？
3. 为什么 action space 与 state representation 应对齐？
4. 12D 到 15D 扩展时，为什么不能简单在矩阵末尾补零？
5. epoch 0 配对验证能排除哪些混杂因素，不能排除哪些因素？

## 22. 从 2D grounding 到 3D action

像素 grounding 回答“目标在图像哪里”，Cartesian action 回答“末端在世界坐标中往哪里走”。
二者中间还隔着相机投影：

\[
s
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
=K[R\mid t]
\begin{bmatrix}
X\\Y\\Z\\1
\end{bmatrix}
\]

固定相机和平面物体可以用标定或单应矩阵显式求解；端到端网络也可以学习这个映射，但小数据时
不一定精确。Stage 4 的失败正是：模型能靠近目标上方，也能预测 descend，却仍会在横向未对准
时下降和闭夹。

Stage 5 增加 3D auxiliary grounding：

\[
\hat p_{\text{target}},\hat p_{\text{goal}}
=g_\theta(z_{\text{target}},z_{\text{goal}},\hat u,\hat v)
\]

训练时用 MuJoCo 世界坐标监督，推理时只保留网络预测。动作分支接收：

\[
[p_{\text{proprio}},\hat p_{\text{target}},\hat p_{\text{goal}}]
\]

这和“推理时把真值坐标塞给模型”有本质区别。辅助标签可以在仿真中低成本生成，在真机中可由
标定板、AprilTag、motion capture 或人工标注获得；部署时必须移除这些标注源。

同时将 phase softmax 显式投影到 action query。phase head 不再只是一个用于展示的辅助分类器，
动作 loss 也能沿这条路径约束阶段表示。为了不破坏旧策略，两个新增 action residual 的最后一层
仍然零初始化。

### 必须警惕

- 3D head 的存在不代表 3D 预测准确，必须报告米制误差；
- world-grounding loss 下降不代表闭环成功；
- predicted phase 仍可能抖动，单帧策略没有显式历史；
- 辅助真值只能出现在训练标签和诊断中，不能进入部署 action path。

## 23. 为什么最终采用 hybrid VLA

“VLA 输出动作”并不要求所有控制都必须塞进一个网络。机器人系统常见三种边界：

1. **direct policy**：图像、语言和 proprio 直接输出关节或 Cartesian action；
2. **affordance policy**：网络输出目标点、抓取姿态或轨迹条件，经典控制器执行；
3. **hierarchical policy**：高层网络选技能或子目标，低层策略执行。

本项目同时保留第 1 和第 2 条研究路径。Stage 5 direct policy 的输入输出确实是
vision-language-action，但 12 条开发 episode 为 0 成功。最终 selected pipeline 属于第 2 条：

```text
RGB + language
      |
      v
Tiny-VLA target/goal grounding
      |
      v
train-only pixel calibration
      |
      v
camera ray-plane geometry
      |
      v
Cartesian visual servo + contact state machine
      |
      v
DLS-IK + MuJoCo actuator
```

它仍然研究 VLA 的语言指代和视觉定位，但成功率不能写成“端到端 action decoder 34/60”。正确
说法是“VLA-grounded hybrid policy 34/60；direct action policy 0/12 dev”。这种边界意识本身
就是机器人系统能力。

## 24. 像素怎样变成世界坐标

对于像素 \((u,v)\)，先用相机内参 \(K\) 变成相机坐标系射线：

\[
r_c=K^{-1}[u,v,1]^\top
\]

再用相机外参旋转到世界系：

\[
r_w=R_{wc}r_c,\qquad o_w=t_{wc}
\]

若目标位于已知桌面平面 \(z=z_0\)，射线与平面交点满足：

\[
p_w=o_w+\tau r_w,\qquad
\tau=\frac{z_0-o_{w,z}}{r_{w,z}}
\]

这里最容易犯四类错误：

- 把 world-to-camera 与 camera-to-world 旋转方向用反；
- 忘记图像坐标 \(v\) 向下，而世界坐标轴定义不同；
- 像素归一化时混用 \([0,1]\)、\([-1,1]\) 和实际分辨率；
- 物体已被抬起后仍假设它位于桌面平面。

最后一项正是本项目评测指标的真实 bug。平面反投影适合桌面上未抓取物体和固定目标区；抓起后
若要估计完整 3D，必须使用深度、双目、多视角或另一个高度模型。

## 25. Train-only calibration 在学什么

网络预测像素可能有稳定偏差，例如整体向左 3 像素。相机几何是精确的，但输入射线偏了，世界
坐标仍会错。因此拟合：

\[
\hat Y=XW
\]

其中每行 \(X=[1,\hat u,\hat v]\)，\(Y=[u_{true},v_{true}]\)。带 ridge 的闭式解为：

\[
W=(X^\top X+\lambda I)^{-1}X^\top Y
\]

`language_conditioned_affine` 还加入颜色和目标侧 one-hot 特征。必须遵守：

- \(W\) 只用 train split 真值拟合；
- validation 只选 identity/global/language 三种候选；
- final scene 标签既不能拟合，也不能选候选；
- JSON 保存 checkpoint SHA、dataset fingerprint、矩阵和 split。

为什么这不算推理时使用 privileged state？因为拟合完成后部署只读取网络像素、语言和固定矩阵，
不读取当前场景真值。它与相机标定类似。但如果任务只有两个固定 goal station，语言特征可以直接
记住站点，所以接近零的 goal error 不能推广成“一般目标定位能力”。

## 26. 高分辨率 grounding 的结构

原 Transformer 在 8 x 8 patch grid 上做语义选择，擅长回答“红色方块是哪一个”，但一个 patch
覆盖 8 x 8 原像素，精细抓取坐标不足。Stage 6 把语义与几何分工：

1. coarse grounding 得到与语言角色相关的 feature \(q\)；
2. 轻量 CNN 在原图上产生 16 x 16 spatial keys \(k_{ij}\)；
3. 计算
   \[
   a_{ij}=\operatorname{softmax}_{ij}(q^\top k_{ij}/\sqrt d)
   \]
4. 用 soft-argmax 得到高分辨率坐标；
5. 用可学习 gate \(g\) 混合旧、新坐标。

\[
c=(1-\sigma_g)c_{coarse}+\sigma_g c_{highres}
\]

实现中 gate 初始化使新分支贡献为零，因此从 Stage 5 加载时输出逐元素相同。第一次 smoke 让全
模型一起训练，action MAE 立即退化；正式训练冻结 backbone，只训练 145,601 个 high-resolution
参数。冻结模块还要保持 `eval()`，否则 dropout 即使不更新权重也会改变输出分布。

最终不是只看 heatmap loss。证据链是：

- validation target world error 下降；
- 相同开发 seeds 的 paired success 提升；
- 全新 final seeds 的 contact funnel 提升；
- 同时报告 CPU p50/p95 变慢。

## 27. 接触恢复为什么不是“偷偷用真值”

控制器只知道：

- 视觉估计的 target/goal；
- 合法 proprio 中的末端位置；
- 双指是否接触；
- 当前状态机 phase 和计时器。

它不知道真实方块世界坐标。第一次下降失败后，若仍回到同一估计点，确定性偏差会让每次都失败。
因此按固定 18 mm 网格尝试相邻 offset：

```text
center -> -x -> +x -> -y -> +y -> diagonals
```

这是 contact feedback 下的局部主动搜索。它类似人在看不准时用触觉小范围试探，不是 teleport。
搜索半径必须在开发集选择，不能根据 final 失败逐条修改。

close timeout 的消融也很关键。把等待从 18 steps 降到 8 steps 后，成功从 10/12 降到 6/12。
原因是“输出 close action”“夹爪关节移动”“形成双指接触”“辅助抓取约束激活”发生在不同物理
时刻。控制算法必须尊重 actuator dynamics，而不是把 command 当作立即完成的状态。

## 28. 一次完整训练与评测应该怎样做

不要从最终成功率倒推流程。推荐顺序：

1. 审计数据 fingerprint、任务平衡和 episode split；
2. 训练 baseline，记录 global/initial/phase-wise 指标；
3. 做 direct closed-loop，保存第一处分叉；
4. 根据失败机制选择 DAgger、状态表示或空间分支；
5. 在 train/validation 上选择 checkpoint 和 calibration；
6. 用小但独立的 development seeds 选控制超参数；
7. 冻结所有配置；
8. 一次跑完 task-balanced final seeds；
9. 报 success、Wilson interval、contact/grasp/lift funnel、空间误差、延迟和失败轨迹。

Stage 6 精确训练配置已经写入 `results/vla_stage6/training_summary.json` 的
`runtime.command`。关键参数是：

```bash
evla-train-vla \
  --dataset datasets/so_arm_pick_place_v2_120_dr \
  --correction-dataset datasets/dagger_v1_seed30000_beta050 \
  --correction-dataset datasets/dagger_v2_seed35000_beta020 \
  --initialize-checkpoint checkpoints/tiny_vla_stage5_best_initial.pt \
  --include-end-effector-position \
  --grounding-action-conditioning \
  --grounding-coordinate-refinement \
  --high-resolution-grounding \
  --freeze-backbone-for-high-resolution-grounding \
  --world-grounding \
  --world-grounding-action-conditioning \
  --phase-action-conditioning \
  --epochs 8 --samples-per-epoch 12000 --seed 24 \
  --output-dir outputs/my_stage6
```

训练后先拟合 calibration：

```bash
evla-fit-grounding-calibration \
  --checkpoint outputs/my_stage6/checkpoints/last.pt \
  --dataset datasets/so_arm_pick_place_v2_120_dr \
  --output-dir outputs/my_stage6_calibration
```

再运行新的评测 seed，不能继续用 60000-60059 假装“未见测试”。复现实验可用原 seed；做新结论
应另选未消费 seed 段。

## 29. 最终必答题

1. direct VLA、affordance policy 和 hierarchical policy 的边界是什么？
2. 怎样从像素、内参、外参和桌面高度得到世界坐标？
3. calibration 为什么只能用 train label 拟合？
4. 语言条件 goal calibration 为什么可能产生任务捷径？
5. coarse semantic query 与 high-resolution key 各自解决什么问题？
6. 零门控和冻结 backbone 怎样保证增量训练可比？
7. 为什么 contact search 不属于 privileged leakage？
8. 为什么 command time、joint motion time 和 contact time 不相同？
9. 为什么必须同时报告 success funnel 和 latency distribution？
10. 为什么 34/60 不能写成 direct action decoder 的成功率？
