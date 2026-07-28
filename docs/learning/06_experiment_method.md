# 实验方法：如何让项目结果经得住追问

## 1. 先写 hypothesis，再运行

一个有效实验不是“改参数看看”。格式应是：

```text
问题：
假设：
唯一自变量：
固定变量：
指标：
失败判据：
结果：
解释：
```

例子：

```text
问题：一次执行更多 chunk action 是否更快但更不稳？
假设：execution_horizon=4 的推理次数更少，但 unseen-seed 成功率低于 1。
唯一自变量：execution_horizon in {1, 2, 4}
固定变量：checkpoint、测试 seed、任务顺序、grasp mode
指标：success rate、episode steps、policy latency
失败判据：任一设置少于 50 个 episode，或使用不同测试任务
```

## 2. Reproducibility contract

每个正式结果至少绑定：

- Git commit；
- 数据集 SHA256 fingerprint；
- 模型和训练配置；
- 随机 seed；
- Python、PyTorch、MuJoCo 版本；
- checkpoint；
- 原始逐 episode JSONL；
- 聚合 summary JSON；
- 生成图表的脚本。

仓库中的 run directory guard 防止两个训练器误写同一个目录。断电后必须显式指定
`--resume-checkpoint`，并验证 stale PID、metrics 最后 epoch、模型配置、seed 和数据指纹。

## 3. 随机种子不是什么

seed 控制初始化、数据 shuffle、环境随机化和采样，但它不能消除：

- 多线程算子的非确定性；
- 操作系统调度；
- 不同硬件和库版本；
- 浮点累积顺序；
- 断点恢复时未保存的 RNG 状态。

因此“固定 seed”表示提高可复现性，不表示任何机器逐 bit 相同。正式实验用多个 seed 估计算法
方差，而不是挑最好的一次。

## 4. 训练、验证和测试

- **train**：更新参数；
- **validation**：选 epoch、调超参数；
- **test**：最终一次报告，不能反复用来调参。

本项目按 episode 划分 train/validation。闭环 test 使用未参与训练的环境 seed。若你根据 test
失败反复改模型，这组 test 已经变成 validation，必须再准备新的 final test seeds。

## 5. 成功率的不确定性

成功率是 Bernoulli 均值。20 个 episode 的 75% 只是 15/20，区间会很宽。近似标准误：

\[
\operatorname{SE}(\hat p)
=\sqrt{\frac{\hat p(1-\hat p)}{n}}
\]

当 \(n=20,\hat p=0.75\) 时，SE 约 0.097。不能把 75% 与 70% 的小差异说成显著提升。

建议：

- 快速开发：20 episodes；
- 简历结果：每个 seed 至少 50 episodes；
- 论文比较：多个训练 seed，并报告 bootstrap 或 Wilson interval。

## 6. 多 seed 聚合

训练 seed 是独立实验单位。推荐先对每个 seed 求测试成功率，再在 seed 之间报告：

\[
\bar x=\frac{1}{K}\sum_{k=1}^{K}x_k
\]

\[
s=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(x_k-\bar x)^2}
\]

不要把 3 个 seed 的所有 episode 混成一个大样本后假装没有训练方差。

## 7. Ablation

消融要回答“哪一部分带来效果”，而不是堆一张大表。推荐顺序：

1. state oracle 与 multimodal policy；
2. RGB + language 与 RGB + language + proprio；
3. action horizon 1/4/8；
4. execution horizon 1/2/4；
5. deterministic 与 flow matching；
6. 无/有 phase auxiliary loss；
7. 无/有 grounding loss；
8. 无/有 domain randomization。

每次只改变目标因素，复用相同数据 split 和 test seeds。

## 8. Counterfactual language test

固定同一 RGB、proprio 和场景状态，只替换语言：

```text
pick red -> left
pick blue -> left
pick red -> right
```

检查：

- target grounding 是否随颜色改变；
- goal grounding 是否随左右改变；
- 第一段 action chunk 是否改变；
- 闭环最终是否操作指定物体。

如果预测几乎不变，模型可能依赖视觉位置先验而忽略语言。

## 9. Failure taxonomy

不要只保存成功 GIF。失败至少分为：

| 类别 | 第一处可观察错误 |
| --- | --- |
| language error | 朝错误颜色或错误目标运动 |
| grounding error | 热力图不在目标/目标区 |
| approach error | 末端方向正确但位置偏差 |
| grasp error | 双指未形成接触或提前闭合 |
| lift error | 抓住后滑落 |
| transfer error | 搬运路径漂移或碰撞 |
| release error | 到达目标但未正确放开 |
| horizon drift | chunk 后半段偏离，新观测来得太晚 |

找“第一处分叉”，不要只描述最终物体没进目标区。

## 10. Offline 与 closed-loop 对照

四种典型情况：

| 离线 MAE | 闭环成功 | 解释 |
| --- | --- | --- |
| 高 | 低 | 基本没拟合 |
| 低 | 低 | covariate shift、平均动作、关键帧错误 |
| 高 | 高 | 误差指标被无关维度或时序对齐影响 |
| 低 | 高 | 理想，但仍需 unseen distribution 测试 |

动作误差应该分维报告。当前立方体数据的 wrist 维恒为零，总 MAE 会被这一维美化。

## 11. 推理性能

至少测：

- 单次 forward 中位数和 p95；
- deterministic 与 flow 采样步数；
- CPU 线程数；
- 输入分辨率；
- batch size 1；
- end-to-end 控制周期。

若 policy latency 高于 20 ms，就无法在当前 50 Hz 控制循环每步重规划。可以降低重规划频率、
执行更多 chunk action、量化模型或拆分低层控制。

## 12. Claim ledger

准备 `results/claim_ledger.md`，每条简历表述写：

```text
claim:
evidence:
scope:
known limitation:
reproduction command:
```

例如“PPO 成功率 75%”必须同时写明：state observation、reach task、single seed、20-episode
independent eval。完成三 seed 后才能升级表述。

## 13. 实验命令

### 数据审计

```bash
evla-audit-dataset \
  datasets/so_arm_pick_place_v2_120_dr
```

### PPO

```bash
evla-train-ppo \
  --task reach \
  --total-steps 100000 \
  --seed 1 \
  --output-dir outputs/ppo_reach_seed1
```

### Tiny-VLA

```bash
evla-train-vla \
  --dataset datasets/so_arm_pick_place_v2_120_dr \
  --output-dir outputs/tiny_vla_det_seed1 \
  --epochs 15 \
  --action-horizon 8 \
  --action-head deterministic \
  --seed 1
```

### 断点恢复

```bash
evla-train-vla \
  --dataset datasets/so_arm_pick_place_v2_120_dr \
  --output-dir outputs/tiny_vla_det_seed1 \
  --epochs 15 \
  --action-horizon 8 \
  --action-head deterministic \
  --seed 1 \
  --resume-checkpoint outputs/tiny_vla_det_seed1/checkpoints/best.pt
```

### 闭环评测

```bash
evla-eval-vla \
  --checkpoint outputs/tiny_vla_det_seed1/checkpoints/best.pt \
  --output-dir outputs/tiny_vla_det_seed1_eval \
  --episodes 50 \
  --execution-horizon 1
```

先运行 `--help` 核对当前 CLI，不要把文档命令当成无需理解的咒语。
