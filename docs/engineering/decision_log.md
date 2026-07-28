# 工程决策索引

`project_journal.md` 保留完整时间线，本页把关键问题压缩成可检索的决策表。面试准备时先选一行，
再回到原始 JSON、trace、图和代码复述，不要只背本表。

| ID | 现象 | 如何排除猜测 | 决策 | 结果与边界 |
| --- | --- | --- | --- | --- |
| D01 | expert 偶尔 1 step 成功 | 查 reset 分布、success 条件和 lift 历史 | 初始化排除目标区，success 必须先 lift | 消除假成功；加 terminal 回归测试 |
| D02 | DataLoader 训练越来越慢 | profile NPZ 解压和 cache hit | 正式小数据集常驻 episode cache | 降低重复解压；16 GB 机器适用 |
| D03 | 1.15M 模型闭环推理极慢 | 固定输入消融 Torch 1/2/4/8 threads | batch-1 evaluator 固定 1 thread | 当前机器约 176x 加速；只限该硬件/负载 |
| D04 | offline action MAE 0.064，但 closed loop 0/2 | 同状态 privileged expert 只做诊断；按 initial/phase 切片 | 不再用全局 MAE 代表能力 | initial MAE 0.33566，定位关键状态缺口 |
| D05 | learner 一偏离就没有专家标签 | 在 learner-visited state 查询 expert | 两轮 DAgger，beta 0.5 -> 0.2 | 4,225 + 5,239 correction states；direct 仍未成功 |
| D06 | Cartesian action 方向相关性弱 | 12D/15D epoch-0 配对输出等价 | 加合法 end-effector XYZ proprio | initial dy correlation 0.025 -> 0.409；闭环仍 0/12 |
| D07 | 能靠近但下降时横向未对准 | 记录同状态 model/expert action 和米制误差 | 加 3D grounding/phase action residual | world L2 3-4 cm，仍大于 6-13 mm 抓取窗口 |
| D08 | direct policy 反复失败 | 检查 grounding 是否有语义、几何和控制价值 | 保留 direct 失败，建立 affordance-style hybrid | learned pixel + geometry + servo 首次 3/12 |
| D09 | 成功轨迹 target error 反而更大 | 对齐方块高度、phase 和反投影平面 | target 平面误差只统计抓取前 | 动作不变，指标恢复物理含义 |
| D10 | grounding 有稳定像素偏差 | train 拟合、validation 选模型、dev 闭环配对 | train-only affine calibration | global 3/12 -> 7/12；无 final label leakage |
| D11 | language calibration 离线更好 | 同一 12 dev seeds 闭环 | Stage 5 拒绝离线最优 language 版本 | 10/12 -> 8/12，证明代理指标不等于任务目标 |
| D12 | 接触失败后总回同一落点 | trace 搜索 offset 和接触状态 | 18 mm 固定局部搜索 | search-only 9/12；不读取物体真值 |
| D13 | close 等待从 18 降到 8 steps | 只改 timeout 做配对消融 | 保留 18 steps | 10/12 -> 6/12；command 不等于接触已建立 |
| D14 | dev 10/12 看似很好 | 冻结配置后跑 60 条六任务均衡新 seeds | 简历只写 final | final-v1 30/60；28/30 失败无接触 |
| D15 | final-v1 主要卡 target grounding | coarse 语义与 fine geometry 分支解耦 | 16x16 high-res branch，零 gate，冻结主干 | target error 35.02 -> 27.06 mm；contact 32 -> 39 |
| D16 | 第一次 high-res smoke 伤害 action | 比较 epoch-0 与更新后 action MAE | 只训练 145,601 个新参数，冻结模块保持 eval | action 保持；成功 30/60 -> 34/60 |
| D17 | success 上升但 p95 变慢 | 同时记录 success funnel 和 latency distribution | 保留精度/实时性双指标 | p50/p95 18.17/48.27 ms；不隐藏 50 Hz 超期 |
| D18 | final-v1/v2 数字不同 | Wilson interval、任务分项和 seed protocol | 不声明统计显著 | 两个 95% CI 重叠 |
| D19 | domain-randomized 18/30 高于 clean 34/60 | 比较 CI 并检查 host load | 只说未观察到明显崩溃 | 样本小；后台并发 latency 不可公平比较 |
| D20 | calibration 写了 SHA 却可能混用权重 | 沿 evaluator 读取路径审计 | 加载后强制校验 checkpoint hash | 匹配通过，篡改权重测试被拒绝 |
| D21 | ROS probe 成功但 launch 未退出 | 查 PowerShell/Bash 展开和精确 PID | 独立 Bash、`$!`、`trap`、分阶段 strict mode | 11/11 arm checks，clean exit |
| D22 | assisted 抓取口径可能过于乐观 | 同 100 seeds 只切换 grasp mode | 两种结果并列报告 | assisted 98/100，strict 38/100 |
| D23 | PPO seed 1 达到 15/20 | 固定 100,352 steps 补 seed 2/3 | 用 seed mean/std，不挑最好 | 15/20、4/20、16/20；58.3% mean，33.3% sample std |
| D24 | ROS depth 数值正确但预览近乎全白 | 检查动态范围并测试独立导入 | 分位数伪彩；轻量函数移到 `image_utils` | 解决循环导入；重建和 11/11 probe 通过 |
| D25 | Windows 全量 pytest 在 WSL symlink 收集阶段失败 | 错误发生在 test collection 前 | `testpaths = ["tests"]` | 50 tests 通过；构建产物不再参与收集 |
| D26 | 旧 PPO JSON 缺少后来新增的 false 字段 | 比较 dataclass 默认值和真实配置 | 只规范化明确的 legacy default | false 可聚合，true 仍由测试拒绝 |

## 面试复述模板

每个故事最多讲六步：

1. **现象**：可观察且带数字；
2. **假设**：至少两个可能原因；
3. **隔离**：这次实验只改变什么；
4. **证据**：JSON、trace、图或测试在哪里；
5. **决策**：为什么选当前方案、拒绝了什么；
6. **限制**：结论在哪些条件下不成立。

一个合格回答会主动提被拒绝的方案。比如 calibration 不能只讲“标定后更准”，还要讲
language-conditioned 版本离线更好却使 Stage 5 闭环从 10/12 降到 8/12，因此当时被拒绝。

## 原始证据入口

- 完整时间线：`docs/engineering/project_journal.md`
- 正式实验索引：`results/experiment_registry.md`
- 简历 claim 与边界：`results/claim_ledger.md`
- 逐实验记录模板：`docs/engineering/experiment_template.md`
- 已整理 STAR 故事：`docs/engineering/interview_stories.md`
- 最终结果 JSON/JSONL：`results/`
- 可视化：`docs/assets/`
