# GitHub Mini-VLA Design Audit

Audit date: 2026-07-27

Implementation status updated: 2026-07-28

This document records design ideas, not copied source code. The implementation in
this repository is written for the SO-ARM100 MuJoCo task and is constrained to run
on a CPU-only 16 GB laptop.

## Repositories reviewed

| Repository | Useful idea | Decision in this project |
| --- | --- | --- |
| [keivalya/mini-vla](https://github.com/keivalya/mini-vla) | A small, readable end-to-end pipeline; diffusion action generation; saved rollout videos | Keep the full path from expert data to closed-loop GIF easy to trace. Use it as motivation for a generative action-head ablation, not as source code. |
| [zycrobot/minimind-vla](https://github.com/zycrobot/minimind-vla) | Side-by-side epsilon prediction, velocity prediction (flow matching), and x-prediction action experts; episode-based storage | Compare deterministic action chunks with a small flow-matching head on exactly the same demonstrations and seeds. |
| [Stanford-ILIAD/openvla-mini](https://github.com/Stanford-ILIAD/openvla-mini) | Residual-VQ action chunking, multiple images, Qwen2.5 0.5B backbone, three-seed evaluation | Run action-horizon and execution-horizon ablations. Keep wrist/history images as an extension. Do not claim to train the 0.5B model on this laptop. |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) SmolVLA | A pretrained VLM plus flow-matching action expert; image masks; state/action padding; action queue; cached visual-language prefix | Add a CPU-scale flow-matching action expert and an explicit receding-horizon action queue. Preserve valid-action masks at episode ends. |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Flow-based action chunks, FAST action tokenization, normalization statistics, dataset conversion, policy-server boundary | Save normalization statistics with checkpoints and keep inference behind a stable policy/ROS boundary. FAST is documented but out of scope for the CPU baseline. |
| [octo-models/octo](https://github.com/octo-models/octo) | Multiple RGB cameras and a two-observation history window; language or goal-image conditioning | Add a temporal-context ablation after the single-frame baseline is stable. Do not add unused modalities only for keywords. |
| [openvla/openvla](https://github.com/openvla/openvla) | Large-scale action tokenization, RLDS mixtures, strict offline/inference sanity checks, multi-seed closed-loop evaluation | Keep episode-level train/validation splits, replay demonstrations before training, compare offline action error with closed-loop success, and report mean plus standard deviation. |

## Adopted experiment matrix and current status

| Area | Adopted idea | Status in this repository |
| --- | --- | --- |
| Action representation | Deterministic and flow-matching continuous chunks | Both heads implemented; deterministic is the selected trained pipeline; formal matched Flow Matching result remains future work |
| Deployment | Replan every step or execute 2/4 actions | Receding-horizon evaluator implemented; formal execution-horizon matrix remains future work |
| Inputs | RGB + language + proprioception | Implemented with 15D proprioception; previous-frame and wrist-camera remain extensions |
| Spatial reasoning | Auditable target/goal grounding | Implemented at coarse 8 x 8 and language-conditioned 16 x 16 resolution, plus train-only calibration |
| Generalization | Domain randomization and language counterfactuals | Infrastructure implemented; formal robustness run is recorded separately from the clean final |
| Evaluation | Episode-disjoint offline metrics and unseen closed loop | Implemented with task balance, Wilson interval, stage funnel, latency and failure traces |
| Reproducibility | Saved normalization, metadata, action queue and policy boundary | Checkpoint, dataset fingerprint, runtime command and machine-readable hybrid policy boundary are saved |

## Deliberate differences

- This project uses a real articulated SO-ARM100 model, MuJoCo contact, and a
  Cartesian DLS-IK controller instead of a point robot or direct object teleport.
- The 1.39M-parameter selected model is trained from scratch so every module can be
  explained. It is not called a foundation model.
- Grounding heads are supervised with camera projections and exported as
  heatmaps. Attention is treated as an auditable signal, not proof of causal
  reasoning.
- `contact_assisted` grasping is labeled explicitly and benchmarked separately
  from strict contact physics.
- SLAM is assigned to a mobile sensor platform. A fixed tabletop arm does not
  need SLAM, so the project does not manufacture that dependency.
- Direct action learning and the selected hybrid controller are kept as separate
  claims. The direct Stage 5 policy failed 0/12 development episodes; the
  high-resolution VLA-grounded hybrid pipeline reached 34/60 on a frozen final.

## Reading notes

- OpenVLA's troubleshooting guide recommends replaying demonstrations, matching
  offline inference errors to training metrics, collecting at an appropriate
  control rate, avoiding idle-action dominance, and ensuring coverage of all
  test variations. These checks are part of this repository's data audit.
- SmolVLA's action queue separates predicted chunk length from the number of
  actions executed before replanning. This distinction is retained in
  `execution_horizon`.
- Residual VQ and FAST solve a different problem from direct regression: they
  compress or tokenize multimodal action sequences. They are useful extensions,
  but neither is required to call the small baseline a VLA.
