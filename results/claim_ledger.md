# Claim Ledger

## Dataset

**Claim:** The formal expert dataset has 120 successful episodes and 18,972
frames, balanced across three target colors and two goal sides.

**Evidence:** `datasets/so_arm_pick_place_v2_120_dr/statistics.json`

**Scope:** MuJoCo, `contact_assisted`, domain-randomized demonstrations.

**Known limitation:** Cubes have no orientation requirement, so wrist action is
constant zero.

**Reproduce:**

```bash
evla-audit-dataset datasets/so_arm_pick_place_v2_120_dr
```

## PPO Reach

**Claim:** Across three training seeds, the state-based PPO reach policy used
100,352 environment steps per seed and reached 58.3% mean final success with
33.3 percentage-point sample standard deviation. Seed results were 15/20,
4/20 and 16/20; pooled evaluation success was 35/60 with Wilson 95% interval
45.7%-69.9%.

**Evidence:** `results/ppo_reach/aggregate_summary.json`, per-seed summary and
metrics files, and `docs/assets/ppo_multiseed_summary.png`.

**Scope:** Privileged 37D state, reach task, three training seeds, 20 independent
final evaluation episodes per trained policy.

**Known limitation:** This is not an RGB policy and not the pick-place task. The
large seed-level variance shows that 100k-step training is not stable. The pooled
Wilson interval is conditional on only three trained policies, so seed-level
mean and sample standard deviation remain the primary summary.

**Reproduce:** `evla-train-ppo --task reach --total-steps 100000 ...`

## Tiny-VLA Offline

**Claim:** The original 1,153,169-parameter deterministic Tiny-VLA completed
15 epochs and reached 0.06443 validation action MAE, 95.14% phase accuracy and
0.03511 grounding L2 on an episode-disjoint, task-stratified validation split.
The selected Stage 6 model has 1,389,783 total parameters, freezes the Stage 5
backbone, and trains 145,601 high-resolution grounding parameters.

**Evidence:** Stage 6 metrics and summary in `results/vla_stage6/`.

**Scope:** Deterministic 8-step action chunk; 15D proprioception; episode-disjoint,
task-stratified validation; 16 x 16 high-resolution grounding branch.

**Known limitation:** The original global metrics hid a 0.33566 first-action MAE
at episode initial states. The Stage 6 validation target error after train-only
calibration is 24.30 mm and only 23.3% of samples fall inside a 13 mm window.
Offline action or grounding error does not establish closed-loop success.

## Direct Tiny-VLA Closed Loop

**Claim:** The original checkpoint failed 2/2 development episodes by timeout;
a one-epoch grounding-conditioned ablation failed 6/6; the Stage 5 direct action
policy failed 12/12 task-balanced development episodes without bilateral contact.

**Evidence:** Experiment IDs `vla-det-closed-loop-dev`,
`ablation-B-grounded`, and `geometry-aligned-stage5-direct-dev12` in
`results/experiment_registry.md`.

**Diagnosis:** The global first-action MAE was 0.05682 but initial-state MAE was
0.33566. Cartesian proprioception improved directional correlation, but the
remaining 3-4 cm camera-to-world spatial error was larger than the 6-13 mm grasp
window.

**Resume boundary:** Do not describe the hybrid result below as direct action
decoder or end-to-end policy success.

## Hybrid Tiny-VLA Closed Loop

**Claim:** The frozen Stage 6 hybrid pipeline succeeded in 34/60 task-balanced
episodes on seeds 60000-60059, with Wilson 95% interval 44.1%-68.4%. It reached
bilateral contact, grasp and lift in 39/60 episodes. CPU latency was p50
18.17 ms and p95 48.27 ms.

**Evidence:** `results/vla_final_v2/summary.json`,
`results/vla_final_v2/episodes.jsonl`, and
`docs/assets/vla_final_comparison.png`.

**Scope:** MuJoCo `contact_assisted`; no domain randomization in this final;
Tiny-VLA predicts language-conditioned target/goal pixels; train-only affine
calibration and camera geometry convert pixels to world coordinates; a Cartesian
state machine and contact-failure local search generate actions.

**Policy boundary:** The action path does not receive privileged target or goal
coordinates. Simulation truth is written to traces only for diagnostics. This is
a VLA-grounded hybrid policy, not direct learned action success.

**Known limitation:** The two final Wilson intervals overlap, so the observed
30/60 to 34/60 increase is not claimed as statistically significant. p95 latency
exceeds the 20 ms period of a strict 50 Hz controller. Goal calibration exploits
two fixed left/right goal stations and is not a general target-pose result.

## Domain-Randomized Robustness

**Claim:** With the selected Stage 6 pipeline frozen, a new 30-episode,
task-balanced domain-randomized run reached 18/30 success, Wilson 95% interval
42.3%-75.4%, with 23 grasps and 22 lifts.

**Evidence:** `results/vla_robustness_dr/summary.json`,
`results/vla_robustness_dr/episodes.jsonl`, and
`docs/assets/vla_domain_randomized_success.gif`.

**Scope:** Seeds 70000-70029 with MuJoCo domain randomization enabled.

**Known limitation:** The interval overlaps the clean final interval and there
are only five episodes per task, so this does not show a significant improvement
or establish broad robustness. The run executed in the background while other
host work continued; its latency is not a controlled comparison.

## Expert Controller

**Claim:** On paired seeds 10000-10099, the waypoint expert succeeded in 98/100
episodes with `contact_assisted` and 38/100 with strict MuJoCo `contact`.
Wilson 95% intervals are 93.0%-99.4% and 29.1%-47.8%, respectively.

**Evidence:** `results/expert_benchmark/contact_assisted/`,
`results/expert_benchmark/contact/`, and
`docs/assets/expert_grasp_mode_comparison.png`.

**Scope:** State-based physical waypoint expert, clean MuJoCo visual/physics
settings, six tasks rotated over the same seed range, 300-step budget.

**Known limitation:** `contact_assisted` activates a weld-like stabilization only
after bilateral finger contact and a close command. It is a deliberate
experimental simplification, not strict frictional grasp performance.

## ROS2 SLAM

**Claim:** The MuJoCo mobile platform produced a nonempty 0.05 m occupancy grid
through ROS2 Jazzy `slam_toolbox`; under intentionally strong synthetic drift,
the recorded endpoint error was lower for SLAM pose than raw odometry.

**Evidence:** `results/ros_slam/summary.json` and trajectory image after
promotion from `outputs/ros_slam_capture`.

**Scope:** 60-second synthetic LiDAR run in WSL2, endpoint position error with
timestamp-nearest ground truth.

**Known limitation:** Endpoint error is not full-trajectory ATE/RPE and is not a
real sensor accuracy claim.

## ROS2 Arm Bridge

**Claim:** The MuJoCo arm bridge built and ran in WSL2 ROS2 Jazzy, and an
independent probe passed 11/11 checks covering six-joint `JointState`, RGB8,
32FC1 depth, matching image shape, CameraInfo, task metadata and camera TF.

**Evidence:** `results/ros_arm/summary.json`,
`docs/assets/ros2_arm_rgb.png`, `docs/assets/ros2_arm_depth.png`, and
`scripts/run_ros_arm_probe_wsl.sh`.

**Scope:** Simulated SO-ARM100 sensor and command interface.

**Known limitation:** This verifies the ROS2 message/TF contract in simulation;
it is not a SO-101 hardware deployment claim.
