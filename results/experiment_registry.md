# Experiment Registry

该表是工程日志的快速索引。`final` 只表示该实验本身已完成，不表示项目最终结论。

| ID | Status | Purpose | Main result | Evidence |
| --- | --- | --- | --- | --- |
| dataset-v2-120 | final | 平衡专家数据 | 120 episodes, 18,972 frames | `datasets/so_arm_pick_place_v2_120_dr/statistics.json` |
| expert-assisted-seed10000-n100 | final | waypoint expert, assisted grasp | 98/100; Wilson 93.0%-99.4% | `results/expert_benchmark/contact_assisted` |
| expert-contact-seed10000-n100 | final | same seeds, strict MuJoCo contact | 38/100; Wilson 29.1%-47.8% | `results/expert_benchmark/contact` |
| ppo-reach-s1 | final | PPO state baseline seed 1 | 15/20 success, 100,352 steps | `results/ppo_reach/seed1` |
| ppo-reach-s2 | final | same config seed 2 | 4/20 success, 100,352 steps | `results/ppo_reach/seed2` |
| ppo-reach-s3 | final | same config seed 3 | 16/20 success, 100,352 steps | `results/ppo_reach/seed3` |
| ppo-reach-3seed | final aggregate | training-seed variance | 58.3% mean, 33.3% sample std; pooled 35/60 | `results/ppo_reach/aggregate_summary.json` |
| vla-det-s1-e15 | final offline | 原始 Tiny-VLA BC | action MAE 0.06443, phase 95.14% | `outputs/tiny_vla_det_seed1_v2` |
| vla-det-closed-loop-dev | failed diagnostic | 原模型闭环 | 0/2, both timeout | `outputs/tiny_vla_det_seed1_v2_failure_trace` |
| vla-offline-sliced | final diagnostic | 找第一处分叉 | initial MAE 0.33566 vs global first-action 0.05682 | `outputs/tiny_vla_det_seed1_v2_offline_diagnostics` |
| dagger-v1-beta050 | final dataset | learner-state corrections | 4,225 states, 7 phases covered | `datasets/dagger_v1_seed30000_beta050/manifest.json` |
| dagger-naive-e1 | completed, rejected | 朴素混合微调 | correction 改善，validation early 变差 | `outputs/tiny_vla_dagger_v1_smoke_epoch1` |
| ablation-A-critical | completed | 仅关键状态采样 | initial MAE 0.32393 | `outputs/ablation_critical_sampling_only_e1` |
| ablation-B-grounded | completed | 坐标进入 action query | initial MAE 0.31086, dev closed loop 0/6 | `outputs/ablation_grounded_action_e1` |
| ablation-C-refined | completed | 亚 patch 坐标精修 | grounding L2 0.02102 | `outputs/ablation_grounded_refined_action_e1` |
| grounded-refined-stage2-s22 | completed | 正式候选微调 | initial MAE 0.21481, early MAE 0.07485 | `outputs/tiny_vla_grounded_refined_dagger_v1_stage2_seed22` |
| grounded-refined-dev12 | failed diagnostic | 二次闭环检查 | 0/12; approach waypoint transition failure | `outputs/tiny_vla_grounded_refined_dagger_v1_stage2_seed22_closed_loop_dev12` |
| dagger-v2-beta020 | completed | waypoint/phase-boundary corrections | 5,239 states, 8/24 mixed success | `datasets/dagger_v2_seed35000_beta020` |
| dagger-v2-stage3-s23 | completed | 聚合 demo + DAgger v1/v2 | initial MAE 0.20135, early MAE 0.08737 | `outputs/tiny_vla_dagger_v2_stage3_seed23` |
| dagger-v2-stage3-dev12 | failed diagnostic | Stage 3 闭环检查 | 0/12, no bilateral contact | `outputs/tiny_vla_dagger_v2_stage3_seed23_closed_loop_dev12_gain1` |
| cartesian-gain2-dev12 | completed, rejected | 检验统一动作欠幅 | 0/12; distance/return 均退化 | `outputs/tiny_vla_dagger_v2_stage3_seed23_closed_loop_dev12_gain2` |
| cartesian-proprio-stage4 | completed | 12D -> 15D proprio 配对消融 | initial MAE 0.17546; dy corr 0.409 | `outputs/tiny_vla_cartesian_proprio_stage4_seed23` |
| cartesian-proprio-stage4-dev12 | failed diagnostic | 两个 15D checkpoint 闭环 | both 0/12; 3/12 reach descend condition | `outputs/tiny_vla_cartesian_proprio_stage4_seed23_closed_loop_dev12_best_action` |
| geometry-aligned-stage5 | completed | 3D grounding + phase-conditioned action | initial MAE 0.16886; world L2 0.03965 m | `outputs/tiny_vla_geometry_aligned_stage5_seed23` |
| geometry-aligned-stage5-direct-dev12 | failed diagnostic | 直接执行 Stage 5 action | 0/12, no bilateral contact | `outputs/tiny_vla_geometry_aligned_stage5_seed23_closed_loop_dev12` |
| hybrid-calibrated-first-dev12 | completed | learned pixel + camera geometry + servo | 3/12 first genuine VLA-grounded successes | `outputs/hybrid_vla_calibrated_dev12_seed25000` |
| grounding-global-affine | completed | train-only pixel calibration | target 31.56 -> 27.83 mm on validation | `outputs/grounding_calibration_stage5_train_only` |
| hybrid-global-calibration-dev12 | completed | calibration-only closed loop | 7/12 | `outputs/hybrid_vla_dev12_calibration_only` |
| hybrid-contact-search-dev12 | completed | contact-failure local search | 9/12 without calibration | `outputs/hybrid_vla_dev12_search_only_r18` |
| hybrid-final-config-dev12 | completed, selected | global calibration + 18 mm search + 400 steps | 10/12 | `outputs/hybrid_vla_dev12_calibration_search_r18_h400` |
| hybrid-close8-dev12 | completed, rejected | reduce close wait 18 -> 8 | 6/12; physics/contact delay | `outputs/hybrid_vla_dev12_calibration_search_r18_h400_close8` |
| grounding-language-affine | completed, rejected for control | color/side-conditioned calibration | offline improved, closed loop 8/12 | `outputs/hybrid_vla_dev12_language_calibration_search_r18_h400` |
| hybrid-final-v1-seed50000-n60 | final, frozen | untouched-seed balanced test | 30/60; 95% CI 37.7%-62.3%; contact 32/60 | `outputs/hybrid_vla_final_test_seed50000_n60` |
| highres-stage6-s24 | completed | frozen-backbone 16x16 language grounding | target 31.56 -> 27.27 mm; action preserved | `outputs/tiny_vla_highres_stage6_seed24` |
| highres-stage6-dev24 | completed, selected | matched new-seed comparison | 14/24 vs Stage 5 11/24 | `outputs/hybrid_vla_dev24_v2_seed26000_stage6` |
| hybrid-final-v2-seed60000-n60 | final, frozen | Stage 6 untouched-seed balanced test | 34/60; contact 39/60; p50 18.17 ms | `outputs/hybrid_vla_final_v2_seed60000_n60` |
| hybrid-robustness-dr-seed70000-n30 | final, frozen | new-seed domain-randomized test | 18/30; contact 23/30; latency confounded by host load | `results/vla_robustness_dr` |
| ros2-arm-probe-jazzy | final | WSL2 ROS2 arm sensor/TF contract | 11/11 checks; 6 joints; RGB-D; 12 TF pairs | `results/ros_arm` |
