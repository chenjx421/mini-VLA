# Reproducible Results

This directory contains reviewable result artifacts promoted from formal
experiment runs. Raw traces stay under ignored `outputs/`; selected JSON,
JSONL, NPZ evidence and deployable checkpoints are tracked in Git.

Rules:

1. No number is typed into the README before its source JSON exists.
2. Training seeds are aggregated by seed mean and sample standard deviation.
3. Episode success also reports a Wilson 95% interval.
4. Offline Tiny-VLA metrics and closed-loop success are never conflated.
5. `contact` and `contact_assisted` results remain separate.
6. Interrupted or smoke-test directories are not promoted here.
7. `scripts/build_artifact_manifest.py` hashes every promoted result, checkpoint,
   dataset file and documentation asset.

## Contents

- `ppo_reach/`: three 100,352-step runs and aggregate seed statistics.
- `vla_stage6/`: high-resolution grounding training and calibration summaries.
- `vla_final_v1/`: frozen Stage 5 60-episode final.
- `vla_final_v2/`: frozen Stage 6 60-episode final.
- `vla_robustness_dr/`: frozen 30-episode domain-randomized run.
- `expert_benchmark/`: paired strict/assisted 100-episode benchmarks.
- `ros_arm/`: ROS2 RGB-D/JointState/TF probe.
- `ros_slam/`: map/trajectory capture summary.
- `claim_ledger.md`: allowed claims, scope and limitations.
- `experiment_registry.md`: complete experiment index, including rejected runs.
- `artifact_manifest.json`: path, size and SHA256 for promoted artifacts.

Regenerate the manifest after changing a promoted artifact:

```bash
python scripts/build_artifact_manifest.py
```
