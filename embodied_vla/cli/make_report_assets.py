from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from embodied_vla.metrics import wilson_score_interval

PHASE_NAMES = (
    "pregrasp",
    "descend",
    "close",
    "lift",
    "transfer",
    "lower",
    "release",
)
ACTION_NAMES = ("dx", "dy", "dz", "wrist", "jaw")
TASK_BAR_COLORS = {
    "red": "#d54c4c",
    "green": "#3c9b70",
    "blue": "#3f72b5",
}


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def _prepare_axes(rows: int, columns: int, *, title: str) -> tuple[Any, Any]:
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4.4 * rows),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#f7f8f6")
    figure.suptitle(title, fontsize=17, fontweight="bold")
    axes_array = np.asarray(axes).reshape(-1)
    for axis in axes_array:
        axis.set_facecolor("#ffffff")
        axis.grid(axis="y", color="#d9ddda", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    return figure, axes_array


def plot_dataset_overview(dataset_root: Path, output_path: Path) -> None:
    statistics = json.loads(
        (dataset_root / "statistics.json").read_text(encoding="utf-8")
    )
    task_counts = statistics["task_counts"]
    phase_counts = np.asarray(statistics["phase_counts"][: len(PHASE_NAMES)])
    action_statistics = statistics["normalization"]["action_normalized_interface"]
    action_mean = np.asarray(action_statistics["mean"])
    action_std = np.asarray(action_statistics["std"])

    figure, axes = _prepare_axes(1, 3, title="Expert Dataset Audit")
    task_labels = list(task_counts)
    task_values = [task_counts[label] for label in task_labels]
    task_colors = [
        TASK_BAR_COLORS[label.split("->", maxsplit=1)[0]]
        for label in task_labels
    ]
    axes[0].bar(np.arange(len(task_labels)), task_values, color=task_colors)
    axes[0].set_xticks(np.arange(len(task_labels)), task_labels, rotation=35, ha="right")
    axes[0].set_ylabel("successful episodes")
    axes[0].set_title("Balanced language tasks")
    axes[0].set_ylim(0, max(task_values) * 1.25)
    for index, value in enumerate(task_values):
        axes[0].text(index, value + 0.4, str(value), ha="center", fontsize=9)

    axes[1].bar(
        np.arange(len(PHASE_NAMES)),
        phase_counts,
        color=["#5187a8", "#629b8b", "#d49b3e", "#9a6fb0", "#466b96", "#b46858", "#747474"],
    )
    axes[1].set_xticks(np.arange(len(PHASE_NAMES)), PHASE_NAMES, rotation=35, ha="right")
    axes[1].set_ylabel("frames")
    axes[1].set_title("Expert phase imbalance")

    x_positions = np.arange(len(ACTION_NAMES))
    axes[2].bar(x_positions, action_mean, yerr=action_std, color="#477a96", capsize=4)
    axes[2].axhline(0.0, color="#222222", linewidth=0.9)
    axes[2].set_xticks(x_positions, ACTION_NAMES)
    axes[2].set_ylabel("normalized action")
    axes[2].set_title("Action mean and standard deviation")
    axes[2].text(
        0.02,
        0.97,
        (
            f"{statistics['episodes']} episodes | {statistics['total_steps']:,} frames\n"
            f"length median {statistics['episode_length']['median']:.1f} | "
            f"rejected {statistics['rejected']}"
        ),
        transform=axes[2].transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "#f1f3f1", "edgecolor": "#c9ceca", "pad": 6},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_ppo_run(run_dir: Path, output_path: Path) -> None:
    records = _read_json_lines(run_dir / "metrics.jsonl")
    steps = np.asarray([record["global_step"] for record in records])
    train_success = np.asarray([record["train_recent_success_rate"] for record in records])
    train_return = np.asarray([record["train_recent_return"] for record in records])
    eval_steps = np.asarray(
        [record["global_step"] for record in records if "eval_success_rate" in record]
    )
    eval_success = np.asarray(
        [record["eval_success_rate"] for record in records if "eval_success_rate" in record]
    )

    figure, axes = _prepare_axes(2, 2, title="PPO Reach Training")
    axes[0].plot(steps, train_success, color="#477a96", linewidth=2, label="recent train")
    if len(eval_steps):
        axes[0].scatter(
            eval_steps,
            eval_success,
            color="#cf4f45",
            s=42,
            zorder=3,
            label="independent eval",
        )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False)

    axes[1].plot(steps, train_return, color="#47896f", linewidth=2)
    axes[1].set_ylabel("recent episodic return")

    axes[2].plot(
        steps,
        [record["approx_kl"] for record in records],
        color="#8d65a6",
        linewidth=1.8,
        label="approx KL",
    )
    axes[2].plot(
        steps,
        [record["clip_fraction"] for record in records],
        color="#d0943f",
        linewidth=1.8,
        label="clip fraction",
    )
    axes[2].set_ylabel("update diagnostic")
    axes[2].legend(frameon=False)

    axes[3].plot(
        steps,
        [record["entropy"] for record in records],
        color="#477a96",
        linewidth=1.8,
        label="entropy",
    )
    value_axis = axes[3].twinx()
    value_axis.plot(
        steps,
        [record["value_loss"] for record in records],
        color="#cf4f45",
        linewidth=1.5,
        label="value loss",
    )
    axes[3].set_ylabel("policy entropy")
    value_axis.set_ylabel("value loss")
    handles_a, labels_a = axes[3].get_legend_handles_labels()
    handles_b, labels_b = value_axis.get_legend_handles_labels()
    axes[3].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    for axis in axes:
        axis.set_xlabel("environment steps")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_ppo_multiseed(run_dirs: list[Path], output_path: Path) -> None:
    if len(run_dirs) < 2:
        raise ValueError("at least two PPO runs are required")
    records_by_run = [_read_json_lines(run_dir / "metrics.jsonl") for run_dir in run_dirs]
    summaries = [
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    seeds = [int(summary["seed"]) for summary in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("PPO run seeds must be unique")
    colors = ("#477a96", "#cf4f45", "#47896f", "#8d65a6", "#d0943f")
    figure, axes = _prepare_axes(2, 2, title="PPO Reach: Training-Seed Variance")

    for index, (records, seed) in enumerate(
        zip(records_by_run, seeds, strict=True)
    ):
        color = colors[index % len(colors)]
        steps = np.asarray([record["global_step"] for record in records])
        axes[0].plot(
            steps,
            [record["train_recent_success_rate"] for record in records],
            color=color,
            linewidth=1.8,
            label=f"seed {seed}",
        )
        eval_records = [record for record in records if "eval_success_rate" in record]
        axes[1].plot(
            [record["global_step"] for record in eval_records],
            [record["eval_success_rate"] for record in eval_records],
            color=color,
            marker="o",
            linewidth=1.4,
            label=f"seed {seed}",
        )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("recent train success")
    axes[0].legend(frameon=False)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("environment steps")
    axes[1].set_ylabel("independent eval success")
    axes[1].legend(frameon=False)

    final_rates = np.asarray(
        [summary["final_evaluation"]["success_rate"] for summary in summaries]
    )
    final_intervals = []
    for summary in summaries:
        evaluation = summary["final_evaluation"]
        successes = int(
            evaluation.get(
                "successes",
                round(evaluation["success_rate"] * evaluation["episodes"]),
            )
        )
        final_intervals.append(wilson_score_interval(successes, int(evaluation["episodes"])))
    final_intervals_array = np.asarray(final_intervals)
    x_positions = np.arange(len(seeds))
    bar_colors = [colors[index % len(colors)] for index in range(len(seeds))]
    axes[2].bar(x_positions, final_rates, color=bar_colors, width=0.62)
    axes[2].errorbar(
        x_positions,
        final_rates,
        yerr=np.vstack(
            (
                final_rates - final_intervals_array[:, 0],
                final_intervals_array[:, 1] - final_rates,
            )
        ),
        fmt="none",
        ecolor="#252525",
        capsize=5,
        linewidth=1.4,
    )
    axes[2].set_xticks(x_positions, [f"seed {seed}" for seed in seeds])
    axes[2].set_ylim(0.0, 1.08)
    axes[2].set_ylabel("final success rate")
    axes[2].set_title(
        f"mean {np.mean(final_rates):.3f}, sample std {np.std(final_rates, ddof=1):.3f}"
    )

    final_returns = [
        summary["final_evaluation"]["mean_return"] for summary in summaries
    ]
    axes[3].bar(x_positions, final_returns, color=bar_colors, width=0.62)
    axes[3].set_xticks(x_positions, [f"seed {seed}" for seed in seeds])
    axes[3].set_ylabel("final mean return")
    axes[3].set_title("Same 100,352 training steps per seed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_vla_run(run_dir: Path, output_path: Path) -> None:
    records = _read_json_lines(run_dir / "metrics.jsonl")
    epochs = np.asarray([record["epoch"] for record in records])
    figure, axes = _prepare_axes(1, 3, title="Tiny-VLA Offline Training")

    axes[0].plot(
        epochs,
        [record["train_total"] for record in records],
        color="#477a96",
        marker="o",
        label="train total",
    )
    axes[0].plot(
        epochs,
        [record["validation_total"] for record in records],
        color="#cf4f45",
        marker="o",
        label="validation total",
    )
    axes[0].set_ylabel("weighted loss")
    axes[0].legend(frameon=False)

    axes[1].plot(
        epochs,
        [record["validation_action_mae"] for record in records],
        color="#47896f",
        marker="o",
        label="action MAE",
    )
    axes[1].plot(
        epochs,
        [record["validation_grounding_l2"] for record in records],
        color="#d0943f",
        marker="o",
        label="grounding L2",
    )
    axes[1].set_ylabel("normalized error")
    axes[1].legend(frameon=False)

    axes[2].plot(
        epochs,
        [record["validation_phase_accuracy"] for record in records],
        color="#8d65a6",
        marker="o",
    )
    axes[2].set_ylim(0.0, 1.03)
    axes[2].set_ylabel("phase accuracy")

    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_xticks(epochs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_final_vla_comparison(
    final_v1_path: Path,
    final_v2_path: Path,
    output_path: Path,
) -> None:
    summaries = [
        json.loads(final_v1_path.read_text(encoding="utf-8")),
        json.loads(final_v2_path.read_text(encoding="utf-8")),
    ]
    labels = ("Stage 5 / final-v1", "Stage 6 / final-v2")
    colors = ("#477a96", "#cf4f45")
    figure, axes = _prepare_axes(2, 2, title="VLA Closed-loop Final Evaluation")

    rates = np.asarray([summary["success_rate"] for summary in summaries])
    intervals = np.asarray(
        [summary["success_wilson_95"] for summary in summaries]
    )
    axes[0].bar(labels, rates, color=colors, width=0.58)
    axes[0].errorbar(
        np.arange(2),
        rates,
        yerr=np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)),
        fmt="none",
        ecolor="#252525",
        capsize=5,
        linewidth=1.5,
    )
    axes[0].set_ylim(0.0, 0.85)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("60 unseen episodes per pipeline")
    for index, summary in enumerate(summaries):
        axes[0].text(
            index,
            rates[index] + 0.035,
            f"{summary['successes']}/{summary['episodes']}",
            ha="center",
            fontweight="bold",
        )

    funnel_names = ("contact", "grasp", "lift", "success")
    x_positions = np.arange(len(funnel_names))
    width = 0.34
    for index, (summary, label, color) in enumerate(
        zip(summaries, labels, colors, strict=True)
    ):
        values = (
            summary["episodes_with_bilateral_contact"],
            summary["episodes_ever_grasped"],
            summary["episodes_ever_lifted"],
            summary["successes"],
        )
        axes[1].bar(
            x_positions + (index - 0.5) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
    axes[1].set_xticks(x_positions, funnel_names)
    axes[1].set_ylabel("episodes")
    axes[1].set_title("Task-stage funnel")
    axes[1].legend(frameon=False)

    error_names = ("pre-grasp target", "descend / close target", "goal")
    for index, (summary, label, color) in enumerate(
        zip(summaries, labels, colors, strict=True)
    ):
        values_mm = np.asarray(
            [
                summary["mean_pregrasp_target_world_xy_error_m"],
                summary["mean_descend_close_target_world_xy_error_m"],
                summary["mean_goal_world_xy_error_m"],
            ]
        ) * 1_000.0
        axes[2].bar(
            np.arange(3) + (index - 0.5) * width,
            values_mm,
            width=width,
            color=color,
            label=label,
        )
    axes[2].axhline(13.0, color="#d0943f", linestyle="--", label="13 mm window")
    axes[2].set_xticks(np.arange(3), error_names, rotation=18, ha="right")
    axes[2].set_ylabel("world XY error (mm)")
    axes[2].set_title("Calibrated spatial error")
    axes[2].legend(frameon=False)

    latency_names = ("p50", "p95")
    for index, (summary, label, color) in enumerate(
        zip(summaries, labels, colors, strict=True)
    ):
        latency = summary["inference_latency_ms"]
        axes[3].bar(
            np.arange(2) + (index - 0.5) * width,
            [latency["p50"], latency["p95"]],
            width=width,
            color=color,
            label=label,
        )
    axes[3].axhline(20.0, color="#d0943f", linestyle="--", label="50 Hz period")
    axes[3].set_xticks(np.arange(2), latency_names)
    axes[3].set_ylabel("CPU inference latency (ms)")
    axes[3].set_title("Accuracy / real-time trade-off")
    axes[3].legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_expert_grasp_mode_comparison(
    assisted_summary_path: Path,
    contact_summary_path: Path,
    output_path: Path,
) -> None:
    summaries = [
        json.loads(contact_summary_path.read_text(encoding="utf-8")),
        json.loads(assisted_summary_path.read_text(encoding="utf-8")),
    ]
    labels = ("strict contact", "contact assisted")
    colors = ("#cf4f45", "#47896f")
    figure, axes = _prepare_axes(1, 3, title="Expert Grasp-Mode Benchmark")

    rates = np.asarray([summary["success_rate"] for summary in summaries])
    intervals = np.asarray([summary["success_wilson_95"] for summary in summaries])
    axes[0].bar(labels, rates, color=colors, width=0.58)
    axes[0].errorbar(
        np.arange(2),
        rates,
        yerr=np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)),
        fmt="none",
        ecolor="#252525",
        capsize=5,
        linewidth=1.5,
    )
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("100 paired seeds per mode")
    for index, summary in enumerate(summaries):
        axes[0].text(
            index,
            rates[index] + 0.04,
            f"{summary['successes']}/{summary['episodes']}",
            ha="center",
            fontweight="bold",
        )

    phase_order = (
        "approach",
        "descend_grasp",
        "close_gripper",
        "lift",
        "transport",
        "descend_release",
        "open_gripper",
    )
    failure_phases = [
        phase
        for phase in phase_order
        if any(phase in summary["failure_phase_histogram"] for summary in summaries)
    ]
    x_positions = np.arange(len(failure_phases))
    width = 0.34
    for index, (summary, label, color) in enumerate(
        zip(summaries, labels, colors, strict=True)
    ):
        values = [
            summary["failure_phase_histogram"].get(phase, 0)
            for phase in failure_phases
        ]
        axes[1].bar(
            x_positions + (index - 0.5) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
    axes[1].set_xticks(x_positions, failure_phases, rotation=28, ha="right")
    axes[1].set_ylabel("failed episodes")
    axes[1].set_title("Failure phase at timeout")
    axes[1].legend(frameon=False)

    task_labels = list(summaries[0]["task_success"])
    task_positions = np.arange(len(task_labels))
    for index, (summary, label, color) in enumerate(
        zip(summaries, labels, colors, strict=True)
    ):
        values = [
            summary["task_success"][task]["success_rate"] for task in task_labels
        ]
        axes[2].bar(
            task_positions + (index - 0.5) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
    axes[2].set_xticks(task_positions, task_labels, rotation=35, ha="right")
    axes[2].set_ylim(0.0, 1.08)
    axes[2].set_ylabel("success rate")
    axes[2].set_title("Task-balanced breakdown")
    axes[2].legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reproducible GitHub report plots.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--ppo-run", type=Path)
    parser.add_argument("--ppo-runs", type=Path, nargs="+")
    parser.add_argument("--vla-run", type=Path)
    parser.add_argument("--highres-vla-run", type=Path)
    parser.add_argument("--final-v1-summary", type=Path)
    parser.add_argument("--final-v2-summary", type=Path)
    parser.add_argument("--expert-assisted-summary", type=Path)
    parser.add_argument("--expert-contact-summary", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generated: list[Path] = []
    if args.dataset is not None:
        path = args.output_dir / "dataset_overview.png"
        plot_dataset_overview(args.dataset, path)
        generated.append(path)
    if args.ppo_run is not None:
        path = args.output_dir / "ppo_learning_curve.png"
        plot_ppo_run(args.ppo_run, path)
        generated.append(path)
    if args.ppo_runs is not None:
        path = args.output_dir / "ppo_multiseed_summary.png"
        plot_ppo_multiseed(args.ppo_runs, path)
        generated.append(path)
    if args.vla_run is not None:
        path = args.output_dir / "vla_training_curve.png"
        plot_vla_run(args.vla_run, path)
        generated.append(path)
    if args.highres_vla_run is not None:
        path = args.output_dir / "vla_highres_training_curve.png"
        plot_vla_run(args.highres_vla_run, path)
        generated.append(path)
    if (args.final_v1_summary is None) != (args.final_v2_summary is None):
        raise SystemExit(
            "--final-v1-summary and --final-v2-summary must be supplied together"
        )
    if args.final_v1_summary is not None and args.final_v2_summary is not None:
        path = args.output_dir / "vla_final_comparison.png"
        plot_final_vla_comparison(
            args.final_v1_summary,
            args.final_v2_summary,
            path,
        )
        generated.append(path)
    if (args.expert_assisted_summary is None) != (args.expert_contact_summary is None):
        raise SystemExit(
            "--expert-assisted-summary and --expert-contact-summary must be supplied together"
        )
    if (
        args.expert_assisted_summary is not None
        and args.expert_contact_summary is not None
    ):
        path = args.output_dir / "expert_grasp_mode_comparison.png"
        plot_expert_grasp_mode_comparison(
            args.expert_assisted_summary,
            args.expert_contact_summary,
            path,
        )
        generated.append(path)
    if not generated:
        raise SystemExit("provide at least one report input")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
