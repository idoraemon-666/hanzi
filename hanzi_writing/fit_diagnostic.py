"""Read-only isolated-component trajectory diagnostics for trained Hanzi policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hanzi_writing.geometry import (
    ACTIVE_RULES,
    PROJECT,
    SPEED_NAMES,
    MoveCondition,
    StrokeCondition,
    checkpoint_stroke_groups,
    load_geometry_config,
    move_conditions,
)
from hanzi_writing.envs import HanziComponentEnv
from hanzi_writing.training import (
    _assert_state_equal,
    _make_effector,
    _rollout,
    _state_clone,
    load_hanzi_policy_checkpoint,
)
from train import _fixed_rng


METRIC_NAMES = (
    "target_path_length_m",
    "actual_path_length_m",
    "path_length_ratio",
    "path_length_ratio_absolute_deviation",
    "movement_mean_l1_m",
    "movement_mean_euclidean_m",
    "endpoint_l1_m",
    "endpoint_euclidean_m",
    "movement_mean_l1_per_target_path_length",
    "movement_mean_euclidean_per_target_path_length",
    "endpoint_l1_per_target_path_length",
    "endpoint_euclidean_per_target_path_length",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def load_fit_diagnostic_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("fit diagnostic configuration must be a JSON object")
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "device",
            "seed",
            "geometry_config",
            "checkpoints",
            "speeds",
            "delay_steps",
            "network_noise",
            "deterministic_observation",
            "include_jitter",
            "diagnostic_contract",
            "output_directory",
        },
        "fit diagnostic configuration",
    )
    if (
        config["project"] != PROJECT
        or config["run_kind"] != "fit_diagnostic"
        or config["variant"] != "shared9_dev42"
        or config["device"] != "cpu"
    ):
        raise ValueError("fit diagnostic names the wrong project, run kind, variant, or device")
    if config["seed"] != 1042:
        raise ValueError("fit diagnostic seed must be 1042")
    if config["speeds"] != list(SPEED_NAMES) or config["delay_steps"] != 50:
        raise ValueError("fit diagnostic must use all three speeds and delay_steps=50")
    if (
        config["network_noise"] is not False
        or config["deterministic_observation"] is not True
        or config["include_jitter"] is not False
    ):
        raise ValueError("fit diagnostic must be deterministic, noise-free, and exact-only")
    if config["checkpoints"] != {
        "best": "runs/hanzi_stroke_temporal_composition/train/dev42/best_checkpoint.pt",
        "final": (
            "runs/hanzi_stroke_temporal_composition/train/dev42/"
            "final_continuation_checkpoint.pt"
        ),
    }:
        raise ValueError("fit diagnostic checkpoint paths differ from the frozen run")
    if config["diagnostic_contract"] != {
        "conditions": "all_15_primitives_all_exact_rule_starts_and_all_12_exact_moves",
        "aggregation": "per_condition_and_equal_mean_over_9_rules",
        "behavioral_pass_fail": "not_defined",
        "path_completion_metric": "not_defined",
    }:
        raise ValueError("fit diagnostic contract differs from the accepted read-only audit")
    geometry = load_geometry_config(config["geometry_config"])
    if geometry.validation_seed != config["seed"]:
        raise ValueError("fit diagnostic and geometry seeds disagree")
    return config


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def movement_metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError("actual and target must have identical [time, 2] shapes")
    if actual.shape[0] < 2:
        raise ValueError("movement trajectories must contain at least two samples")
    target_length = _path_length(target)
    if target_length <= 0.0:
        raise ValueError("target movement path length must be positive")
    actual_length = _path_length(actual)
    l1 = np.abs(actual - target).sum(axis=1)
    euclidean = np.linalg.norm(actual - target, axis=1)
    ratio = actual_length / target_length
    return {
        "target_path_length_m": target_length,
        "actual_path_length_m": actual_length,
        "path_length_ratio": ratio,
        "path_length_ratio_absolute_deviation": abs(ratio - 1.0),
        "movement_mean_l1_m": float(l1.mean()),
        "movement_mean_euclidean_m": float(euclidean.mean()),
        "endpoint_l1_m": float(l1[-1]),
        "endpoint_euclidean_m": float(euclidean[-1]),
        "movement_mean_l1_per_target_path_length": float(l1.mean() / target_length),
        "movement_mean_euclidean_per_target_path_length": float(
            euclidean.mean() / target_length
        ),
        "endpoint_l1_per_target_path_length": float(l1[-1] / target_length),
        "endpoint_euclidean_per_target_path_length": float(
            euclidean[-1] / target_length
        ),
    }


def _metadata(
    checkpoint_name: str,
    speed_name: str,
    condition: StrokeCondition | MoveCondition,
) -> dict[str, Any]:
    if isinstance(condition, StrokeCondition):
        return {
            "checkpoint": checkpoint_name,
            "speed": speed_name,
            "category": "stroke",
            "rule": condition.rule,
            "primitive_id": condition.primitive_id,
            "condition_id": condition.condition_id,
            "character": condition.character,
            "component_index": condition.stroke_index,
            "variant": condition.variant,
        }
    return {
        "checkpoint": checkpoint_name,
        "speed": speed_name,
        "category": "move",
        "rule": "move",
        "primitive_id": "",
        "condition_id": condition.condition_id,
        "character": condition.character,
        "component_index": condition.transition_index,
        "variant": condition.variant,
    }


def _extract_curves(
    checkpoint_name: str,
    speed_name: str,
    conditions: Iterable[StrokeCondition | MoveCondition],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    condition_tuple = tuple(conditions)
    movement_start, movement_end = result["epoch_bounds"]["movement"]
    actual_batch = result["xy"][:, movement_start:movement_end].detach().cpu().numpy()
    target_batch = result["target"][:, movement_start:movement_end].detach().cpu().numpy()
    if len(condition_tuple) != actual_batch.shape[0]:
        raise RuntimeError("rollout batch and condition count disagree")
    curves = []
    for index, condition in enumerate(condition_tuple):
        metadata = _metadata(checkpoint_name, speed_name, condition)
        metrics = movement_metrics(actual_batch[index], target_batch[index])
        curves.append(
            {
                "metadata": metadata,
                "metrics": metrics,
                "movement_intervals": int(actual_batch.shape[1] - 1),
                "actual": actual_batch[index],
                "target": target_batch[index],
            }
        )
    return curves


def _require_zero_environment_noise(env: HanziComponentEnv) -> None:
    if any(float(value) != 0.0 for value in env.obs_noise):
        raise RuntimeError("fit diagnostic requires zero environment observation noise")
    if any(float(value) != 0.0 for value in env.action_noise):
        raise RuntimeError("fit diagnostic requires zero environment action noise")


def _checkpoint_curves(
    checkpoint_name: str,
    checkpoint_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    geometry = load_geometry_config(config["geometry_config"])
    policy, checkpoint = load_hanzi_policy_checkpoint(checkpoint_path)
    policy.eval()
    state_before = _state_clone(policy)
    hp = checkpoint["hp"]
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    _require_zero_environment_noise(env)
    curves: list[dict[str, Any]] = []
    rollout_groups = 0
    with _fixed_rng(config["seed"]):
        for speed_name in config["speeds"]:
            for group in checkpoint_stroke_groups(geometry):
                result = _rollout(
                    policy,
                    env,
                    hp,
                    group,
                    speed_name,
                    config["delay_steps"],
                    network_noise=config["network_noise"],
                    deterministic_observation=config["deterministic_observation"],
                    track_gradients=False,
                )
                curves.extend(
                    _extract_curves(checkpoint_name, speed_name, group, result)
                )
                rollout_groups += 1
            for condition in move_conditions(geometry, include_jitter=False):
                result = _rollout(
                    policy,
                    env,
                    hp,
                    (condition,),
                    speed_name,
                    config["delay_steps"],
                    network_noise=config["network_noise"],
                    deterministic_observation=config["deterministic_observation"],
                    track_gradients=False,
                )
                curves.extend(
                    _extract_curves(checkpoint_name, speed_name, (condition,), result)
                )
                rollout_groups += 1
    _assert_state_equal(state_before, policy)
    return curves, {
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "checkpoint_update": int(checkpoint["update"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "sha256": _sha256(checkpoint_path),
        "rollout_group_count": rollout_groups,
        "policy_state_bitwise_unchanged": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in METRIC_NAMES
    }


def aggregate_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["checkpoint"]), str(row["speed"]))].append(row)
    output = []
    for checkpoint_name, speed_name in sorted(grouped):
        rows = grouped[(checkpoint_name, speed_name)]
        rows_by_rule = {
            rule: [row for row in rows if row["rule"] == rule]
            for rule in ACTIVE_RULES
        }
        if any(not rule_rows for rule_rows in rows_by_rule.values()):
            raise RuntimeError("diagnostic aggregation did not cover all nine rules")
        per_rule = {
            rule: _mean_metrics(rule_rows)
            for rule, rule_rows in rows_by_rule.items()
        }
        equal_rule_mean = {
            name: float(np.mean([per_rule[rule][name] for rule in ACTIVE_RULES]))
            for name in METRIC_NAMES
        }
        worst_mean = max(
            rows,
            key=lambda row: float(
                row["movement_mean_euclidean_per_target_path_length"]
            ),
        )
        worst_endpoint = max(
            rows,
            key=lambda row: float(row["endpoint_euclidean_per_target_path_length"]),
        )
        output.append(
            {
                "checkpoint": checkpoint_name,
                "speed": speed_name,
                "aggregation": "equal_mean_over_9_rules",
                "condition_count": len(rows),
                "condition_weighted_mean": _mean_metrics(rows),
                "equal_rule_mean": equal_rule_mean,
                "per_rule": per_rule,
                "worst_relative_movement_mean_condition": {
                    "condition_id": worst_mean["condition_id"],
                    "value": worst_mean[
                        "movement_mean_euclidean_per_target_path_length"
                    ],
                },
                "worst_relative_endpoint_condition": {
                    "condition_id": worst_endpoint["condition_id"],
                    "value": worst_endpoint[
                        "endpoint_euclidean_per_target_path_length"
                    ],
                },
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_groups(
    curves: list[dict[str, Any]],
    group_ids: list[str],
    path: Path,
    title: str,
    *,
    columns: int,
) -> None:
    rows = int(np.ceil(len(group_ids) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.6 * rows))
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, group_id in zip(flat_axes, group_ids):
        group = [
            curve
            for curve in curves
            if (curve["metadata"]["primitive_id"] or curve["metadata"]["condition_id"])
            == group_id
        ]
        for curve in group:
            origin = curve["target"][0]
            target = curve["target"] - origin
            actual = curve["actual"] - origin
            axis.plot(
                target[:, 0], target[:, 1], color="black", linewidth=1.1, alpha=0.45
            )
            axis.plot(actual[:, 0], actual[:, 1], color="tab:red", linewidth=1.0, alpha=0.65)
        mean_ratio = np.mean([curve["metrics"]["path_length_ratio"] for curve in group])
        mean_error = np.mean(
            [
                curve["metrics"][
                    "movement_mean_euclidean_per_target_path_length"
                ]
                for curve in group
            ]
        )
        axis.set_title(f"{group_id}\nlength ratio={mean_ratio:.2f}, relative mean={mean_error:.2f}")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.set_xlabel("relative x (m)")
        axis.set_ylabel("relative y (m)")
    for axis in flat_axes[len(group_ids) :]:
        axis.set_visible(False)
    figure.suptitle(f"{title}\nblack=target, red=actual")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_plots(output_dir: Path, curves: list[dict[str, Any]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir()
    checkpoints = sorted({curve["metadata"]["checkpoint"] for curve in curves})
    for checkpoint_name in checkpoints:
        for speed_name in SPEED_NAMES:
            selected = [
                curve
                for curve in curves
                if curve["metadata"]["checkpoint"] == checkpoint_name
                and curve["metadata"]["speed"] == speed_name
            ]
            stroke_ids = sorted(
                {
                    curve["metadata"]["primitive_id"]
                    for curve in selected
                    if curve["metadata"]["category"] == "stroke"
                }
            )
            move_ids = sorted(
                {
                    curve["metadata"]["condition_id"]
                    for curve in selected
                    if curve["metadata"]["category"] == "move"
                }
            )
            _plot_groups(
                selected,
                stroke_ids,
                plot_dir / f"strokes_{checkpoint_name}_{speed_name}.png",
                f"isolated strokes: {checkpoint_name}, {speed_name}",
                columns=5,
            )
            _plot_groups(
                selected,
                move_ids,
                plot_dir / f"moves_{checkpoint_name}_{speed_name}.png",
                f"isolated moves: {checkpoint_name}, {speed_name}",
                columns=4,
            )
            corner_ids = [
                curve["metadata"]["primitive_id"]
                for curve in selected
                if curve["metadata"]["category"] == "stroke"
                and curve["metadata"]["rule"] in {"hengzhe", "shugou"}
                and curve["metadata"]["condition_id"].endswith("start_00_exact")
            ]
            _plot_groups(
                selected,
                corner_ids,
                plot_dir / f"compound_corners_{checkpoint_name}_{speed_name}.png",
                f"compound corners at center start: {checkpoint_name}, {speed_name}",
                columns=2,
            )


def run_fit_diagnostic(config_path: str | Path) -> dict[str, Any]:
    config = load_fit_diagnostic_config(config_path)
    checkpoint_paths = {
        name: Path(value) for name, value in config["checkpoints"].items()
    }
    for checkpoint_path in checkpoint_paths.values():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
    output_dir = Path(config["output_directory"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "resolved_config.json", config)

    all_curves: list[dict[str, Any]] = []
    checkpoint_reports = {}
    for checkpoint_name, checkpoint_path in checkpoint_paths.items():
        curves, report = _checkpoint_curves(
            checkpoint_name, checkpoint_path, config
        )
        all_curves.extend(curves)
        checkpoint_reports[checkpoint_name] = report

    metric_rows = [
        {
            **curve["metadata"],
            "movement_intervals": curve["movement_intervals"],
            **curve["metrics"],
        }
        for curve in all_curves
    ]
    metric_fields = list(metric_rows[0])
    _write_csv(output_dir / "component_metrics.csv", metric_rows, metric_fields)

    sample_rows = []
    for curve in all_curves:
        for movement_index, (actual, target) in enumerate(
            zip(curve["actual"], curve["target"])
        ):
            sample_rows.append(
                {
                    **curve["metadata"],
                    "movement_index": movement_index,
                    "actual_x_m": float(actual[0]),
                    "actual_y_m": float(actual[1]),
                    "target_x_m": float(target[0]),
                    "target_y_m": float(target[1]),
                }
            )
    _write_csv(
        output_dir / "movement_trajectory_samples.csv",
        sample_rows,
        list(sample_rows[0]),
    )
    _write_plots(output_dir, all_curves)

    summaries = aggregate_metrics(metric_rows)
    exact_strokes = sum(
        curve["metadata"]["category"] == "stroke"
        for curve in all_curves
    ) // (len(config["checkpoints"]) * len(config["speeds"]))
    exact_moves = sum(
        curve["metadata"]["category"] == "move"
        for curve in all_curves
    ) // (len(config["checkpoints"]) * len(config["speeds"]))
    summary = {
        "project": PROJECT,
        "run_kind": "fit_diagnostic",
        "variant": config["variant"],
        "device": "cpu",
        "diagnostic_only": True,
        "formal_training_started": False,
        "optimizer_created": False,
        "behavioral_pass_fail_defined": False,
        "path_completion_metric_defined": False,
        "network_noise": False,
        "environment_observation_and_action_noise_verified_zero": True,
        "deterministic_seed": config["seed"],
        "include_jitter": False,
        "condition_grid": {
            "primitive_count": 15,
            "exact_stroke_placement_count": exact_strokes,
            "exact_move_count": exact_moves,
            "speed_count": len(config["speeds"]),
            "checkpoint_count": len(config["checkpoints"]),
            "metric_row_count": len(metric_rows),
            "trajectory_sample_row_count": len(sample_rows),
            "rollout_group_count": sum(
                report["rollout_group_count"]
                for report in checkpoint_reports.values()
            ),
        },
        "checkpoints": checkpoint_reports,
        "aggregates": summaries,
        "metric_definitions": {
            "movement_slice": "movement epoch only, including its first and last sample",
            "l1": "absolute x error plus absolute y error",
            "euclidean": "two-dimensional Euclidean position error",
            "relative_error": "position error divided by target path length",
            "path_length_ratio": "actual movement path length divided by target path length",
        },
    }
    _write_json(output_dir / "fit_diagnostic_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run_fit_diagnostic(arguments.config)


if __name__ == "__main__":
    main()
