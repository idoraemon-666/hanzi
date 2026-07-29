"""Exact-only best/final diagnostics for the fixed-duration nine-task pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hanzi_writing.envs import HanziComponentEnv
from hanzi_writing.fit_diagnostic import _write_plots
from hanzi_writing.fixed_duration_protocol import (
    EXPECTED_ATOMIC_CONDITIONS,
    EXPECTED_CORNER_CONDITIONS,
    EXPECTED_DIAGNOSTIC_METRIC_ROWS,
    EXPECTED_DIAGNOSTIC_PLOTS,
    EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS,
    EXPECTED_EXACT_MOVES,
    EXPECTED_EXACT_STROKES,
    aggregate_exact_metrics,
    build_fixed_condition_manifest,
    compound_movement_metrics,
    condition_primitive_id,
    condition_rule,
    corner_reference_mapping,
)
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    FIXED_DURATION_TIMING_MODE,
    PROJECT,
    SPEED_NAMES,
    ComponentTrajectory,
    MoveCondition,
    StrokeCondition,
    checkpoint_stroke_groups,
    load_geometry_config,
    move_conditions,
)
from hanzi_writing.training import (
    _assert_state_equal,
    _make_effector,
    _rollout,
    _state_clone,
    load_hanzi_policy_checkpoint,
    validate_training_config,
)
from train import _fixed_rng


BASE_METRIC_NAMES = (
    "target_path_length_m",
    "actual_spatial_path_length_m",
    "total_actual_path_length_m",
    "spatial_path_length_ratio",
    "spatial_path_length_ratio_absolute_deviation",
    "movement_mean_l1_m",
    "spatial_movement_mean_euclidean_m",
    "total_movement_mean_euclidean_m",
    "endpoint_l1_m",
    "endpoint_euclidean_m",
    "movement_mean_l1_per_target_path_length",
    "spatial_movement_mean_euclidean_per_target_path_length",
    "total_movement_mean_euclidean_per_target_path_length",
    "endpoint_l1_per_target_path_length",
    "endpoint_euclidean_per_target_path_length",
)

CORNER_METRIC_NAMES = (
    "pre_corner_segment_mean_error",
    "post_corner_segment_mean_error",
    "dwell_path_length_m",
    "path_length_identity_residual",
    "incoming_actual_norm_m",
    "outgoing_actual_norm_m",
    "incoming_target_norm_m",
    "outgoing_target_norm_m",
    "actual_corner_angle_deg",
    "target_corner_angle_deg",
    "corner_angle_error_deg",
    "corner_angle_absolute_error_deg",
    "exit_direction_error_deg",
    "dwell_mean_speed_mps",
    "dwell_max_speed_mps",
    "dwell_max_excursion_m",
    "dwell_max_target_corner_error_m",
)

METRIC_FIELDS = (
    "checkpoint",
    "duration_condition",
    "speed",
    "category",
    "rule",
    "primitive_id",
    "condition_id",
    "source_character",
    "source_component_index",
    "variant",
    "movement_intervals",
    "movement_samples",
    "base_movement_intervals",
    "corner_dwell_intervals",
    "base_movement_duration_s",
    "total_movement_duration_s",
    "target_mean_spatial_path_speed_mps",
    "target_mean_total_path_speed_mps",
    "max_target_step_distance_m",
    "dwell_entry_index",
    "dwell_exit_index",
    "pre_full_index",
    "post_full_index",
    "target_execution_max_abs_difference_from_authority_m",
    *BASE_METRIC_NAMES,
    *CORNER_METRIC_NAMES,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def load_fixed_duration_diagnostic_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("fixed-duration diagnostic config must be an object")
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "device",
            "seed",
            "pilot_config",
            "geometry_config",
            "frozen_preflight_condition_manifest",
            "frozen_preflight_condition_manifest_sha256",
            "checkpoints",
            "duration_conditions",
            "delay_steps",
            "network_noise",
            "deterministic_observation",
            "include_jitter",
            "diagnostic_contract",
            "output_directory",
        },
        "fixed-duration diagnostic config",
    )
    if (
        config["project"] != PROJECT
        or config["run_kind"] != "fixed_duration_fit_diagnostic"
        or config["variant"] != "fixed_duration_9task_pilot_v1"
        or config["device"] != "cpu"
    ):
        raise ValueError("fixed-duration diagnostic identity differs")
    if config["seed"] != 1042 or config["delay_steps"] != 50:
        raise ValueError("diagnostic seed or delay differs from the frozen protocol")
    if config["duration_conditions"] != list(SPEED_NAMES):
        raise ValueError("diagnostic must contain all three duration conditions")
    if (
        config["network_noise"] is not False
        or config["deterministic_observation"] is not True
        or config["include_jitter"] is not False
    ):
        raise ValueError("diagnostic must be deterministic, noise-free, and exact-only")
    if config["pilot_config"] != (
        "configurations/hanzi_stroke_temporal_composition_fixed_duration_pilot_v1.json"
    ):
        raise ValueError("diagnostic pilot config path differs")
    if config["geometry_config"] != (
        "configurations/hanzi_stroke_temporal_composition_fixed_duration_geometry.json"
    ):
        raise ValueError("diagnostic geometry config path differs")
    if config["checkpoints"] != {
        "best": (
            "runs/hanzi_stroke_temporal_composition/fixed_duration_pilot/dev42/"
            "best_checkpoint.pt"
        ),
        "final": (
            "runs/hanzi_stroke_temporal_composition/fixed_duration_pilot/dev42/"
            "final_continuation_checkpoint.pt"
        ),
    }:
        raise ValueError("diagnostic checkpoint paths differ")
    if config["diagnostic_contract"] != {
        "conditions": "46_exact_stroke_placements_and_12_exact_moves",
        "aggregation": "atomic_then_primitive_then_rule_then_equal_9_rule_mean",
        "behavioral_pass_fail": "not_defined",
        "causal_effect": "not_claimed",
        "complete_character_rollout": False,
        "N_corner": EXPECTED_CORNER_CONDITIONS,
        "metric_rows": EXPECTED_DIAGNOSTIC_METRIC_ROWS,
        "trajectory_rows": EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS,
        "plots": EXPECTED_DIAGNOSTIC_PLOTS,
    }:
        raise ValueError("fixed-duration diagnostic contract differs")
    geometry = load_geometry_config(config["geometry_config"])
    if geometry.timing_mode != FIXED_DURATION_TIMING_MODE:
        raise ValueError("diagnostic geometry is not fixed-duration")
    return config


def _render_manifest_row(row: dict[str, Any], fields: Iterable[str]) -> dict[str, str]:
    return {
        field: "" if row[field] == "" else str(row[field])
        for field in fields
    }


def verify_frozen_manifest(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, str]]]:
    path = Path(config["frozen_preflight_condition_manifest"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != config["frozen_preflight_condition_manifest_sha256"]:
        raise RuntimeError("frozen preflight condition manifest SHA-256 differs")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        frozen_rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    geometry = load_geometry_config(config["geometry_config"])
    current_rows, audit = build_fixed_condition_manifest(geometry)
    if len(frozen_rows) != len(current_rows):
        raise RuntimeError("current and frozen condition manifest row counts differ")
    for index, (frozen, current) in enumerate(zip(frozen_rows, current_rows)):
        if frozen != _render_manifest_row(current, fields):
            raise RuntimeError(f"condition manifest differs at row {index}")
    exact = {
        (row["condition_id"], row["duration_condition"]): row
        for row in frozen_rows
        if row["atomic_exact_condition"] == "True"
    }
    if len(exact) != EXPECTED_ATOMIC_CONDITIONS * len(SPEED_NAMES):
        raise RuntimeError("frozen exact-condition manifest coverage differs")
    return audit, exact


def fixed_movement_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    trajectory: ComponentTrajectory,
    mapping: dict[str, Any] | None,
    dt_seconds: float,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if actual.shape != target.shape or actual.shape != trajectory.points_m.shape:
        raise ValueError("fixed diagnostic movement traces are not aligned")
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise RuntimeError("fixed diagnostic movement trace is non-finite")
    target_length = float(trajectory.target_path_length_m)
    if target_length <= 0.0:
        raise RuntimeError("authority target path length must be positive")
    errors_l1 = np.abs(actual - target).sum(axis=1)
    errors_euclidean = np.linalg.norm(actual - target, axis=1)
    actual_steps = np.linalg.norm(np.diff(actual, axis=0), axis=1)
    actual_path = float(actual_steps.sum(dtype=np.float64))
    metrics: dict[str, float] = {
        "target_path_length_m": target_length,
        "actual_spatial_path_length_m": actual_path,
        "total_actual_path_length_m": actual_path,
        "movement_mean_l1_m": float(errors_l1.mean()),
        "spatial_movement_mean_euclidean_m": float(errors_euclidean.mean()),
        "total_movement_mean_euclidean_m": float(errors_euclidean.mean()),
        "endpoint_l1_m": float(errors_l1[-1]),
        "endpoint_euclidean_m": float(errors_euclidean[-1]),
    }
    if trajectory.corner_dwell_intervals:
        if mapping is None:
            raise RuntimeError("compound trajectory is missing its frozen mapping")
        metrics.update(
            compound_movement_metrics(
                actual, target, trajectory, mapping, dt_seconds
            )
        )
    spatial_ratio = metrics["actual_spatial_path_length_m"] / target_length
    metrics.update(
        {
            "spatial_path_length_ratio": spatial_ratio,
            "spatial_path_length_ratio_absolute_deviation": abs(spatial_ratio - 1.0),
            "movement_mean_l1_per_target_path_length": (
                metrics["movement_mean_l1_m"] / target_length
            ),
            "spatial_movement_mean_euclidean_per_target_path_length": (
                metrics["spatial_movement_mean_euclidean_m"] / target_length
            ),
            "total_movement_mean_euclidean_per_target_path_length": (
                metrics["total_movement_mean_euclidean_m"] / target_length
            ),
            "endpoint_l1_per_target_path_length": (
                metrics["endpoint_l1_m"] / target_length
            ),
            "endpoint_euclidean_per_target_path_length": (
                metrics["endpoint_euclidean_m"] / target_length
            ),
        }
    )
    return metrics


def _metadata(
    checkpoint: str,
    duration: str,
    condition: StrokeCondition | MoveCondition,
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "duration_condition": duration,
        "speed": duration,
        "category": "stroke" if isinstance(condition, StrokeCondition) else "move",
        "rule": condition_rule(condition),
        "primitive_id": condition_primitive_id(condition),
        "condition_id": condition.condition_id,
        "source_character": condition.character,
        "source_component_index": (
            condition.stroke_index
            if isinstance(condition, StrokeCondition)
            else condition.transition_index
        ),
        "variant": condition.variant,
    }


def _extract_curves(
    checkpoint: str,
    duration: str,
    conditions: tuple[StrokeCondition | MoveCondition, ...],
    result: dict[str, Any],
    env: HanziComponentEnv,
) -> list[dict[str, Any]]:
    movement_start, movement_end = result["epoch_bounds"]["movement"]
    actual_batch = (
        result["xy"][:, movement_start:movement_end].detach().cpu().numpy()
    )
    target_batch = (
        result["target"][:, movement_start:movement_end].detach().cpu().numpy()
    )
    trajectories = env.component_trajectories
    if len(conditions) != len(trajectories) or len(conditions) != len(actual_batch):
        raise RuntimeError("diagnostic rollout batch alignment differs")
    anchor = np.asarray(env.anchor_m, dtype=np.float64)
    curves = []
    for index, (condition, trajectory) in enumerate(zip(conditions, trajectories)):
        actual = np.asarray(actual_batch[index], dtype=np.float64) - anchor
        target = np.asarray(target_batch[index], dtype=np.float64) - anchor
        mapping = (
            corner_reference_mapping(trajectory)
            if trajectory.corner_dwell_intervals
            else None
        )
        metrics = fixed_movement_metrics(
            actual, target, trajectory, mapping, env.geometry_config.dt_seconds
        )
        row = {
            **_metadata(checkpoint, duration, condition),
            "movement_intervals": trajectory.movement_intervals,
            "movement_samples": len(trajectory.points_m),
            "base_movement_intervals": trajectory.base_movement_intervals,
            "corner_dwell_intervals": trajectory.corner_dwell_intervals,
            "base_movement_duration_s": trajectory.base_movement_duration_s,
            "total_movement_duration_s": trajectory.total_movement_duration_s,
            "target_mean_spatial_path_speed_mps": (
                trajectory.target_mean_spatial_path_speed_mps
            ),
            "target_mean_total_path_speed_mps": (
                trajectory.target_mean_total_path_speed_mps
            ),
            "max_target_step_distance_m": trajectory.max_target_step_distance_m,
            "dwell_entry_index": trajectory.dwell_entry_index,
            "dwell_exit_index": trajectory.dwell_exit_index,
            "pre_full_index": None if mapping is None else mapping["pre_full_index"],
            "post_full_index": None if mapping is None else mapping["post_full_index"],
            "target_execution_max_abs_difference_from_authority_m": float(
                np.max(np.abs(target - trajectory.points_m))
            ),
            **{name: None for name in CORNER_METRIC_NAMES},
            **metrics,
        }
        subphase = (
            np.asarray(trajectory.movement_subphase, dtype="U16")
            if trajectory.movement_subphase is not None
            else np.full(len(actual), "movement", dtype="U16")
        )
        curves.append(
            {
                "metadata": row,
                "metrics": {
                    **row,
                    "path_length_ratio": row["spatial_path_length_ratio"],
                    "movement_mean_euclidean_per_target_path_length": row[
                        "spatial_movement_mean_euclidean_per_target_path_length"
                    ],
                },
                "movement_intervals": trajectory.movement_intervals,
                "actual": actual,
                "target": target,
                "movement_subphase": subphase,
            }
        )
    return curves


def _checkpoint_curves(
    checkpoint_name: str,
    checkpoint_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy, checkpoint = load_hanzi_policy_checkpoint(checkpoint_path)
    expected_kind = "best_shared9" if checkpoint_name == "best" else "final_continuation"
    if checkpoint.get("checkpoint_kind") != expected_kind:
        raise RuntimeError(f"{checkpoint_name} checkpoint kind differs")
    if checkpoint.get("variant") != config["variant"]:
        raise RuntimeError(f"{checkpoint_name} checkpoint variant differs")
    with Path(config["pilot_config"]).open("r", encoding="utf-8") as handle:
        pilot_config = json.load(handle)
    validate_training_config(pilot_config)
    if checkpoint.get("protocol_config") != pilot_config:
        raise RuntimeError(f"{checkpoint_name} embedded pilot config differs")
    update = int(checkpoint["update"])
    if checkpoint_name == "best":
        if update < 0 or update > 9500 or update % 500:
            raise RuntimeError("best checkpoint update is not a scheduled validation")
    elif update != 9999:
        raise RuntimeError("final checkpoint must follow update 9999")
    policy.eval()
    state_before = _state_clone(policy)
    hp = checkpoint["hp"]
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    if any(float(value) != 0.0 for value in env.obs_noise + env.action_noise):
        raise RuntimeError("diagnostic environment noise must be zero")
    geometry = load_geometry_config(config["geometry_config"])
    curves: list[dict[str, Any]] = []
    rollout_groups = 0
    with _fixed_rng(config["seed"]):
        for duration in config["duration_conditions"]:
            for group in checkpoint_stroke_groups(geometry):
                group_tuple = tuple(group)
                result = _rollout(
                    policy,
                    env,
                    hp,
                    group_tuple,
                    duration,
                    config["delay_steps"],
                    network_noise=False,
                    deterministic_observation=True,
                    track_gradients=False,
                )
                curves.extend(
                    _extract_curves(
                        checkpoint_name, duration, group_tuple, result, env
                    )
                )
                rollout_groups += 1
            for condition in move_conditions(geometry, include_jitter=False):
                result = _rollout(
                    policy,
                    env,
                    hp,
                    (condition,),
                    duration,
                    config["delay_steps"],
                    network_noise=False,
                    deterministic_observation=True,
                    track_gradients=False,
                )
                curves.extend(
                    _extract_curves(
                        checkpoint_name, duration, (condition,), result, env
                    )
                )
                rollout_groups += 1
    _assert_state_equal(state_before, policy)
    return curves, {
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_update": update,
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "sha256": _sha256(checkpoint_path),
        "rollout_group_count": rollout_groups,
        "policy_state_bitwise_unchanged": True,
        "optimizer_created": False,
    }


def _corner_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["corner_dwell_intervals"]) == 5:
            grouped[
                (row["checkpoint"], row["duration_condition"], row["rule"])
            ].append(row)
    if len(grouped) != 2 * 3 * 2:
        raise RuntimeError("compound diagnostic context count must equal 12")
    return {
        ":".join(key): {
            "condition_count": len(values),
            **{
                metric: float(np.mean([float(row[metric]) for row in values]))
                for metric in CORNER_METRIC_NAMES
            },
        }
        for key, values in sorted(grouped.items())
    }


def _trajectory_arrays(curves: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for curve in curves:
        metadata = curve["metadata"]
        samples = len(curve["actual"])
        values["actual_xy_m"].append(curve["actual"])
        values["target_xy_m"].append(curve["target"])
        values["checkpoint"].append(np.full(samples, metadata["checkpoint"], dtype="U8"))
        values["duration_condition"].append(
            np.full(samples, metadata["duration_condition"], dtype="U8")
        )
        values["condition_id"].append(
            np.full(samples, metadata["condition_id"], dtype="U64")
        )
        values["rule"].append(np.full(samples, metadata["rule"], dtype="U16"))
        values["primitive_id"].append(
            np.full(samples, metadata["primitive_id"], dtype="U64")
        )
        values["movement_index"].append(np.arange(samples, dtype=np.int64))
        values["movement_intervals"].append(
            np.full(samples, curve["movement_intervals"], dtype=np.int64)
        )
        values["phase"].append(np.full(samples, "movement", dtype="U8"))
        values["movement_subphase"].append(curve["movement_subphase"])
    return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def _require_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError(f"non-finite diagnostic value at {path}")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Fixed-duration nine-task 10k pilot diagnostic",
        "",
        "This report describes deterministic exact-only isolated-component fits. "
        "It does not define behavioral pass/fail, run complete characters, or "
        "identify a causal effect of duration or corner dwell.",
        "",
        "## Equal-weight nine-rule metrics",
        "",
        "| checkpoint | duration | spatial mean / target length | total mean / target length | endpoint / target length | spatial path ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for checkpoint in ("best", "final"):
        for duration in SPEED_NAMES:
            metrics = summary["aggregates"][f"{checkpoint}:{duration}"][
                "equal_mean_over_9_rules"
            ]
            lines.append(
                f"| {checkpoint} | {duration} | "
                f"{metrics['spatial_movement_mean_euclidean_per_target_path_length']:.9g} | "
                f"{metrics['total_movement_mean_euclidean_per_target_path_length']:.9g} | "
                f"{metrics['endpoint_euclidean_per_target_path_length']:.9g} | "
                f"{metrics['spatial_path_length_ratio']:.9g} |"
            )
    lines.extend(
        [
            "",
            "## Final minus best descriptive differences",
            "",
            "| duration | spatial normalized error difference | total normalized error difference | endpoint normalized error difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for duration in SPEED_NAMES:
        differences = summary["final_minus_best"][duration]
        lines.append(
            f"| {duration} | "
            f"{differences['spatial_movement_mean_euclidean_per_target_path_length']:.9g} | "
            f"{differences['total_movement_mean_euclidean_per_target_path_length']:.9g} | "
            f"{differences['endpoint_euclidean_per_target_path_length']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Compound-corner descriptive metrics",
            "",
            "| checkpoint | duration | rule | pre error (m) | post error (m) | actual angle (deg) | target angle (deg) | angle error (deg) | dwell path (m) | dwell max excursion (m) | dwell mean speed (m/s) | dwell max speed (m/s) | exit direction error (deg) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in ("best", "final"):
        for duration in SPEED_NAMES:
            for rule in ("hengzhe", "shugou"):
                metrics = summary["compound_corner_aggregates"][
                    f"{checkpoint}:{duration}:{rule}"
                ]
                lines.append(
                    f"| {checkpoint} | {duration} | {rule} | "
                    f"{metrics['pre_corner_segment_mean_error']:.9g} | "
                    f"{metrics['post_corner_segment_mean_error']:.9g} | "
                    f"{metrics['actual_corner_angle_deg']:.9g} | "
                    f"{metrics['target_corner_angle_deg']:.9g} | "
                    f"{metrics['corner_angle_error_deg']:.9g} | "
                    f"{metrics['dwell_path_length_m']:.9g} | "
                    f"{metrics['dwell_max_excursion_m']:.9g} | "
                    f"{metrics['dwell_mean_speed_mps']:.9g} | "
                    f"{metrics['dwell_max_speed_mps']:.9g} | "
                    f"{metrics['exit_direction_error_deg']:.9g} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The fixed-duration model was newly trained from random initialization. "
            "Duration condition changes timing samples, duration cue, realized target "
            "speed distribution, and the learned policy. For hengzhe and shugou, a "
            "fixed 50 ms dwell also has a different relative share across duration "
            "conditions. These values are descriptive, not independent causal effects.",
            "",
            "This is one seed and 10,000 updates. It does not estimate cross-seed "
            "stability or the attainable result at 75,000 updates.",
        ]
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def run_fixed_duration_diagnostic(config_path: str | Path) -> dict[str, Any]:
    config = load_fixed_duration_diagnostic_config(config_path)
    output = Path(config["output_directory"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    checkpoint_paths = {
        name: Path(path) for name, path in config["checkpoints"].items()
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hashes_before = {
        name: _sha256(path) for name, path in checkpoint_paths.items()
    }
    manifest_audit, frozen_exact = verify_frozen_manifest(config)
    output.mkdir(parents=True)
    _write_json(output / "resolved_config.json", config)

    curves: list[dict[str, Any]] = []
    checkpoint_reports = {}
    for checkpoint_name in ("best", "final"):
        checkpoint_curves, report = _checkpoint_curves(
            checkpoint_name, checkpoint_paths[checkpoint_name], config
        )
        curves.extend(checkpoint_curves)
        checkpoint_reports[checkpoint_name] = report

    rows = [dict(curve["metadata"]) for curve in curves]
    if len(rows) != EXPECTED_DIAGNOSTIC_METRIC_ROWS:
        raise RuntimeError("diagnostic metric row count must equal 348")
    keys = {
        (row["checkpoint"], row["duration_condition"], row["condition_id"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise RuntimeError("diagnostic metric combinations are missing or duplicated")
    for row in rows:
        frozen = frozen_exact[(row["condition_id"], row["duration_condition"])]
        if (
            str(row["movement_intervals"]) != frozen["movement_intervals"]
            or str(row["movement_samples"]) != frozen["movement_samples"]
            or str(row["corner_dwell_intervals"])
            != frozen["corner_dwell_intervals"]
        ):
            raise RuntimeError("diagnostic trajectory differs from frozen manifest")
    _require_finite(rows)
    _write_csv(output / "component_metrics.csv", rows)

    arrays = _trajectory_arrays(curves)
    trajectory_rows = len(arrays["movement_index"])
    if trajectory_rows != EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS:
        raise RuntimeError("diagnostic trajectory row count must equal 35268")
    if any(array.dtype == object for array in arrays.values()):
        raise RuntimeError("object arrays are forbidden in diagnostic NPZ")
    if not np.isfinite(arrays["actual_xy_m"]).all() or not np.isfinite(
        arrays["target_xy_m"]
    ).all():
        raise RuntimeError("diagnostic NPZ contains non-finite trajectories")
    np.savez_compressed(output / "movement_trajectories.npz", **arrays)
    with np.load(output / "movement_trajectories.npz", allow_pickle=False) as saved:
        if len(saved["movement_index"]) != EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS:
            raise RuntimeError("saved diagnostic NPZ row count differs")

    _write_plots(output, curves)
    plots = sorted((output / "plots").glob("*.png"))
    if len(plots) != EXPECTED_DIAGNOSTIC_PLOTS:
        raise RuntimeError("diagnostic plot count must equal 18")

    aggregates = aggregate_exact_metrics(rows, BASE_METRIC_NAMES)
    if set(aggregates) != {
        f"{checkpoint}:{duration}"
        for checkpoint in ("best", "final")
        for duration in SPEED_NAMES
    }:
        raise RuntimeError("diagnostic aggregate contexts differ")
    final_minus_best = {}
    for duration in SPEED_NAMES:
        best = aggregates[f"best:{duration}"]["equal_mean_over_9_rules"]
        final = aggregates[f"final:{duration}"]["equal_mean_over_9_rules"]
        final_minus_best[duration] = {
            metric: float(final[metric] - best[metric])
            for metric in BASE_METRIC_NAMES
        }
    corner_aggregates = _corner_aggregates(rows)
    checkpoint_hashes_after = {
        name: _sha256(path) for name, path in checkpoint_paths.items()
    }
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise RuntimeError("diagnostic modified a source checkpoint")

    artifacts = [
        output / "component_metrics.csv",
        output / "movement_trajectories.npz",
        *plots,
    ]
    summary = {
        "completed": True,
        "project": PROJECT,
        "run_kind": config["run_kind"],
        "variant": config["variant"],
        "seed": config["seed"],
        "diagnostic_only": True,
        "optimizer_created": False,
        "network_noise": False,
        "environment_noise_zero": True,
        "include_jitter": False,
        "complete_character_rollout_started": False,
        "boundary_intervention_started": False,
        "behavioral_pass_fail_defined": False,
        "causal_effect_claimed": False,
        "frozen_preflight_manifest": {
            "path": config["frozen_preflight_condition_manifest"],
            "sha256": config["frozen_preflight_condition_manifest_sha256"],
            "current_manifest_equal": True,
        },
        "checkpoints": checkpoint_reports,
        "integrity": {
            "exact_stroke_placement_count": EXPECTED_EXACT_STROKES,
            "exact_move_count": EXPECTED_EXACT_MOVES,
            "atomic_exact_condition_count": EXPECTED_ATOMIC_CONDITIONS,
            "N_corner": manifest_audit["N_corner"],
            "metric_rows": len(rows),
            "trajectory_rows": trajectory_rows,
            "plots": len(plots),
            "rollout_groups": sum(
                report["rollout_group_count"]
                for report in checkpoint_reports.values()
            ),
            "all_numeric_values_finite": True,
            "checkpoint_files_unchanged": True,
        },
        "aggregates": aggregates,
        "final_minus_best": final_minus_best,
        "compound_corner_aggregates": corner_aggregates,
        "artifact_sha256": {
            str(path.relative_to(output)): _sha256(path) for path in artifacts
        },
        "interpretation_boundary": (
            "Single-seed 10k exact-only isolated-component description; no "
            "behavioral threshold, complete-character claim, old-protocol comparison, "
            "or independent causal duration/dwell effect."
        ),
    }
    _require_finite(summary)
    _write_json(output / "fixed_duration_diagnostic_summary.json", summary)
    _write_report(output / "FIXED_DURATION_PILOT_REPORT.md", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run_fixed_duration_diagnostic(arguments.config)


if __name__ == "__main__":
    main()
