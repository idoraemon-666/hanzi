"""Frozen target, grouping, corner, and integrity rules for the fixed-duration pilot."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    SPEED_NAMES,
    ComponentTrajectory,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    build_component_trajectory,
    checkpoint_stroke_groups,
    cue_scale,
    move_conditions,
    training_conditions_by_rule,
)


HARD_COMPOUND_TIMING = {
    ("hengzhe", "fast"): (50, 25, 25, 25, 30, 55, 56),
    ("hengzhe", "medium"): (100, 49, 51, 49, 54, 105, 106),
    ("hengzhe", "slow"): (150, 74, 76, 74, 79, 155, 156),
    ("shugou", "fast"): (50, 43, 7, 43, 48, 55, 56),
    ("shugou", "medium"): (100, 87, 13, 87, 92, 105, 106),
    ("shugou", "slow"): (150, 130, 20, 130, 135, 155, 156),
}
AUTHORITY_SIGNED_TURN_DEG = {"hengzhe": -100.0, "shugou": -120.0}
EXPECTED_EXACT_STROKES = 46
EXPECTED_EXACT_MOVES = 12
EXPECTED_ATOMIC_CONDITIONS = 58
EXPECTED_CORNER_CONDITIONS = 4
EXPECTED_DIAGNOSTIC_METRIC_ROWS = 348
EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS = 35268
EXPECTED_DIAGNOSTIC_PLOTS = 18


def _dense_points(condition: StrokeCondition | MoveCondition) -> np.ndarray:
    if isinstance(condition, StrokeCondition):
        return condition.relative_points_m + condition.start_xy_m
    return authority.straight_line(condition.start_xy_m, condition.goal_xy_m)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def exact_conditions(
    geometry: GeometryConfig,
) -> tuple[StrokeCondition | MoveCondition, ...]:
    strokes = tuple(
        condition
        for group in checkpoint_stroke_groups(geometry)
        for condition in group
    )
    moves = move_conditions(geometry, include_jitter=False)
    if len(strokes) != EXPECTED_EXACT_STROKES or len(moves) != EXPECTED_EXACT_MOVES:
        raise RuntimeError("fixed exact-condition counts differ from 46 strokes and 12 moves")
    conditions = strokes + moves
    identifiers = [condition.condition_id for condition in conditions]
    if len(set(identifiers)) != EXPECTED_ATOMIC_CONDITIONS:
        raise RuntimeError("exact condition IDs are missing or duplicated")
    return conditions


def condition_rule(condition: StrokeCondition | MoveCondition) -> str:
    return condition.rule if isinstance(condition, StrokeCondition) else "move"


def condition_primitive_id(condition: StrokeCondition | MoveCondition) -> str:
    return condition.primitive_id if isinstance(condition, StrokeCondition) else condition.condition_id


def corner_reference_mapping(trajectory: ComponentTrajectory) -> dict[str, Any]:
    if trajectory.corner_xy_m is None or trajectory.dwell_entry_index is None:
        raise ValueError("corner reference mapping requires a compound trajectory")
    points = np.asarray(trajectory.points_m, dtype=np.float64)
    entry = int(trajectory.dwell_entry_index)
    exit_index = int(trajectory.dwell_exit_index)
    corner = np.asarray(trajectory.corner_xy_m, dtype=np.float64)
    first = points[: entry + 1]
    second = np.concatenate((corner[None, :], points[exit_index + 1 :]), axis=0)
    first_cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(first, axis=0), axis=1)))
    )
    second_cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(second, axis=0), axis=1)))
    )
    first_length = float(first_cumulative[-1])
    second_length = float(second_cumulative[-1])
    pre_reference_from_start = 0.9 * first_length
    post_reference_from_corner = 0.1 * second_length
    pre_segment_index = int(
        np.searchsorted(first_cumulative, pre_reference_from_start, side="left")
    )
    post_segment_index = int(
        np.searchsorted(second_cumulative, post_reference_from_corner, side="left")
    )
    base_post_index = entry + post_segment_index
    pre_full_index = pre_segment_index
    post_full_index = base_post_index + trajectory.corner_dwell_intervals
    if not (pre_full_index < entry < exit_index < post_full_index):
        raise RuntimeError("corner reference full-trace indices are not strictly ordered")

    def neighbors(cumulative: np.ndarray, index: int) -> dict[str, float]:
        return {
            "previous_m": float(cumulative[max(0, index - 1)]),
            "selected_m": float(cumulative[index]),
            "next_m": float(cumulative[min(len(cumulative) - 1, index + 1)]),
        }

    incoming = corner - points[pre_full_index]
    outgoing = points[post_full_index] - corner
    target_angle = signed_angle_deg(incoming, outgoing)
    target_angle_difference = wrap_deg(
        target_angle - AUTHORITY_SIGNED_TURN_DEG[trajectory.rule]
    )
    if abs(target_angle_difference) > 1e-9:
        raise RuntimeError("discrete corner reference direction differs from authority")
    return {
        "pre_reference_distance_from_corner_m": 0.1 * first_length,
        "post_reference_distance_from_corner_m": post_reference_from_corner,
        "pre_reference_search_distance_from_segment_start_m": pre_reference_from_start,
        "pre_segment_local_index": pre_segment_index,
        "post_segment_local_index": post_segment_index,
        "pre_base_spatial_index": pre_segment_index,
        "post_base_spatial_index": base_post_index,
        "pre_full_index": pre_full_index,
        "corner_entry_full_index": entry,
        "corner_exit_full_index": exit_index,
        "post_full_index": post_full_index,
        "pre_cumulative_neighbors": neighbors(first_cumulative, pre_segment_index),
        "post_cumulative_neighbors": neighbors(second_cumulative, post_segment_index),
        "pre_actual_fraction_from_corner": float(
            (first_length - first_cumulative[pre_segment_index]) / first_length
        ),
        "post_actual_fraction_from_corner": float(
            second_cumulative[post_segment_index] / second_length
        ),
        "target_corner_angle_deg": target_angle,
        "authority_signed_turn_deg": AUTHORITY_SIGNED_TURN_DEG[trajectory.rule],
        "target_angle_minus_authority_deg": target_angle_difference,
        "numpy_version": np.__version__,
        "target_array_sha256": _sha256_array(points),
    }


def validate_fixed_trajectory(trajectory: ComponentTrajectory) -> dict[str, Any]:
    points = np.asarray(trajectory.points_m, dtype=np.float64)
    if not np.isfinite(points).all():
        raise RuntimeError("fixed-duration target contains non-finite values")
    if len(points) != trajectory.movement_intervals + 1:
        raise RuntimeError("movement samples must equal intervals plus one")
    differences = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(differences, axis=1)
    if trajectory.corner_dwell_intervals:
        entry = int(trajectory.dwell_entry_index)
        exit_index = int(trajectory.dwell_exit_index)
        if exit_index - entry != 5:
            raise RuntimeError("compound dwell must contain exactly five intervals")
        spatial_steps = np.concatenate((step_lengths[:entry], step_lengths[exit_index:]))
        if np.any(spatial_steps <= 0.0):
            raise RuntimeError("compound non-dwell target interval is not strictly positive")
        if not np.array_equal(
            differences[entry:exit_index], np.zeros((5, 2), dtype=np.float64)
        ):
            raise RuntimeError("compound dwell target differences are not exactly zero")
        expected = HARD_COMPOUND_TIMING[(trajectory.rule, trajectory.speed_name)]
        base, first, second, expected_entry, expected_exit, total, samples = expected
        actual = (
            trajectory.base_movement_intervals,
            entry,
            trajectory.base_movement_intervals - entry,
            entry,
            exit_index,
            trajectory.movement_intervals,
            len(points),
        )
        if actual != expected:
            raise RuntimeError(f"compound timing table mismatch: {actual} != {expected}")
        if not np.array_equal(
            points[entry : exit_index + 1],
            np.repeat(trajectory.corner_xy_m[None, :], 6, axis=0),
        ):
            raise RuntimeError("all six dwell samples must equal the authority corner")
        if first < 1 or second < 1 or first + second != base:
            raise RuntimeError("compound spatial interval allocation is invalid")
    elif np.any(step_lengths <= 0.0):
        raise RuntimeError("ordinary target interval is not strictly positive")
    computed_length = float(step_lengths.sum())
    if not math.isclose(
        computed_length,
        trajectory.target_path_length_m,
        rel_tol=0.0,
        abs_tol=8 * np.finfo(np.float64).eps * max(1.0, computed_length),
    ):
        raise RuntimeError("resampled target path length differs from authority path length")
    return {
        "movement_intervals": trajectory.movement_intervals,
        "movement_samples": len(points),
        "corner_dwell_intervals": trajectory.corner_dwell_intervals,
        "target_path_length_m": trajectory.target_path_length_m,
        "base_movement_duration_s": trajectory.base_movement_duration_s,
        "total_movement_duration_s": trajectory.total_movement_duration_s,
        "target_mean_spatial_path_speed_mps": trajectory.target_mean_spatial_path_speed_mps,
        "target_mean_total_path_speed_mps": trajectory.target_mean_total_path_speed_mps,
        "max_target_step_distance_m": trajectory.max_target_step_distance_m,
    }


def build_fixed_condition_manifest(
    geometry: GeometryConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalizer = cue_scale(geometry)
    grouped = training_conditions_by_rule(geometry)
    all_conditions = tuple(
        condition for rule in ACTIVE_RULES for condition in grouped[rule]
    )
    exact = exact_conditions(geometry)
    exact_ids = {condition.condition_id for condition in exact}
    corner_mappings: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for condition in all_conditions:
        rule = condition_rule(condition)
        for speed_name in SPEED_NAMES:
            key = (condition.condition_id, speed_name)
            if key in seen:
                raise RuntimeError(f"duplicate condition-duration manifest row: {key}")
            seen.add(key)
            trajectory = build_component_trajectory(
                condition, speed_name, normalizer, geometry
            )
            metrics = validate_fixed_trajectory(trajectory)
            mapping = None
            if trajectory.corner_dwell_intervals:
                mapping = corner_reference_mapping(trajectory)
                mapping_key = (rule, speed_name)
                if mapping_key not in corner_mappings:
                    corner_mappings[mapping_key] = mapping
                else:
                    index_keys = (
                        "pre_segment_local_index",
                        "post_segment_local_index",
                        "pre_base_spatial_index",
                        "post_base_spatial_index",
                        "pre_full_index",
                        "corner_entry_full_index",
                        "corner_exit_full_index",
                        "post_full_index",
                    )
                    if any(
                        mapping[name] != corner_mappings[mapping_key][name]
                        for name in index_keys
                    ):
                        raise RuntimeError("compound placements produced different corner mappings")
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "category": "stroke" if isinstance(condition, StrokeCondition) else "move",
                    "rule": rule,
                    "primitive_id": condition_primitive_id(condition),
                    "source_character": condition.character,
                    "source_component_index": (
                        condition.stroke_index
                        if isinstance(condition, StrokeCondition)
                        else condition.transition_index
                    ),
                    "variant": condition.variant,
                    "atomic_exact_condition": condition.condition_id in exact_ids,
                    "duration_condition": speed_name,
                    "duration_cue": authority.TRAIN_SPEED_SCALAR[speed_name],
                    **metrics,
                    "dwell_entry_index": "" if trajectory.dwell_entry_index is None else trajectory.dwell_entry_index,
                    "dwell_exit_index": "" if trajectory.dwell_exit_index is None else trajectory.dwell_exit_index,
                    "pre_full_index": "" if mapping is None else mapping["pre_full_index"],
                    "post_full_index": "" if mapping is None else mapping["post_full_index"],
                    "target_array_sha256": _sha256_array(trajectory.points_m),
                }
            )

    corner_exact = [
        condition
        for condition in exact
        if condition_rule(condition) in geometry.corner_dwell_rules
    ]
    if len(corner_exact) != EXPECTED_CORNER_CONDITIONS:
        raise RuntimeError("N_corner must equal the frozen hard count 4")
    dynamic_trajectory_rows = 2 * sum(
        build_component_trajectory(condition, speed, normalizer, geometry).movement_intervals + 1
        for condition in exact
        for speed in SPEED_NAMES
    )
    dynamic_metric_rows = len(exact) * len(SPEED_NAMES) * 2
    if dynamic_metric_rows != EXPECTED_DIAGNOSTIC_METRIC_ROWS:
        raise RuntimeError("future diagnostic metric row count must equal 348")
    if dynamic_trajectory_rows != EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS:
        raise RuntimeError("future diagnostic trajectory row count must equal 35268")
    speed_ranges = {}
    for speed_name in SPEED_NAMES:
        selected = [row for row in rows if row["duration_condition"] == speed_name]
        speed_ranges[speed_name] = {
            "target_mean_spatial_path_speed_mps_min": min(
                row["target_mean_spatial_path_speed_mps"] for row in selected
            ),
            "target_mean_spatial_path_speed_mps_max": max(
                row["target_mean_spatial_path_speed_mps"] for row in selected
            ),
            "target_mean_total_path_speed_mps_min": min(
                row["target_mean_total_path_speed_mps"] for row in selected
            ),
            "target_mean_total_path_speed_mps_max": max(
                row["target_mean_total_path_speed_mps"] for row in selected
            ),
        }
    audit = {
        "training_manifest_row_count": len(rows),
        "atomic_exact_condition_count": len(exact),
        "stroke_exact_placement_count": EXPECTED_EXACT_STROKES,
        "exact_move_count": EXPECTED_EXACT_MOVES,
        "N_corner": len(corner_exact),
        "future_diagnostic_metric_rows": dynamic_metric_rows,
        "future_diagnostic_trajectory_rows": dynamic_trajectory_rows,
        "future_diagnostic_plot_count": EXPECTED_DIAGNOSTIC_PLOTS,
        "speed_ranges": speed_ranges,
        "corner_reference_mappings": {
            f"{rule}:{speed}": mapping
            for (rule, speed), mapping in sorted(corner_mappings.items())
        },
        "aggregation_order": [
            "atomic_exact_condition",
            "primitive_occurrence_equal_mean",
            "rule_equal_mean_over_primitives",
            "equal_mean_over_9_rules",
        ],
    }
    if len(corner_mappings) != 6:
        raise RuntimeError("exactly six rule-duration corner mappings must be frozen")
    return rows, audit


def write_condition_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("condition manifest cannot be empty")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def signed_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.linalg.norm(first) == 0.0 or np.linalg.norm(second) == 0.0:
        raise RuntimeError("corner angle vector norm is zero")
    cross = float(first[0] * second[1] - first[1] * second[0])
    dot = float(np.dot(first, second))
    return wrap_deg(math.degrees(math.atan2(cross, dot)))


def compound_movement_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    trajectory: ComponentTrajectory,
    mapping: dict[str, Any],
    dt_seconds: float,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if actual.shape != target.shape or actual.shape != trajectory.points_m.shape:
        raise ValueError("compound actual and target traces must be aligned")
    entry = int(trajectory.dwell_entry_index)
    exit_index = int(trajectory.dwell_exit_index)
    last = len(actual) - 1
    errors = np.linalg.norm(actual - target, axis=1)
    spatial_indices = np.concatenate(
        (np.arange(0, entry + 1), np.arange(exit_index + 1, last + 1))
    )
    if len(spatial_indices) != trajectory.base_movement_intervals + 1:
        raise RuntimeError("spatial error index count must equal D+1")
    steps = np.linalg.norm(np.diff(actual, axis=0), axis=1)
    pre_path = float(steps[:entry].sum(dtype=np.float64))
    dwell_path = float(steps[entry:exit_index].sum(dtype=np.float64))
    post_path = float(steps[exit_index:].sum(dtype=np.float64))
    spatial_path = pre_path + post_path
    total_path = pre_path + dwell_path + post_path
    residual = abs(total_path - spatial_path - dwell_path)
    tolerance = 8 * np.finfo(np.float64).eps * max(1.0, total_path)
    if residual > tolerance:
        raise RuntimeError("actual path-length identity residual exceeds tolerance")
    pre_index = int(mapping["pre_full_index"])
    post_index = int(mapping["post_full_index"])
    incoming_actual = actual[entry] - actual[pre_index]
    outgoing_actual = actual[post_index] - actual[exit_index]
    corner = np.asarray(trajectory.corner_xy_m, dtype=np.float64)
    incoming_target = corner - target[pre_index]
    outgoing_target = target[post_index] - corner
    actual_angle = signed_angle_deg(incoming_actual, outgoing_actual)
    target_angle = signed_angle_deg(incoming_target, outgoing_target)
    dwell_speeds = steps[entry:exit_index] / dt_seconds
    return {
        "spatial_movement_mean_euclidean_m": float(errors[spatial_indices].sum() / len(spatial_indices)),
        "total_movement_mean_euclidean_m": float(errors.sum() / len(errors)),
        "pre_corner_segment_mean_error": float(errors[: entry + 1].mean()),
        "post_corner_segment_mean_error": float(errors[exit_index:].mean()),
        "actual_spatial_path_length_m": spatial_path,
        "total_actual_path_length_m": total_path,
        "dwell_path_length_m": dwell_path,
        "path_length_identity_residual": residual,
        "incoming_actual_norm_m": float(np.linalg.norm(incoming_actual)),
        "outgoing_actual_norm_m": float(np.linalg.norm(outgoing_actual)),
        "incoming_target_norm_m": float(np.linalg.norm(incoming_target)),
        "outgoing_target_norm_m": float(np.linalg.norm(outgoing_target)),
        "actual_corner_angle_deg": actual_angle,
        "target_corner_angle_deg": target_angle,
        "corner_angle_error_deg": wrap_deg(actual_angle - target_angle),
        "corner_angle_absolute_error_deg": abs(wrap_deg(actual_angle - target_angle)),
        "exit_direction_error_deg": wrap_deg(
            math.degrees(math.atan2(outgoing_actual[1], outgoing_actual[0]))
            - math.degrees(math.atan2(outgoing_target[1], outgoing_target[0]))
        ),
        "dwell_mean_speed_mps": float(dwell_speeds.mean()),
        "dwell_max_speed_mps": float(dwell_speeds.max()),
        "dwell_max_excursion_m": float(
            np.linalg.norm(actual[entry : exit_index + 1] - actual[entry], axis=1).max()
        ),
        "dwell_max_target_corner_error_m": float(
            np.linalg.norm(actual[entry : exit_index + 1] - corner, axis=1).max()
        ),
    }


def aggregate_exact_metrics(
    rows: Iterable[dict[str, Any]], metric_names: Iterable[str]
) -> dict[str, Any]:
    row_list = list(rows)
    metrics = tuple(metric_names)
    atomic_keys = [
        (row["checkpoint"], row["duration_condition"], row["condition_id"])
        for row in row_list
    ]
    if len(atomic_keys) != len(set(atomic_keys)):
        raise RuntimeError("atomic diagnostic metric rows are duplicated")
    by_context: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in row_list:
        by_context[(row["checkpoint"], row["duration_condition"])].append(row)
    output = {}
    for context, context_rows in by_context.items():
        by_primitive: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in context_rows:
            by_primitive[(row["rule"], row["primitive_id"])].append(row)
        primitive_means = {
            key: {metric: float(np.mean([row[metric] for row in values])) for metric in metrics}
            for key, values in by_primitive.items()
        }
        rule_means = {}
        for rule in ACTIVE_RULES:
            primitives = [value for (item_rule, _), value in primitive_means.items() if item_rule == rule]
            if not primitives:
                raise RuntimeError(f"aggregation is missing rule: {rule}")
            rule_means[rule] = {
                metric: float(np.mean([value[metric] for value in primitives]))
                for metric in metrics
            }
        output[f"{context[0]}:{context[1]}"] = {
            "per_rule": rule_means,
            "equal_mean_over_9_rules": {
                metric: float(np.mean([rule_means[rule][metric] for rule in ACTIVE_RULES]))
                for metric in metrics
            },
        }
    return output
