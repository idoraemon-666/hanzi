"""Frozen canonical single-duration targets and Stage-0 audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.geometry import (
    CANONICAL_GEOMETRY_VARIANT,
    CANONICAL_STROKE_RULES,
    CANONICAL_TIMING_MODE,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    load_geometry_config,
    move_conditions,
)


COMPOUND_RULES = ("hengzhe", "shugou")
EXPECTED_TARGET_ROWS = 3120
STAGE0_ARTIFACT_NAMES = (
    "canonical_condition_manifest.json",
    "canonical_target_trajectories.npz",
    "canonical_target_audit.png",
)
_TOLERANCE = 1e-12


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _characters(config: GeometryConfig) -> dict[str, authority.Character]:
    characters, _ = authority.physical_characters(config.target_long_medium_steps)
    return characters


def _unique_points(points: list[np.ndarray]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for point in points:
        value = np.asarray(point, dtype=np.float64)
        if not any(np.allclose(value, other, rtol=0.0, atol=_TOLERANCE) for other in output):
            output.append(value.copy())
    return output


def _occurrence_records(config: GeometryConfig) -> list[dict[str, Any]]:
    occurrences = authority.primitive_occurrences(_characters(config))
    starts: dict[str, list[np.ndarray]] = {
        rule: [np.zeros(2, dtype=np.float64)] for rule in CANONICAL_STROKE_RULES
    }
    for occurrence in occurrences:
        starts[str(occurrence["rule"])].append(
            np.asarray(occurrence["start_xy_m"], dtype=np.float64)
        )
    unique_starts = {rule: _unique_points(values) for rule, values in starts.items()}
    records: list[dict[str, Any]] = []
    for occurrence_index, occurrence in enumerate(occurrences):
        rule = str(occurrence["rule"])
        start = np.asarray(occurrence["start_xy_m"], dtype=np.float64)
        matching_indices = [
            index
            for index, value in enumerate(unique_starts[rule])
            if np.allclose(start, value, rtol=0.0, atol=_TOLERANCE)
        ]
        if len(matching_indices) != 1:
            raise RuntimeError("authority occurrence does not have one exact start")
        primitive_id = f"primitive_{occurrence_index:02d}_{rule}"
        original_condition_id = (
            f"{primitive_id}_start_{matching_indices[0]:02d}_exact"
        )
        records.append(
            {
                "rule": rule,
                "source_character": str(occurrence["character"]),
                "source_stroke_index": int(occurrence["stroke_index"]),
                "primitive_id": primitive_id,
                "original_condition_id": original_condition_id,
                "original_start_xy_m": start,
                "original_end_xy_m": np.asarray(
                    occurrence["end_xy_m"], dtype=np.float64
                ),
                "original_path_length_m": float(occurrence["length_m"]),
                "relative_points_m": np.asarray(
                    occurrence["relative_points_m"], dtype=np.float64
                ),
            }
        )
    return records


def canonical_occurrence_manifest(config: GeometryConfig) -> tuple[dict[str, Any], ...]:
    if config.timing_mode != CANONICAL_TIMING_MODE:
        raise ValueError("canonical occurrence selection requires canonical timing")
    grouped: dict[str, list[dict[str, Any]]] = {
        rule: [] for rule in CANONICAL_STROKE_RULES
    }
    for record in _occurrence_records(config):
        grouped[record["rule"]].append(record)
    selected: list[dict[str, Any]] = []
    for rule in CANONICAL_STROKE_RULES:
        candidates = grouped[rule]
        if not candidates:
            raise RuntimeError(f"canonical rule has no authority occurrence: {rule}")
        ordered = sorted(
            candidates,
            key=lambda row: (
                -float(row["original_path_length_m"]),
                str(row["source_character"]),
                int(row["source_stroke_index"]),
                str(row["original_condition_id"]),
            ),
        )
        winner = ordered[0]
        identity = (
            winner["source_character"],
            winner["source_stroke_index"],
            winner["original_condition_id"],
        )
        duplicates = [
            row
            for row in ordered
            if float(row["original_path_length_m"])
            == float(winner["original_path_length_m"])
            and (
                row["source_character"],
                row["source_stroke_index"],
                row["original_condition_id"],
            )
            == identity
        ]
        if len(duplicates) != 1:
            raise RuntimeError(f"canonical selection is not unique: {rule}")
        selected.append(
            {
                **winner,
                "selection_reason": (
                    "maximum physical arc length; ties resolved by "
                    "source_character, source_stroke_index, condition_id"
                ),
            }
        )
    if {row["rule"] for row in selected} != set(CANONICAL_STROKE_RULES):
        raise RuntimeError("canonical selection did not cover eight stroke rules")
    return tuple(selected)


def canonical_stroke_conditions(config: GeometryConfig) -> tuple[StrokeCondition, ...]:
    output = []
    for record in canonical_occurrence_manifest(config):
        output.append(
            StrokeCondition(
                condition_id=str(record["original_condition_id"]),
                rule=str(record["rule"]),
                primitive_id=str(record["primitive_id"]),
                character=str(record["source_character"]),
                stroke_index=int(record["source_stroke_index"]),
                variant="exact",
                start_xy_m=np.asarray(record["original_start_xy_m"], dtype=np.float64),
                relative_points_m=np.asarray(
                    record["relative_points_m"], dtype=np.float64
                ),
            )
        )
    return tuple(output)


def _unit(vector: np.ndarray, label: str) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise RuntimeError(f"canonical compound has zero-length {label}")
    return np.asarray(vector, dtype=np.float64) / length, length


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _require_straight_dense_segment(points: np.ndarray, direction: np.ndarray) -> None:
    differences = np.diff(points, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    if np.any(lengths <= 0.0):
        raise RuntimeError("authority compound contains a zero-length dense interval")
    units = differences / lengths[:, None]
    if np.max(np.abs(units @ direction - 1.0)) > 1e-10:
        raise RuntimeError("authority compound segment is not one directed straight line")
    if np.max(np.abs(units[:, 0] * direction[1] - units[:, 1] * direction[0])) > 1e-10:
        raise RuntimeError("authority compound segment is not collinear")


def _rounded_compound(
    dense: np.ndarray, config: GeometryConfig
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    dense = np.asarray(dense, dtype=np.float64)
    corner_index = authority.compound_corner_index(dense)
    before_dense = dense[: corner_index + 1]
    after_dense = dense[corner_index:]
    s = dense[0].copy()
    c = dense[corner_index].copy()
    e = dense[-1].copy()
    u_in, l_in = _unit(c - s, "incoming segment")
    u_out, l_out = _unit(e - c, "outgoing segment")
    _require_straight_dense_segment(before_dense, u_in)
    _require_straight_dense_segment(after_dense, u_out)

    trim_distance = config.canonical_trim_ratio * min(l_in, l_out)
    a = c - trim_distance * u_in
    b = c + trim_distance * u_out
    l_before = float(np.linalg.norm(a - s))
    l_after = float(np.linalg.norm(e - b))
    straight_total = (
        config.canonical_compound_intervals
        - config.canonical_transition_intervals
    )
    n_before = int(math.floor(straight_total * l_before / (l_before + l_after) + 0.5))
    n_after = straight_total - n_before
    if n_before < 1 or n_after < 1:
        raise RuntimeError("canonical compound interval allocation is empty")
    if (
        n_before + config.canonical_transition_intervals + n_after
        != config.canonical_compound_intervals
    ):
        raise RuntimeError("canonical compound interval allocation differs")

    first = authority.resample_by_arclength(np.stack((s, a)), n_before)
    parameter = np.linspace(
        0.0, 1.0, config.canonical_bezier_reference_samples, dtype=np.float64
    )[:, None]
    bezier_dense = (
        (1.0 - parameter) ** 2 * a
        + 2.0 * (1.0 - parameter) * parameter * c
        + parameter**2 * b
    )
    transition = authority.resample_by_arclength(
        bezier_dense, config.canonical_transition_intervals
    )
    last = authority.resample_by_arclength(np.stack((b, e)), n_after)
    points = np.concatenate((first, transition[1:], last[1:]), axis=0)
    if len(points) != config.canonical_compound_intervals + 1:
        raise RuntimeError("canonical compound sample count differs")
    if not np.array_equal(points[0], s) or not np.array_equal(points[-1], e):
        raise RuntimeError("canonical rounded endpoints differ from the authority")
    if not np.array_equal(first[-1], a):
        raise RuntimeError("canonical incoming line does not end at A")
    if not np.array_equal(transition[0], a) or not np.array_equal(transition[-1], b):
        raise RuntimeError("canonical transition endpoints differ from A/B")
    if not np.array_equal(last[0], b):
        raise RuntimeError("canonical outgoing line does not start at B")
    if not np.isfinite(points).all():
        raise RuntimeError("canonical compound target is non-finite")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0.0):
        raise RuntimeError("canonical compound target contains a zero interval")

    derivative_start = 2.0 * (c - a)
    derivative_end = 2.0 * (b - c)
    derivative_start_unit, _ = _unit(derivative_start, "Bezier start derivative")
    derivative_end_unit, _ = _unit(derivative_end, "Bezier end derivative")
    tangent = {
        "start_dot": float(derivative_start_unit @ u_in),
        "start_cross": _cross_2d(derivative_start_unit, u_in),
        "end_dot": float(derivative_end_unit @ u_out),
        "end_cross": _cross_2d(derivative_end_unit, u_out),
    }
    if (
        abs(tangent["start_dot"] - 1.0) > _TOLERANCE
        or abs(tangent["end_dot"] - 1.0) > _TOLERANCE
        or abs(tangent["start_cross"]) > _TOLERANCE
        or abs(tangent["end_cross"]) > _TOLERANCE
    ):
        raise RuntimeError("canonical Bezier is not analytically tangent")

    original_length = authority.arc_length(dense)
    rounded_length = authority.arc_length(points)
    subphase = tuple(
        ["pre_straight"] * n_before
        + ["transition"] * (config.canonical_transition_intervals + 1)
        + ["post_straight"] * n_after
    )
    if len(subphase) != len(points):
        raise RuntimeError("canonical compound subphase length differs")
    audit = {
        "S_xy_m": s.tolist(),
        "A_xy_m": a.tolist(),
        "C_xy_m": c.tolist(),
        "B_xy_m": b.tolist(),
        "E_xy_m": e.tolist(),
        "trim_ratio": config.canonical_trim_ratio,
        "trim_distance_m": trim_distance,
        "incoming_length_m": l_in,
        "outgoing_length_m": l_out,
        "n_before": n_before,
        "transition_intervals": config.canonical_transition_intervals,
        "n_after": n_after,
        "pre_indices": [0, n_before + 1],
        "transition_indices": [
            n_before,
            n_before + config.canonical_transition_intervals + 1,
        ],
        "post_indices": [
            n_before + config.canonical_transition_intervals,
            len(points),
        ],
        "analytic_tangency": tangent,
        "start_difference_m": float(np.linalg.norm(points[0] - dense[0])),
        "end_difference_m": float(np.linalg.norm(points[-1] - dense[-1])),
        "original_path_length_m": original_length,
        "rounded_path_length_m": rounded_length,
        "path_length_difference_m": rounded_length - original_length,
        "minimum_distance_to_original_corner_m": float(
            np.linalg.norm(points - c, axis=1).min()
        ),
    }
    return points, subphase, audit


def canonical_target_trajectory(
    condition: StrokeCondition | MoveCondition, config: GeometryConfig
) -> dict[str, Any]:
    if (
        config.timing_mode != CANONICAL_TIMING_MODE
        or config.geometry_variant != CANONICAL_GEOMETRY_VARIANT
    ):
        raise ValueError("canonical target requires the strict canonical geometry")
    if isinstance(condition, StrokeCondition):
        dense = condition.relative_points_m + condition.start_xy_m
        if condition.rule in COMPOUND_RULES:
            points, subphase, rounding = _rounded_compound(dense, config)
        else:
            points = authority.resample_by_arclength(
                dense, config.canonical_simple_intervals
            )
            subphase = tuple("movement" for _ in range(len(points)))
            rounding = None
            if not np.array_equal(points[0], dense[0]) or not np.array_equal(
                points[-1], dense[-1]
            ):
                raise RuntimeError("canonical simple-stroke endpoints changed")
    else:
        dense = authority.straight_line(condition.start_xy_m, condition.goal_xy_m)
        points = authority.resample_by_arclength(dense, config.canonical_move_intervals)
        subphase = tuple("movement" for _ in range(len(points)))
        rounding = None
    if not np.isfinite(points).all():
        raise RuntimeError("canonical target is non-finite")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0.0):
        raise RuntimeError("canonical target contains adjacent duplicate points")
    return {"points_m": points, "subphase": subphase, "rounding": rounding}


def build_canonical_manifest(config: GeometryConfig) -> dict[str, Any]:
    _, physical_scale = authority.physical_characters(
        config.target_long_medium_steps
    )
    strokes = canonical_stroke_conditions(config)
    moves = move_conditions(config, include_jitter=False)
    if len(strokes) != 8 or len(moves) != 12:
        raise RuntimeError("canonical manifest requires eight strokes and twelve moves")
    occurrence_by_rule = {
        row["rule"]: row for row in canonical_occurrence_manifest(config)
    }
    stroke_rows = []
    for condition in strokes:
        target = canonical_target_trajectory(condition, config)
        target_points = np.asarray(target["points_m"], dtype=np.float64)
        record = occurrence_by_rule[condition.rule]
        start_difference = float(
            np.linalg.norm(
                target_points[0]
                - np.asarray(record["original_start_xy_m"])
            )
        )
        end_difference = float(
            np.linalg.norm(
                target_points[-1]
                - np.asarray(record["original_end_xy_m"])
            )
        )
        if start_difference > _TOLERANCE or end_difference > _TOLERANCE:
            raise RuntimeError("canonical stroke placement differs from authority")
        stroke_rows.append(
            {
                "rule": condition.rule,
                "source_character": record["source_character"],
                "source_stroke_index": record["source_stroke_index"],
                "original_condition_id": record["original_condition_id"],
                "original_start_xy_m": record["original_start_xy_m"].tolist(),
                "original_end_xy_m": record["original_end_xy_m"].tolist(),
                "original_path_length_m": record["original_path_length_m"],
                "canonical_target_path_length_m": authority.arc_length(
                    target_points
                ),
                "start_difference_m": start_difference,
                "end_difference_m": end_difference,
                "selection_reason": record["selection_reason"],
                "movement_intervals": len(target_points) - 1,
                "movement_samples": len(target_points),
                "rounding": target["rounding"],
            }
        )
    move_rows = []
    for condition in moves:
        target = canonical_target_trajectory(condition, config)
        move_rows.append(
            {
                "condition_id": condition.condition_id,
                "source_character": condition.character,
                "transition_index": condition.transition_index,
                "start_xy_m": condition.start_xy_m.tolist(),
                "goal_xy_m": condition.goal_xy_m.tolist(),
                "movement_intervals": len(target["points_m"]) - 1,
                "movement_samples": len(target["points_m"]),
            }
        )
    return {
        "project": config.project,
        "timing_mode": config.timing_mode,
        "geometry_variant": config.geometry_variant,
        "geometry_source": config.geometry_source,
        "physical_scale_m_per_design_unit": physical_scale,
        "duration_cue": config.canonical_speed_name,
        "training_delay_steps": list(config.delay_steps),
        "validation_delay_steps": config.validation_delay_steps,
        "stroke_conditions": stroke_rows,
        "move_conditions": move_rows,
        "integrity": {
            "stroke_rule_count": len(stroke_rows),
            "exact_move_count": len(move_rows),
            "simple_stroke_samples": 151,
            "compound_stroke_samples": 201,
            "move_samples": 151,
            "maximum_stroke_start_difference_m": max(
                row["start_difference_m"] for row in stroke_rows
            ),
            "maximum_stroke_end_difference_m": max(
                row["end_difference_m"] for row in stroke_rows
            ),
            "stroke_placement_tolerance_m": _TOLERANCE,
            "all_targets_finite": True,
            "adjacent_duplicate_targets": 0,
        },
    }


def _flatten_targets(config: GeometryConfig) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for condition in canonical_stroke_conditions(config):
        target = canonical_target_trajectory(condition, config)
        entries.append(
            {
                "category": "stroke",
                "rule": condition.rule,
                "condition_id": condition.condition_id,
                **target,
            }
        )
    for condition in move_conditions(config, include_jitter=False):
        target = canonical_target_trajectory(condition, config)
        entries.append(
            {
                "category": "move",
                "rule": "move",
                "condition_id": condition.condition_id,
                **target,
            }
        )
    arrays: dict[str, list[np.ndarray]] = {
        "category": [],
        "rule": [],
        "condition_id": [],
        "sample_index": [],
        "target_xy_m": [],
        "subphase": [],
    }
    for entry in entries:
        points = np.asarray(entry["points_m"], dtype=np.float64)
        samples = len(points)
        arrays["category"].append(np.full(samples, entry["category"], dtype="U8"))
        arrays["rule"].append(np.full(samples, entry["rule"], dtype="U12"))
        arrays["condition_id"].append(
            np.full(samples, entry["condition_id"], dtype="U64")
        )
        arrays["sample_index"].append(np.arange(samples, dtype=np.int64))
        arrays["target_xy_m"].append(points)
        arrays["subphase"].append(np.asarray(entry["subphase"], dtype="U16"))
    flattened = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    if len(flattened["sample_index"]) != EXPECTED_TARGET_ROWS:
        raise RuntimeError("canonical Stage-0 target row count must equal 3120")
    return flattened, entries


def _plot_target_audit(path: Path, entries: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(4, 5, figsize=(15, 12))
    for axis, entry in zip(axes.flat, entries):
        points = np.asarray(entry["points_m"], dtype=np.float64)
        axis.plot(points[:, 0], points[:, 1], color="#4c78a8", linewidth=1.6)
        axis.scatter(points[0, 0], points[0, 1], color="#54a24b", s=18)
        axis.scatter(points[-1, 0], points[-1, 1], color="#e45756", s=18)
        rounding = entry.get("rounding")
        if rounding is not None:
            for label, color in (("A", "#f58518"), ("C", "#b279a2"), ("B", "#f58518")):
                point = np.asarray(rounding[f"{label}_xy_m"], dtype=np.float64)
                axis.scatter(point[0], point[1], color=color, s=24)
                axis.annotate(label, point, fontsize=8)
            start, end = rounding["transition_indices"]
            axis.plot(
                points[start:end, 0],
                points[start:end, 1],
                color="#e45756",
                linewidth=2.2,
            )
        axis.set_title(str(entry["condition_id"]), fontsize=8)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.suptitle("Canonical eight strokes and twelve exact moves")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_stage0_artifacts(
    config: GeometryConfig, output_directory: str | Path
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_canonical_manifest(config)
    arrays, entries = _flatten_targets(config)
    _write_json(output / "canonical_condition_manifest.json", manifest)
    np.savez_compressed(output / "canonical_target_trajectories.npz", **arrays)
    _plot_target_audit(output / "canonical_target_audit.png", entries)
    expected = set(STAGE0_ARTIFACT_NAMES)
    if {path.name for path in output.iterdir()} != expected:
        raise RuntimeError("canonical Stage-0 output set differs")
    return {
        "manifest": manifest,
        "target_rows": len(arrays["sample_index"]),
        "artifacts": sorted(str(output / name) for name in expected),
    }


def run_canonical_stage0(
    geometry_config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    config = load_geometry_config(geometry_config_path)
    if (
        config.timing_mode != CANONICAL_TIMING_MODE
        or config.geometry_variant != CANONICAL_GEOMETRY_VARIANT
    ):
        raise ValueError("canonical Stage 0 requires the strict canonical geometry")
    result = write_stage0_artifacts(config, output_directory)
    print(
        json.dumps(
            {
                "artifact_count": len(result["artifacts"]),
                "output_directory": str(Path(output_directory)),
                "target_rows": result["target_rows"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run_canonical_stage0(arguments.geometry_config, arguments.output)


if __name__ == "__main__":
    main()
