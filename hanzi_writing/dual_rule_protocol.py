"""Frozen 15-stroke and 12-move rule libraries for the dual-RNN experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.canonical_protocol import canonical_target_trajectory
from hanzi_writing.geometry import (
    ComponentTrajectory,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    cue_scale,
    move_conditions,
)


VARIANT = "dual_fixed_rule_loss_comparison_v1"
MODEL_KINDS = ("stroke", "move")
LOSS_ARMS = ("baseline", "full_trial", "onset_window")
FIXED_DELAY_STEPS = 50
FIXED_SPEED_NAME = "slow"
FIXED_SPEED_SCALAR = 0.0

STROKE_RULES = (
    "long_heng_ke_0",
    "long_heng_jiang_5",
    "medium_heng_mu_0",
    "medium_heng_jiang_3",
    "short_heng_ke_3",
    "long_shu_mu_1",
    "medium_shu_jiang_4",
    "short_shu_ke_1",
    "pie_mu_2",
    "na_mu_3",
    "dian_jiang_0",
    "dian_jiang_1",
    "ti_jiang_2",
    "hengzhe_ke_2",
    "shugou_ke_4",
)

MOVE_RULES = (
    "mu_move_0",
    "mu_move_1",
    "mu_move_2",
    "jiang_move_0",
    "jiang_move_1",
    "jiang_move_2",
    "jiang_move_3",
    "jiang_move_4",
    "ke_move_0",
    "ke_move_1",
    "ke_move_2",
    "ke_move_3",
)

# Existing canonical representatives retain 150/200 intervals. Additional
# occurrences inherit the measured distance-per-interval of their successful
# same-family canonical representative.
_STROKE_SPECS = (
    ("long_heng_ke_0", "ke", 0, "heng", 150, "existing_canonical_150"),
    ("long_heng_jiang_5", "jiang", 5, "heng", 126, "scaled_from_canonical_heng"),
    ("medium_heng_mu_0", "mu", 0, "heng", 106, "scaled_from_canonical_heng"),
    ("medium_heng_jiang_3", "jiang", 3, "heng", 75, "scaled_from_canonical_heng"),
    ("short_heng_ke_3", "ke", 3, "heng", 45, "scaled_from_canonical_heng"),
    ("long_shu_mu_1", "mu", 1, "shu", 150, "existing_canonical_150"),
    ("medium_shu_jiang_4", "jiang", 4, "shu", 56, "scaled_from_canonical_shu"),
    ("short_shu_ke_1", "ke", 1, "shu", 39, "scaled_from_canonical_shu"),
    ("pie_mu_2", "mu", 2, "pie", 150, "existing_canonical_150"),
    ("na_mu_3", "mu", 3, "na", 150, "existing_canonical_150"),
    ("dian_jiang_0", "jiang", 0, "dian", 150, "existing_canonical_150"),
    ("dian_jiang_1", "jiang", 1, "dian", 129, "scaled_from_canonical_dian"),
    ("ti_jiang_2", "jiang", 2, "ti", 150, "existing_canonical_150"),
    ("hengzhe_ke_2", "ke", 2, "hengzhe", 200, "existing_canonical_200"),
    ("shugou_ke_4", "ke", 4, "shugou", 200, "existing_canonical_200"),
)

# The common move pace is the median measured distance per interval of the six
# successfully fitted canonical simple strokes: 0.8502257335 mm/interval.
_MOVE_INTERVALS = {
    "mu_move_0": 125,
    "mu_move_1": 161,
    "mu_move_2": 147,
    "jiang_move_0": 61,
    "jiang_move_1": 121,
    "jiang_move_2": 56,
    "jiang_move_3": 73,
    "jiang_move_4": 79,
    "ke_move_0": 199,
    "ke_move_1": 65,
    "ke_move_2": 42,
    "ke_move_3": 94,
}


@dataclass(frozen=True)
class FixedRuleCondition:
    model_kind: str
    rule: str
    condition_id: str
    source_rule: str
    source_character: str
    source_component_index: int
    movement_intervals: int
    points_m: np.ndarray
    spatial_cue: np.ndarray
    movement_subphase: tuple[str, ...]
    canonical_rounding: Mapping[str, object] | None
    timing_basis: str
    source_geometry_sha256: str
    derived_path_geometry_sha256: str
    temporal_sampling_sha256: str


def rule_names(model_kind: str) -> tuple[str, ...]:
    if model_kind == "stroke":
        return STROKE_RULES
    if model_kind == "move":
        return MOVE_RULES
    raise ValueError(f"unknown dual-RNN model kind: {model_kind}")


def rule_index(model_kind: str) -> dict[str, int]:
    return {name: index for index, name in enumerate(rule_names(model_kind))}


def input_size(model_kind: str) -> int:
    return len(rule_names(model_kind)) + 18


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_payload(points: np.ndarray) -> list[list[float]]:
    return np.asarray(points, dtype=np.float64).tolist()


def _source_hash(
    *,
    model_kind: str,
    character: str,
    component_index: int,
    points: np.ndarray,
) -> str:
    return _sha256_json(
        {
            "model_kind": model_kind,
            "source_character": character,
            "source_component_index": component_index,
            "source_points_m": _array_payload(points),
        }
    )


def _sampling_hash(rule: str, points: np.ndarray) -> str:
    return _sha256_json({"rule": rule, "points_m": _array_payload(points)})


def _derived_hash(
    rule: str,
    dense: np.ndarray,
    rounding: Mapping[str, object] | None,
) -> str:
    if rounding is None:
        geometry: object = _array_payload(dense)
    else:
        geometry = {
            name: rounding[name]
            for name in ("S_xy_m", "A_xy_m", "C_xy_m", "B_xy_m", "E_xy_m", "trim_ratio")
        }
    return _sha256_json({"rule": rule, "derived_geometry": geometry})


def _authority_occurrences(geometry: GeometryConfig) -> dict[tuple[str, int], dict[str, Any]]:
    characters, _ = authority.physical_characters(geometry.target_long_medium_steps)
    records = authority.primitive_occurrences(characters)
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["character"]), int(record["stroke_index"]))
        if key in output:
            raise RuntimeError(f"duplicate authority stroke occurrence: {key}")
        output[key] = record
    if len(output) != 15:
        raise RuntimeError("dual Stroke RNN requires exactly 15 authority occurrences")
    return output


def stroke_conditions(geometry: GeometryConfig) -> tuple[FixedRuleCondition, ...]:
    occurrences = _authority_occurrences(geometry)
    output = []
    for rule, character, index, source_rule, intervals, timing_basis in _STROKE_SPECS:
        record = occurrences[(character, index)]
        if str(record["rule"]) != source_rule:
            raise RuntimeError(f"authority rule differs for {rule}")
        start = np.asarray(record["start_xy_m"], dtype=np.float64)
        relative = np.asarray(record["relative_points_m"], dtype=np.float64)
        dense = relative + start
        condition_id = f"stroke_{character}_{index:02d}_exact"
        rounding = None
        if source_rule in {"hengzhe", "shugou"}:
            legacy_condition = StrokeCondition(
                condition_id=condition_id,
                rule=source_rule,
                primitive_id=f"dual_{rule}",
                character=character,
                stroke_index=index,
                variant="exact",
                start_xy_m=start,
                relative_points_m=relative,
            )
            target = canonical_target_trajectory(legacy_condition, geometry)
            points = np.asarray(target["points_m"], dtype=np.float64)
            subphase = tuple(str(value) for value in target["subphase"])
            rounding = target["rounding"]
        else:
            points = authority.resample_by_arclength(dense, intervals)
            subphase = tuple("movement" for _ in range(len(points)))
        if len(points) != intervals + 1:
            raise RuntimeError(f"movement sample count differs for {rule}")
        if not np.array_equal(points[0], dense[0]) or not np.array_equal(
            points[-1], dense[-1]
        ):
            raise RuntimeError(f"authority endpoints changed for {rule}")
        if not np.isfinite(points).all() or np.any(
            np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0.0
        ):
            raise RuntimeError(f"invalid target sampling for {rule}")
        output.append(
            FixedRuleCondition(
                model_kind="stroke",
                rule=rule,
                condition_id=condition_id,
                source_rule=source_rule,
                source_character=character,
                source_component_index=index,
                movement_intervals=intervals,
                points_m=points,
                spatial_cue=np.zeros(2, dtype=np.float64),
                movement_subphase=subphase,
                canonical_rounding=rounding,
                timing_basis=timing_basis,
                source_geometry_sha256=_source_hash(
                    model_kind="stroke",
                    character=character,
                    component_index=index,
                    points=dense,
                ),
                derived_path_geometry_sha256=_derived_hash(
                    rule, dense, rounding
                ),
                temporal_sampling_sha256=_sampling_hash(rule, points),
            )
        )
    if tuple(condition.rule for condition in output) != STROKE_RULES:
        raise RuntimeError("dual stroke rule order differs")
    return tuple(output)


def fixed_move_conditions(geometry: GeometryConfig) -> tuple[FixedRuleCondition, ...]:
    source = {
        (condition.character, condition.transition_index): condition
        for condition in move_conditions(geometry, include_jitter=False)
    }
    normalizer = cue_scale(geometry)
    output = []
    for rule in MOVE_RULES:
        character, _, index_text = rule.partition("_move_")
        index = int(index_text)
        condition: MoveCondition = source[(character, index)]
        dense = authority.straight_line(condition.start_xy_m, condition.goal_xy_m)
        intervals = _MOVE_INTERVALS[rule]
        points = authority.resample_by_arclength(dense, intervals)
        if len(points) != intervals + 1:
            raise RuntimeError(f"movement sample count differs for {rule}")
        if not np.array_equal(points[0], condition.start_xy_m) or not np.array_equal(
            points[-1], condition.goal_xy_m
        ):
            raise RuntimeError(f"authority endpoints changed for {rule}")
        output.append(
            FixedRuleCondition(
                model_kind="move",
                rule=rule,
                condition_id=condition.condition_id,
                source_rule="move",
                source_character=character,
                source_component_index=index,
                movement_intervals=intervals,
                points_m=points,
                spatial_cue=np.asarray(condition.goal_xy_m, dtype=np.float64)
                / normalizer,
                movement_subphase=tuple("movement" for _ in range(len(points))),
                canonical_rounding=None,
                timing_basis="canonical_simple_stroke_median_0.8502257335_mm_per_interval",
                source_geometry_sha256=_source_hash(
                    model_kind="move",
                    character=character,
                    component_index=index,
                    points=dense,
                ),
                derived_path_geometry_sha256=_derived_hash(rule, dense, None),
                temporal_sampling_sha256=_sampling_hash(rule, points),
            )
        )
    if tuple(condition.rule for condition in output) != MOVE_RULES:
        raise RuntimeError("dual move rule order differs")
    return tuple(output)


def conditions(model_kind: str, geometry: GeometryConfig) -> tuple[FixedRuleCondition, ...]:
    if model_kind == "stroke":
        return stroke_conditions(geometry)
    if model_kind == "move":
        return fixed_move_conditions(geometry)
    raise ValueError(f"unknown dual-RNN model kind: {model_kind}")


def component_trajectory(condition: FixedRuleCondition) -> ComponentTrajectory:
    target_length = float(
        np.linalg.norm(np.diff(condition.points_m, axis=0), axis=1).sum()
    )
    duration = condition.movement_intervals * authority.DT_S
    return ComponentTrajectory(
        rule=condition.rule,
        condition_id=condition.condition_id,
        speed_name=FIXED_SPEED_NAME,
        speed_mps=target_length / duration,
        speed_scalar=FIXED_SPEED_SCALAR,
        movement_intervals=condition.movement_intervals,
        points_m=condition.points_m,
        spatial_cue=condition.spatial_cue,
        base_movement_intervals=condition.movement_intervals,
        target_path_length_m=target_length,
        base_movement_duration_s=duration,
        total_movement_duration_s=duration,
        target_mean_spatial_path_speed_mps=target_length / duration,
        target_mean_total_path_speed_mps=target_length / duration,
        max_target_step_distance_m=float(
            np.linalg.norm(np.diff(condition.points_m, axis=0), axis=1).max()
        ),
        movement_subphase=condition.movement_subphase,
        canonical_rounding=condition.canonical_rounding,
    )


def condition_manifest(model_kind: str, geometry: GeometryConfig) -> dict[str, Any]:
    rows = []
    for condition in conditions(model_kind, geometry):
        trajectory = component_trajectory(condition)
        rows.append(
            {
                "rule_index": rule_index(model_kind)[condition.rule],
                "rule": condition.rule,
                "condition_id": condition.condition_id,
                "source_rule": condition.source_rule,
                "source_character": condition.source_character,
                "source_component_index": condition.source_component_index,
                "movement_intervals": condition.movement_intervals,
                "movement_samples": len(condition.points_m),
                "target_path_length_m": trajectory.target_path_length_m,
                "target_mean_speed_mps": trajectory.target_mean_total_path_speed_mps,
                "start_xy_m": condition.points_m[0].tolist(),
                "end_xy_m": condition.points_m[-1].tolist(),
                "spatial_cue": condition.spatial_cue.tolist(),
                "timing_basis": condition.timing_basis,
                "source_geometry_sha256": condition.source_geometry_sha256,
                "derived_path_geometry_sha256": (
                    condition.derived_path_geometry_sha256
                ),
                "temporal_sampling_sha256": condition.temporal_sampling_sha256,
            }
        )
    payload = {
        "variant": VARIANT,
        "model_kind": model_kind,
        "rule_dim": len(rule_names(model_kind)),
        "input_size": input_size(model_kind),
        "rule_names": list(rule_names(model_kind)),
        "fixed_delay_steps": FIXED_DELAY_STEPS,
        "fixed_speed_name": FIXED_SPEED_NAME,
        "fixed_speed_scalar": FIXED_SPEED_SCALAR,
        "conditions": rows,
    }
    payload["condition_manifest_sha256"] = _sha256_json(payload)
    return payload
