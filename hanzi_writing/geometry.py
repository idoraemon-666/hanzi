"""Configured training and validation conditions around the geometry authority."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from hanzi_writing import hanzi_geometry_final as authority


PROJECT = "hanzi_stroke_temporal_composition"
GEOMETRY_SOURCE = "hanzi_writing/hanzi_geometry_final.py"
ACTIVE_RULES = tuple(authority.RULE_INDEX)
SPEED_NAMES = tuple(authority.TRAIN_SPEED_MPS)
LEGACY_TIMING_MODE = "legacy_arc_length_fixed_speed"
FIXED_DURATION_TIMING_MODE = "fixed_movement_duration"
CANONICAL_TIMING_MODE = "canonical_fixed_duration"
CANONICAL_GEOMETRY_VARIANT = "canonical_tangent_rounded_v1"
CANONICAL_STROKE_RULES = ACTIVE_RULES[:-1]


@dataclass(frozen=True)
class GeometryConfig:
    project: str
    geometry_source: str
    target_long_medium_steps: int
    dt_seconds: float
    stable_steps: int
    delay_steps: tuple[int, ...]
    hold_steps: int
    prepare_steps: int
    stroke_jitter_fraction: float
    stroke_jitter_copies: int
    stroke_jitter_seed: int
    validation_delay_steps: int
    validation_seed: int
    validation_network_noise: bool
    timing_mode: str = LEGACY_TIMING_MODE
    movement_intervals: tuple[tuple[str, int], ...] = ()
    corner_dwell_intervals: int = 0
    corner_dwell_rules: tuple[str, ...] = ()
    geometry_variant: str | None = None
    canonical_speed_name: str | None = None
    canonical_simple_intervals: int = 0
    canonical_compound_intervals: int = 0
    canonical_move_intervals: int = 0
    canonical_transition_intervals: int = 0
    canonical_trim_ratio: float = 0.0
    canonical_bezier_reference_samples: int = 0

    def intervals_for_speed(self, speed_name: str) -> int:
        values = dict(self.movement_intervals)
        if speed_name not in values:
            raise KeyError(f"fixed-duration intervals are missing speed: {speed_name}")
        return values[speed_name]


@dataclass(frozen=True)
class StrokeCondition:
    condition_id: str
    rule: str
    primitive_id: str
    character: str
    stroke_index: int
    variant: str
    start_xy_m: np.ndarray
    relative_points_m: np.ndarray


@dataclass(frozen=True)
class MoveCondition:
    condition_id: str
    character: str
    transition_index: int
    variant: str
    start_xy_m: np.ndarray
    goal_xy_m: np.ndarray


@dataclass(frozen=True)
class ComponentTrajectory:
    rule: str
    condition_id: str
    speed_name: str
    speed_mps: float
    speed_scalar: float
    movement_intervals: int
    points_m: np.ndarray
    spatial_cue: np.ndarray
    base_movement_intervals: int
    target_path_length_m: float
    base_movement_duration_s: float
    total_movement_duration_s: float
    target_mean_spatial_path_speed_mps: float
    target_mean_total_path_speed_mps: float
    max_target_step_distance_m: float
    corner_dwell_intervals: int = 0
    dwell_entry_index: int | None = None
    dwell_exit_index: int | None = None
    corner_xy_m: np.ndarray | None = None
    movement_subphase: tuple[str, ...] | None = None
    canonical_rounding: Mapping[str, object] | None = None


def _require_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _load_canonical_geometry_config(raw: dict[str, object]) -> GeometryConfig:
    expected_keys = {
        "project",
        "geometry_source",
        "rule_count_active",
        "rule_dim_total",
        "unused_rule_index",
        "target_long_medium_steps",
        "train_speed_mps",
        "train_speed_scalar",
        "timing_mode",
        "geometry_variant",
        "timing",
        "canonical_conditions",
        "checkpoint_validation",
    }
    _require_keys(raw, expected_keys, "canonical geometry configuration")
    timing = raw["timing"]
    conditions = raw["canonical_conditions"]
    validation = raw["checkpoint_validation"]
    if not all(isinstance(value, dict) for value in (timing, conditions, validation)):
        raise ValueError("canonical timing, conditions, and validation must be objects")
    _require_keys(
        timing,
        {"dt_seconds", "stable_steps", "delay_steps", "hold_steps", "prepare_steps"},
        "canonical timing",
    )
    _require_keys(
        conditions,
        {
            "stroke_rules",
            "stroke_selection",
            "moves",
            "speed_name",
            "simple_stroke_intervals",
            "compound_stroke_intervals",
            "move_intervals",
            "transition_intervals",
            "trim_ratio",
            "bezier_reference_samples",
        },
        "canonical conditions",
    )
    _require_keys(
        validation,
        {
            "delay_steps",
            "validation_seed",
            "network_noise",
            "deterministic_observation",
            "aggregation",
        },
        "canonical checkpoint validation",
    )
    expected_conditions = {
        "stroke_rules": list(CANONICAL_STROKE_RULES),
        "stroke_selection": (
            "maximum_physical_arc_length_then_character_stroke_condition"
        ),
        "moves": "all_12_exact_transitions",
        "speed_name": "slow",
        "simple_stroke_intervals": 150,
        "compound_stroke_intervals": 200,
        "move_intervals": 150,
        "transition_intervals": 20,
        "trim_ratio": 0.1,
        "bezier_reference_samples": 10001,
    }
    if conditions != expected_conditions:
        raise ValueError("canonical condition contract differs")
    if validation != {
        "delay_steps": 50,
        "validation_seed": 1042,
        "network_noise": False,
        "deterministic_observation": True,
        "aggregation": "single_condition_or_equal_12_moves",
    }:
        raise ValueError("canonical validation contract differs")
    if raw["timing_mode"] != CANONICAL_TIMING_MODE:
        raise ValueError("canonical geometry requires canonical_fixed_duration")
    if raw["geometry_variant"] != CANONICAL_GEOMETRY_VARIANT:
        raise ValueError("canonical geometry variant differs")
    if raw["project"] != PROJECT or raw["geometry_source"] != GEOMETRY_SOURCE:
        raise ValueError("configuration names the wrong project or geometry authority")
    if (
        raw["rule_count_active"] != 9
        or raw["rule_dim_total"] != authority.RULE_DIM
        or raw["unused_rule_index"] != authority.UNUSED_RULE_INDEX
    ):
        raise ValueError("rule dimensions differ from the authority")
    if raw["train_speed_mps"] != dict(authority.TRAIN_SPEED_MPS):
        raise ValueError("train_speed_mps differs from the authority")
    if raw["train_speed_scalar"] != dict(authority.TRAIN_SPEED_SCALAR):
        raise ValueError("train_speed_scalar differs from the authority")
    if raw["target_long_medium_steps"] != authority.TARGET_LONG_MEDIUM_STEPS:
        raise ValueError("target_long_medium_steps differs from the authority")
    if timing != {
        "dt_seconds": authority.DT_S,
        "stable_steps": authority.STABLE_STEPS,
        "delay_steps": [25, 50, 75],
        "hold_steps": authority.FINAL_HOLD_STEPS,
        "prepare_steps": authority.PREPARE_STEPS,
    }:
        raise ValueError("canonical timing differs from the frozen contract")
    return GeometryConfig(
        project=PROJECT,
        geometry_source=GEOMETRY_SOURCE,
        target_long_medium_steps=authority.TARGET_LONG_MEDIUM_STEPS,
        dt_seconds=authority.DT_S,
        stable_steps=authority.STABLE_STEPS,
        delay_steps=(25, 50, 75),
        hold_steps=authority.FINAL_HOLD_STEPS,
        prepare_steps=authority.PREPARE_STEPS,
        stroke_jitter_fraction=0.0,
        stroke_jitter_copies=0,
        stroke_jitter_seed=42,
        validation_delay_steps=50,
        validation_seed=1042,
        validation_network_noise=False,
        timing_mode=CANONICAL_TIMING_MODE,
        geometry_variant=CANONICAL_GEOMETRY_VARIANT,
        canonical_speed_name="slow",
        canonical_simple_intervals=150,
        canonical_compound_intervals=200,
        canonical_move_intervals=150,
        canonical_transition_intervals=20,
        canonical_trim_ratio=0.1,
        canonical_bezier_reference_samples=10001,
    )


def load_geometry_config(path: str | Path) -> GeometryConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    legacy_keys = {
        "project",
        "geometry_source",
        "rule_count_active",
        "rule_dim_total",
        "unused_rule_index",
        "target_long_medium_steps",
        "train_speed_mps",
        "train_speed_scalar",
        "timing",
        "stroke_start_sampling",
        "checkpoint_validation",
    }
    timing_mode = str(raw.get("timing_mode", LEGACY_TIMING_MODE))
    if timing_mode == CANONICAL_TIMING_MODE:
        return _load_canonical_geometry_config(raw)
    if timing_mode == LEGACY_TIMING_MODE:
        expected_keys = legacy_keys | ({"timing_mode"} if "timing_mode" in raw else set())
    elif timing_mode == FIXED_DURATION_TIMING_MODE:
        expected_keys = legacy_keys | {
            "timing_mode",
            "movement_intervals",
            "corner_dwell_intervals",
            "corner_dwell_rules",
        }
    else:
        raise ValueError(f"unsupported timing_mode: {timing_mode}")
    _require_keys(raw, expected_keys, "geometry configuration")
    timing = raw["timing"]
    sampling = raw["stroke_start_sampling"]
    validation = raw["checkpoint_validation"]
    if not all(isinstance(value, dict) for value in (timing, sampling, validation)):
        raise ValueError("timing, stroke_start_sampling, and checkpoint_validation must be objects")
    _require_keys(
        timing,
        {"dt_seconds", "stable_steps", "delay_steps", "hold_steps", "prepare_steps"},
        "timing",
    )
    _require_keys(
        sampling,
        {
            "coordinate_frame",
            "base_starts",
            "cross_each_primitive_with_all_rule_starts",
            "jitter",
        },
        "stroke_start_sampling",
    )
    jitter = sampling["jitter"]
    if not isinstance(jitter, dict):
        raise ValueError("stroke_start_sampling.jitter must be an object")
    _require_keys(
        jitter,
        {
            "distribution",
            "fraction_of_global_character_span",
            "copies_per_base_start",
            "seed",
        },
        "stroke_start_sampling.jitter",
    )
    _require_keys(
        validation,
        {
            "strokes",
            "stroke_starts",
            "moves",
            "speeds",
            "delay_steps",
            "include_jitter",
            "include_complete_characters",
            "validation_seed",
            "network_noise",
            "aggregation",
        },
        "checkpoint_validation",
    )

    fixed_intervals: tuple[tuple[str, int], ...] = ()
    corner_dwell_intervals = 0
    corner_dwell_rules: tuple[str, ...] = ()
    if timing_mode == FIXED_DURATION_TIMING_MODE:
        if raw["movement_intervals"] != {"fast": 50, "medium": 100, "slow": 150}:
            raise ValueError("fixed-duration movement_intervals must be 50/100/150")
        if raw["corner_dwell_intervals"] != 5:
            raise ValueError("fixed-duration corner_dwell_intervals must be 5")
        if raw["corner_dwell_rules"] != ["hengzhe", "shugou"]:
            raise ValueError("fixed-duration corner_dwell_rules must be hengzhe/shugou")
        fixed_intervals = tuple(
            (speed_name, int(raw["movement_intervals"][speed_name]))
            for speed_name in SPEED_NAMES
        )
        corner_dwell_intervals = 5
        corner_dwell_rules = ("hengzhe", "shugou")

    config = GeometryConfig(
        project=str(raw["project"]),
        geometry_source=str(raw["geometry_source"]),
        target_long_medium_steps=int(raw["target_long_medium_steps"]),
        dt_seconds=float(timing["dt_seconds"]),
        stable_steps=int(timing["stable_steps"]),
        delay_steps=tuple(int(value) for value in timing["delay_steps"]),
        hold_steps=int(timing["hold_steps"]),
        prepare_steps=int(timing["prepare_steps"]),
        stroke_jitter_fraction=float(jitter["fraction_of_global_character_span"]),
        stroke_jitter_copies=int(jitter["copies_per_base_start"]),
        stroke_jitter_seed=int(jitter["seed"]),
        validation_delay_steps=int(validation["delay_steps"]),
        validation_seed=int(validation["validation_seed"]),
        validation_network_noise=bool(validation["network_noise"]),
        timing_mode=timing_mode,
        movement_intervals=fixed_intervals,
        corner_dwell_intervals=corner_dwell_intervals,
        corner_dwell_rules=corner_dwell_rules,
    )
    if config.project != PROJECT or config.geometry_source != GEOMETRY_SOURCE:
        raise ValueError("configuration names the wrong project or geometry authority")
    if (
        raw["rule_count_active"] != 9
        or raw["rule_dim_total"] != authority.RULE_DIM
        or raw["unused_rule_index"] != authority.UNUSED_RULE_INDEX
    ):
        raise ValueError("rule dimensions differ from the authority")
    if raw["train_speed_mps"] != dict(authority.TRAIN_SPEED_MPS):
        raise ValueError("train_speed_mps differs from the authority")
    if raw["train_speed_scalar"] != dict(authority.TRAIN_SPEED_SCALAR):
        raise ValueError("train_speed_scalar differs from the authority")
    if config.target_long_medium_steps != authority.TARGET_LONG_MEDIUM_STEPS:
        raise ValueError("target_long_medium_steps differs from the authority")
    if not math.isclose(config.dt_seconds, authority.DT_S, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("dt_seconds differs from the authority")
    if (
        config.stable_steps != authority.STABLE_STEPS
        or config.hold_steps != authority.FINAL_HOLD_STEPS
        or config.prepare_steps != authority.PREPARE_STEPS
        or config.delay_steps != (25, 50, 75)
    ):
        raise ValueError("timing differs from the accepted protocol")
    if sampling["coordinate_frame"] != "character_local_then_add_motornet_anchor":
        raise ValueError("unsupported stroke-start coordinate frame")
    if sampling["base_starts"] != "motornet_center_plus_all_unique_actual_rule_starts":
        raise ValueError("unsupported stroke base-start set")
    if sampling["cross_each_primitive_with_all_rule_starts"] is not True:
        raise ValueError("each primitive must cross all starts for its rule")
    if jitter["distribution"] != "uniform_per_axis":
        raise ValueError("stroke jitter must be uniform per axis")
    if (
        config.stroke_jitter_fraction != 0.03
        or config.stroke_jitter_copies != 4
        or config.stroke_jitter_seed != 42
    ):
        raise ValueError("stroke jitter differs from the accepted protocol")
    if (
        authority.MOVE_JITTER_FRACTION != 0.03
        or authority.MOVE_JITTER_COPIES != 4
        or authority.RANDOM_SEED != 42
    ):
        raise ValueError("move jitter differs from the geometry authority")
    if validation != {
        "strokes": "all_15_primitives",
        "stroke_starts": "center_and_all_exact_rule_starts",
        "moves": "all_12_exact_transitions",
        "speeds": ["fast", "medium", "slow"],
        "delay_steps": 50,
        "include_jitter": False,
        "include_complete_characters": False,
        "validation_seed": 1042,
        "network_noise": True,
        "aggregation": "equal_mean_over_9_rules",
    }:
        raise ValueError("checkpoint validation grid differs from the accepted protocol")
    return config


def _characters(config: GeometryConfig) -> dict[str, authority.Character]:
    characters, _ = authority.physical_characters(config.target_long_medium_steps)
    return characters


def _unique_points(points: list[np.ndarray]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for point in points:
        array = np.asarray(point, dtype=np.float64)
        if not any(np.allclose(array, existing, rtol=0.0, atol=1e-12) for existing in output):
            output.append(array.copy())
    return output


def exact_rule_starts(config: GeometryConfig) -> dict[str, tuple[np.ndarray, ...]]:
    occurrences = authority.primitive_occurrences(_characters(config))
    starts: dict[str, list[np.ndarray]] = {rule: [np.zeros(2)] for rule in ACTIVE_RULES[:-1]}
    for occurrence in occurrences:
        starts[str(occurrence["rule"])].append(
            np.asarray(occurrence["start_xy_m"], dtype=np.float64)
        )
    return {rule: tuple(_unique_points(values)) for rule, values in starts.items()}


def _global_character_span(config: GeometryConfig) -> np.ndarray:
    characters = _characters(config)
    points = np.concatenate(
        [stroke.points for character in characters.values() for stroke in character.strokes]
    )
    return np.ptp(points, axis=0)


def stroke_conditions(
    config: GeometryConfig, *, include_jitter: bool
) -> tuple[StrokeCondition, ...]:
    if config.timing_mode == CANONICAL_TIMING_MODE:
        if include_jitter:
            raise ValueError("canonical stroke conditions forbid jitter")
        from hanzi_writing.canonical_protocol import canonical_stroke_conditions

        return canonical_stroke_conditions(config)
    occurrences = authority.primitive_occurrences(_characters(config))
    starts = exact_rule_starts(config)
    rng = np.random.default_rng(config.stroke_jitter_seed)
    jitter_scale = _global_character_span(config) * config.stroke_jitter_fraction
    output: list[StrokeCondition] = []
    for occurrence_index, occurrence in enumerate(occurrences):
        rule = str(occurrence["rule"])
        primitive_id = f"primitive_{occurrence_index:02d}_{rule}"
        for start_index, exact_start in enumerate(starts[rule]):
            placements = [("exact", exact_start)]
            if include_jitter:
                placements.extend(
                    (
                        f"jitter_{copy_index}",
                        exact_start + rng.uniform(-jitter_scale, jitter_scale),
                    )
                    for copy_index in range(config.stroke_jitter_copies)
                )
            for variant, start in placements:
                output.append(
                    StrokeCondition(
                        condition_id=(
                            f"{primitive_id}_start_{start_index:02d}_{variant}"
                        ),
                        rule=rule,
                        primitive_id=primitive_id,
                        character=str(occurrence["character"]),
                        stroke_index=int(occurrence["stroke_index"]),
                        variant=variant,
                        start_xy_m=np.asarray(start, dtype=np.float64),
                        relative_points_m=np.asarray(
                            occurrence["relative_points_m"], dtype=np.float64
                        ),
                    )
                )
    return tuple(output)


def move_conditions(
    config: GeometryConfig, *, include_jitter: bool
) -> tuple[MoveCondition, ...]:
    if config.timing_mode == CANONICAL_TIMING_MODE and include_jitter:
        raise ValueError("canonical move conditions forbid jitter")
    records = authority.move_training_conditions(_characters(config))
    output = []
    for record in records:
        if not include_jitter and record["variant"] != "exact":
            continue
        output.append(
            MoveCondition(
                condition_id=(
                    f"move_{record['character']}_{int(record['transition_index']):02d}_"
                    f"{record['variant']}"
                ),
                character=str(record["character"]),
                transition_index=int(record["transition_index"]),
                variant=str(record["variant"]),
                start_xy_m=np.asarray(record["start_xy_m"], dtype=np.float64),
                goal_xy_m=np.asarray(record["goal_xy_m"], dtype=np.float64),
            )
        )
    return tuple(output)


def training_conditions_by_rule(
    config: GeometryConfig,
) -> dict[str, tuple[StrokeCondition | MoveCondition, ...]]:
    grouped: dict[str, list[StrokeCondition | MoveCondition]] = {
        rule: [] for rule in ACTIVE_RULES
    }
    include_jitter = config.timing_mode != CANONICAL_TIMING_MODE
    for condition in stroke_conditions(config, include_jitter=include_jitter):
        grouped[condition.rule].append(condition)
    grouped["move"].extend(move_conditions(config, include_jitter=include_jitter))
    if any(not values for values in grouped.values()):
        raise RuntimeError("every active rule must have training conditions")
    return {rule: tuple(values) for rule, values in grouped.items()}


def checkpoint_stroke_groups(
    config: GeometryConfig,
) -> tuple[tuple[StrokeCondition, ...], ...]:
    grouped: dict[str, list[StrokeCondition]] = {}
    for condition in stroke_conditions(config, include_jitter=False):
        grouped.setdefault(condition.primitive_id, []).append(condition)
    return tuple(tuple(grouped[key]) for key in sorted(grouped))


def build_component_trajectory(
    condition: StrokeCondition | MoveCondition,
    speed_name: str,
    cue_normalizer_m: float,
    geometry_config: GeometryConfig | None = None,
) -> ComponentTrajectory:
    if speed_name not in authority.TRAIN_SPEED_MPS:
        raise KeyError(f"unknown speed: {speed_name}")
    timing_mode = (
        LEGACY_TIMING_MODE if geometry_config is None else geometry_config.timing_mode
    )
    speed = authority.TRAIN_SPEED_MPS[speed_name]
    if isinstance(condition, StrokeCondition):
        dense = condition.relative_points_m + condition.start_xy_m
        rule = condition.rule
        cue = np.zeros(2, dtype=np.float64)
    else:
        dense = authority.straight_line(condition.start_xy_m, condition.goal_xy_m)
        rule = "move"
        cue = condition.goal_xy_m / cue_normalizer_m
    dense_length = authority.arc_length(dense)
    corner_dwell_intervals = 0
    dwell_entry_index = None
    dwell_exit_index = None
    corner_xy_m = None
    movement_subphase = None
    canonical_rounding = None
    if timing_mode == LEGACY_TIMING_MODE:
        base_intervals = authority.intervals_for_length(dense_length, speed)
        points = authority.resample_by_arclength(dense, base_intervals)
    elif timing_mode == FIXED_DURATION_TIMING_MODE:
        if geometry_config is None:
            raise RuntimeError("fixed-duration trajectory requires a geometry config")
        base_intervals = geometry_config.intervals_for_speed(speed_name)
        if rule in geometry_config.corner_dwell_rules:
            fixed = authority.fixed_duration_compound_trajectory(
                dense,
                base_intervals=base_intervals,
                corner_dwell_intervals=geometry_config.corner_dwell_intervals,
            )
            points = fixed["points"]
            corner_dwell_intervals = geometry_config.corner_dwell_intervals
            dwell_entry_index = int(fixed["dwell_entry_index"])
            dwell_exit_index = int(fixed["dwell_exit_index"])
            corner_xy_m = np.asarray(fixed["corner_xy_m"], dtype=np.float64)
            movement_subphase = tuple(str(value) for value in fixed["movement_subphase"])
        else:
            points = authority.resample_by_arclength(dense, base_intervals)
    elif timing_mode == CANONICAL_TIMING_MODE:
        if geometry_config is None:
            raise RuntimeError("canonical trajectory requires a geometry config")
        if speed_name != geometry_config.canonical_speed_name:
            raise ValueError("canonical trajectories require the frozen slow cue")
        from hanzi_writing.canonical_protocol import canonical_target_trajectory

        canonical = canonical_target_trajectory(condition, geometry_config)
        points = np.asarray(canonical["points_m"], dtype=np.float64)
        base_intervals = len(points) - 1
        movement_subphase = tuple(str(value) for value in canonical["subphase"])
        canonical_rounding = canonical.get("rounding")
    else:
        raise RuntimeError(f"unhandled timing mode: {timing_mode}")
    intervals = len(points) - 1
    target_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    base_duration = base_intervals * authority.DT_S
    total_duration = intervals * authority.DT_S
    spatial_speed = target_length / base_duration
    total_speed = target_length / total_duration
    return ComponentTrajectory(
        rule=rule,
        condition_id=condition.condition_id,
        speed_name=speed_name,
        speed_mps=speed if timing_mode == LEGACY_TIMING_MODE else spatial_speed,
        speed_scalar=authority.TRAIN_SPEED_SCALAR[speed_name],
        movement_intervals=intervals,
        points_m=points,
        spatial_cue=cue,
        base_movement_intervals=base_intervals,
        target_path_length_m=target_length,
        base_movement_duration_s=base_duration,
        total_movement_duration_s=total_duration,
        target_mean_spatial_path_speed_mps=spatial_speed,
        target_mean_total_path_speed_mps=total_speed,
        max_target_step_distance_m=float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).max()
        ),
        corner_dwell_intervals=corner_dwell_intervals,
        dwell_entry_index=dwell_entry_index,
        dwell_exit_index=dwell_exit_index,
        corner_xy_m=corner_xy_m,
        movement_subphase=movement_subphase,
        canonical_rounding=canonical_rounding,
    )


def cue_scale(config: GeometryConfig) -> float:
    return authority.cue_scale_m(_characters(config))


def characters(config: GeometryConfig) -> dict[str, authority.Character]:
    return _characters(config)
