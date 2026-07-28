#!/usr/bin/env python3
"""Final controlled geometry and trial schedule for the Hanzi stroke-composition project.

This file is the single authority for:
- the eight writing-stroke geometries;
- the three validation characters (木, 江, 可);
- physical scaling and the three nominal path speeds;
- isolated-stroke variants and placement coverage;
- the generic pen-up move transitions;
- frozen full-character temporal-composition schedules;
- geometry/time/input self-tests and audit artifacts.

The RNN, MotorNet, optimizer, regularizers, feedback channels, and total 28-D
observation contract are intentionally outside this file and must remain those
of the current project unless the accompanying Codex instruction explicitly
says otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

DT_S = 0.01
TRAIN_SPEED_MPS: Mapping[str, float] = {
    "fast": 0.5,
    "medium": 0.25,
    "slow": 1.0 / 6.0,
}
TRAIN_SPEED_SCALAR: Mapping[str, float] = {
    "fast": 2.0 / 3.0,
    "medium": 1.0 / 3.0,
    "slow": 0.0,
}

# Existing 10 rule columns are preserved. Nine are used; the final column stays zero.
RULE_INDEX: Mapping[str, int] = {
    "heng": 0,
    "shu": 1,
    "pie": 2,
    "na": 3,
    "dian": 4,
    "ti": 5,
    "hengzhe": 6,
    "shugou": 7,
    "move": 8,
}
RULE_DIM = 10
UNUSED_RULE_INDEX = 9

STABLE_STEPS = 25
PREPARE_STEPS = 25
FINAL_HOLD_STEPS = 25
TARGET_LONG_MEDIUM_STEPS = 85
MOVE_JITTER_FRACTION = 0.03
MOVE_JITTER_COPIES = 4
RANDOM_SEED = 42


@dataclass(frozen=True)
class Stroke:
    label: str
    points: np.ndarray


@dataclass(frozen=True)
class Character:
    name: str
    strokes: Tuple[Stroke, ...]


def _line(start: Sequence[float], end: Sequence[float], n: int = 101) -> np.ndarray:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    return (1.0 - t) * start_array + t * end_array


def straight_line(
    start: Sequence[float], end: Sequence[float], n: int = 101
) -> np.ndarray:
    """Return the authority's standard dense straight-line path."""

    return _line(start, end, n)


def _quadratic_bezier(
    start: Sequence[float],
    control: Sequence[float],
    end: Sequence[float],
    n: int = 121,
) -> np.ndarray:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(control, dtype=np.float64)
    p2 = np.asarray(end, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    return (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t**2 * p2


def _cubic_bezier(
    start: Sequence[float],
    control_1: Sequence[float],
    control_2: Sequence[float],
    end: Sequence[float],
    n: int = 121,
) -> np.ndarray:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(control_1, dtype=np.float64)
    p2 = np.asarray(control_2, dtype=np.float64)
    p3 = np.asarray(end, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    return (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t**2 * p2
        + t**3 * p3
    )


def _polyline(points: Sequence[Sequence[float]], n_per_segment: int = 81) -> np.ndarray:
    output: List[np.ndarray] = []
    for index in range(len(points) - 1):
        segment = _line(points[index], points[index + 1], n_per_segment)
        if index:
            segment = segment[1:]
        output.append(segment)
    return np.concatenate(output, axis=0)


def _endpoint(start: Sequence[float], length: float, direction_degrees: float) -> np.ndarray:
    angle = math.radians(direction_degrees)
    return np.asarray(
        [
            float(start[0]) + length * math.cos(angle),
            float(start[1]) + length * math.sin(angle),
        ],
        dtype=np.float64,
    )


def arc_length(points: np.ndarray) -> float:
    differences = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    return float(np.linalg.norm(differences, axis=1).sum())


def resample_by_arclength(points: np.ndarray, intervals: int) -> np.ndarray:
    if intervals < 1:
        raise ValueError("intervals must be at least one")
    points = np.asarray(points, dtype=np.float64)
    differences = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0.0:
        raise ValueError("cannot resample a zero-length trajectory")
    targets = np.linspace(0.0, cumulative[-1], intervals + 1)
    output = np.empty((intervals + 1, 2), dtype=np.float64)
    for dimension in range(2):
        output[:, dimension] = np.interp(targets, cumulative, points[:, dimension])
    output[0] = points[0]
    output[-1] = points[-1]
    return output


def intervals_for_length(length_m: float, speed_mps: float, dt_s: float = DT_S) -> int:
    if length_m <= 0.0 or speed_mps <= 0.0 or dt_s <= 0.0:
        raise ValueError("length, speed, and dt must all be positive")
    return int(math.ceil(length_m / (speed_mps * dt_s)))


def one_hot_rule(label: str) -> np.ndarray:
    output = np.zeros(RULE_DIM, dtype=np.float64)
    output[RULE_INDEX[label]] = 1.0
    return output


def design_characters() -> Dict[str, Character]:
    """Return the user-approved design-space geometry.

    Positive x is rightward and positive y is upward. Each complete character is
    translated afterward so its first stroke begins at (0, 0); this translation
    does not alter any relative geometry.
    """

    # 木: only the final approved na curve is used here.
    mu = Character(
        "mu",
        (
            Stroke("heng", _line((0.0, 0.0), (538.0, 0.0))),
            Stroke("shu", _line((225.0, 290.0), (225.0, -565.0))),
            Stroke("pie", _quadratic_bezier((205.0, -15.0), (20.0, -280.0), (-85.0, -410.0))),
            Stroke(
                "na",
                _cubic_bezier(
                    (285.0, -70.0),
                    (295.0, -160.0),
                    (390.0, -350.0),
                    (670.0, -390.0),
                ),
            ),
        ),
    )

    jiang = Character(
        "jiang",
        (
            Stroke("dian", _line((0.0, 0.0), (97.0, -104.0))),
            Stroke("dian", _line((-80.0, -213.0), (4.0, -301.0))),
            Stroke("ti", _line((-46.0, -711.0), (120.0, -353.0))),
            Stroke("heng", _line((215.0, -189.0), (596.0, -189.0))),
            Stroke("shu", _line((348.0, -213.0), (348.0, -532.0))),
            Stroke("heng", _line((84.0, -579.0), (722.0, -579.0))),
        ),
    )

    hengzhe_start = np.asarray((175.0, -150.0), dtype=np.float64)
    hengzhe_corner = np.asarray((315.0, -150.0), dtype=np.float64)
    # Left-down straight fold. Its supporting line makes an 80-degree acute angle with horizontal.
    hengzhe_end = _endpoint(hengzhe_corner, 145.0, 260.0)

    shugou_start = np.asarray((500.0, -30.0), dtype=np.float64)
    shugou_corner = np.asarray((500.0, -620.0), dtype=np.float64)
    # Left-up straight hook, 60 degrees from vertical.
    shugou_end = _endpoint(shugou_corner, 90.0, 150.0)

    ke = Character(
        "ke",
        (
            Stroke("heng", _line((20.0, 0.0), (784.0, 0.0))),
            Stroke("shu", _line((120.0, -145.0), (120.0, -365.0))),
            Stroke("hengzhe", _polyline((hengzhe_start, hengzhe_corner, hengzhe_end))),
            Stroke("heng", _line((150.0, -325.0), (375.0, -325.0))),
            Stroke("shugou", _polyline((shugou_start, shugou_corner, shugou_end))),
        ),
    )

    return {character.name: normalize_character_origin(character) for character in (mu, jiang, ke)}


def normalize_character_origin(character: Character) -> Character:
    origin = character.strokes[0].points[0].copy()
    return Character(
        character.name,
        tuple(Stroke(stroke.label, stroke.points - origin) for stroke in character.strokes),
    )


def design_scale_m_per_unit(
    characters: Mapping[str, Character],
    target_long_medium_steps: int = TARGET_LONG_MEDIUM_STEPS,
) -> float:
    longest_design_length = max(
        arc_length(stroke.points)
        for character in characters.values()
        for stroke in character.strokes
    )
    target_length_m = TRAIN_SPEED_MPS["medium"] * DT_S * target_long_medium_steps
    return target_length_m / longest_design_length


def physical_characters(
    target_long_medium_steps: int = TARGET_LONG_MEDIUM_STEPS,
) -> Tuple[Dict[str, Character], float]:
    characters = design_characters()
    scale = design_scale_m_per_unit(characters, target_long_medium_steps)
    physical = {
        name: Character(
            name,
            tuple(Stroke(stroke.label, stroke.points * scale) for stroke in character.strokes),
        )
        for name, character in characters.items()
    }
    return physical, scale


def primitive_occurrences(characters: Mapping[str, Character]) -> List[dict]:
    """Return every required writing occurrence as a reusable relative primitive.

    The same rule label is retained across every duration and placement. Therefore
    different lengths are conditions of one task, not new one-hot tasks.
    """

    occurrences: List[dict] = []
    for character_name, character in characters.items():
        for stroke_index, stroke in enumerate(character.strokes):
            relative = stroke.points - stroke.points[0]
            occurrences.append(
                {
                    "character": character_name,
                    "stroke_index": stroke_index,
                    "rule": stroke.label,
                    "start_xy_m": stroke.points[0].tolist(),
                    "end_xy_m": stroke.points[-1].tolist(),
                    "relative_points_m": relative.tolist(),
                    "length_m": arc_length(relative),
                }
            )
    return occurrences


def penup_transitions(characters: Mapping[str, Character]) -> List[dict]:
    transitions: List[dict] = []
    for character_name, character in characters.items():
        for stroke_index in range(len(character.strokes) - 1):
            source = character.strokes[stroke_index]
            target = character.strokes[stroke_index + 1]
            start = source.points[-1]
            goal = target.points[0]
            transitions.append(
                {
                    "character": character_name,
                    "transition_index": stroke_index,
                    "from_rule": source.label,
                    "to_rule": target.label,
                    "start_xy_m": start.tolist(),
                    "goal_xy_m": goal.tolist(),
                    "delta_xy_m": (goal - start).tolist(),
                    "distance_m": float(np.linalg.norm(goal - start)),
                }
            )
    return transitions


def move_training_conditions(
    characters: Mapping[str, Character],
    jitter_fraction: float = MOVE_JITTER_FRACTION,
    jitter_copies: int = MOVE_JITTER_COPIES,
    seed: int = RANDOM_SEED,
) -> List[dict]:
    """Cover every actual transition plus small deterministic start/goal jitter.

    The character identity is metadata only and must never enter the network input.
    """

    base = penup_transitions(characters)
    rng = np.random.default_rng(seed)
    all_points = np.concatenate(
        [stroke.points for character in characters.values() for stroke in character.strokes], axis=0
    )
    geometry_span = np.ptp(all_points, axis=0)
    jitter_scale = np.maximum(geometry_span * jitter_fraction, 1e-6)
    output: List[dict] = []
    for transition in base:
        output.append({**transition, "variant": "exact"})
        start = np.asarray(transition["start_xy_m"], dtype=np.float64)
        goal = np.asarray(transition["goal_xy_m"], dtype=np.float64)
        for copy_index in range(jitter_copies):
            jittered_start = start + rng.uniform(-jitter_scale, jitter_scale)
            jittered_goal = goal + rng.uniform(-jitter_scale, jitter_scale)
            output.append(
                {
                    **transition,
                    "variant": f"jitter_{copy_index}",
                    "start_xy_m": jittered_start.tolist(),
                    "goal_xy_m": jittered_goal.tolist(),
                    "delta_xy_m": (jittered_goal - jittered_start).tolist(),
                    "distance_m": float(np.linalg.norm(jittered_goal - jittered_start)),
                }
            )
    return output


def cue_scale_m(characters: Mapping[str, Character]) -> float:
    points = np.concatenate(
        [stroke.points for character in characters.values() for stroke in character.strokes], axis=0
    )
    maximum = float(np.abs(points).max())
    if maximum <= 0.0:
        raise ValueError("invalid zero cue scale")
    return 1.05 * maximum


def assemble_character_schedule(
    character: Character,
    speed_name: str = "medium",
    stable_steps: int = STABLE_STEPS,
    prepare_steps: int = PREPARE_STEPS,
    final_hold_steps: int = FINAL_HOLD_STEPS,
    cue_normalizer_m: float | None = None,
) -> dict:
    """Build a frozen full-character temporal-composition schedule.

    A brief preparation period is inserted before every move and every subsequent
    writing stroke. This preserves the original delayed-go logic and gives the
    frozen recurrent state time to reconfigure after each rule switch.
    """

    if speed_name not in TRAIN_SPEED_MPS:
        raise KeyError(f"unknown speed: {speed_name}")
    speed = TRAIN_SPEED_MPS[speed_name]
    speed_scalar = TRAIN_SPEED_SCALAR[speed_name]
    if cue_normalizer_m is None:
        cue_normalizer_m = max(
            float(np.abs(stroke.points).max()) for stroke in character.strokes
        ) * 1.05

    target_rows: List[np.ndarray] = []
    rule_rows: List[np.ndarray] = []
    speed_rows: List[float] = []
    go_rows: List[float] = []
    cue_rows: List[np.ndarray] = []
    writing_rows: List[bool] = []
    phase_rows: List[str] = []
    segment_records: List[dict] = []

    def append_constant(
        point: np.ndarray,
        count: int,
        rule: str,
        go: float,
        cue: np.ndarray,
        writing: bool,
        phase: str,
        active_speed: bool,
    ) -> None:
        start_index = len(target_rows)
        for _ in range(count):
            target_rows.append(point.copy())
            rule_rows.append(one_hot_rule(rule))
            speed_rows.append(speed_scalar if active_speed else 0.0)
            go_rows.append(go)
            cue_rows.append(cue.copy())
            writing_rows.append(writing)
            phase_rows.append(phase)
        segment_records.append(
            {
                "phase": phase,
                "rule": rule,
                "start_index": start_index,
                "end_index_exclusive": len(target_rows),
                "writing": writing,
            }
        )

    def append_movement(
        points: np.ndarray,
        rule: str,
        cue: np.ndarray,
        writing: bool,
        phase: str,
    ) -> None:
        start_index = len(target_rows)
        for point in points:
            target_rows.append(point.copy())
            rule_rows.append(one_hot_rule(rule))
            speed_rows.append(speed_scalar)
            go_rows.append(1.0)
            cue_rows.append(cue.copy())
            writing_rows.append(writing)
            phase_rows.append(phase)
        segment_records.append(
            {
                "phase": phase,
                "rule": rule,
                "start_index": start_index,
                "end_index_exclusive": len(target_rows),
                "writing": writing,
            }
        )

    first_stroke = character.strokes[0]
    append_constant(
        first_stroke.points[0],
        stable_steps,
        first_stroke.label,
        0.0,
        np.zeros(2),
        False,
        "initial_stable",
        False,
    )
    append_constant(
        first_stroke.points[0],
        prepare_steps,
        first_stroke.label,
        0.0,
        np.zeros(2),
        False,
        "stroke_prepare_0",
        True,
    )

    for stroke_index, stroke in enumerate(character.strokes):
        stroke_intervals = intervals_for_length(arc_length(stroke.points), speed)
        sampled_stroke = resample_by_arclength(stroke.points, stroke_intervals)
        append_movement(
            sampled_stroke,
            stroke.label,
            np.zeros(2),
            True,
            f"stroke_movement_{stroke_index}",
        )

        if stroke_index == len(character.strokes) - 1:
            append_constant(
                stroke.points[-1],
                final_hold_steps,
                stroke.label,
                0.0,
                np.zeros(2),
                False,
                "final_hold",
                True,
            )
            continue

        next_stroke = character.strokes[stroke_index + 1]
        goal = next_stroke.points[0]
        goal_cue = goal / cue_normalizer_m
        append_constant(
            stroke.points[-1],
            prepare_steps,
            "move",
            0.0,
            goal_cue,
            False,
            f"move_prepare_{stroke_index}",
            True,
        )
        move_points = _line(stroke.points[-1], goal)
        move_intervals = intervals_for_length(arc_length(move_points), speed)
        sampled_move = resample_by_arclength(move_points, move_intervals)
        append_movement(
            sampled_move,
            "move",
            goal_cue,
            False,
            f"move_movement_{stroke_index}",
        )
        append_constant(
            goal,
            prepare_steps,
            next_stroke.label,
            0.0,
            np.zeros(2),
            False,
            f"stroke_prepare_{stroke_index + 1}",
            True,
        )

    return {
        "character": character.name,
        "speed_name": speed_name,
        "speed_mps": speed,
        "cue_scale_m": cue_normalizer_m,
        "target_xy_m": np.asarray(target_rows, dtype=np.float64),
        "rule_input": np.asarray(rule_rows, dtype=np.float64),
        "speed_scalar": np.asarray(speed_rows, dtype=np.float64)[:, None],
        "go_cue": np.asarray(go_rows, dtype=np.float64)[:, None],
        "spatial_goal_cue": np.asarray(cue_rows, dtype=np.float64),
        "writing_mask": np.asarray(writing_rows, dtype=bool),
        "phase": phase_rows,
        "segments": segment_records,
    }


def _angle_between_undirected(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    cosine = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine))


def run_self_test(target_long_medium_steps: int = TARGET_LONG_MEDIUM_STEPS) -> dict:
    characters, scale = physical_characters(target_long_medium_steps)
    design = design_characters()
    failures: List[str] = []

    for name, character in characters.items():
        if not np.allclose(character.strokes[0].points[0], 0.0):
            failures.append(f"{name}: first stroke does not begin at the origin")
        for stroke_index, stroke in enumerate(character.strokes):
            if stroke.label not in RULE_INDEX or stroke.label == "move":
                failures.append(f"{name}:{stroke_index}: invalid writing rule {stroke.label}")
            if not np.isfinite(stroke.points).all():
                failures.append(f"{name}:{stroke_index}: non-finite point")
            if arc_length(stroke.points) <= 0.0:
                failures.append(f"{name}:{stroke_index}: zero length")
            if stroke.label == "heng" and not np.allclose(stroke.points[:, 1], stroke.points[0, 1]):
                failures.append(f"{name}:{stroke_index}: heng is not horizontal")
            if stroke.label == "shu" and not np.allclose(stroke.points[:, 0], stroke.points[0, 0]):
                failures.append(f"{name}:{stroke_index}: shu is not vertical")

    ke = design["ke"]
    hengzhe = ke.strokes[2].points
    corner_index = int(np.argmin(np.linalg.norm(hengzhe - np.asarray((295.0, -150.0)), axis=1)))
    # After whole-character origin normalization, the original x coordinates are shifted by -20.
    corner = hengzhe[corner_index]
    first_vector = corner - hengzhe[0]
    second_vector = hengzhe[-1] - corner
    fold_angle = _angle_between_undirected(first_vector, second_vector)
    if not (second_vector[0] < 0.0 and second_vector[1] < 0.0):
        failures.append("ke hengzhe fold is not left-down")
    if abs(fold_angle - 80.0) > 0.2:
        failures.append(f"ke hengzhe angle is {fold_angle:.6f}, not 80 degrees")

    shugou = ke.strokes[4].points
    hook_corner_index = int(np.argmin(shugou[:, 1]))
    hook_vector = shugou[-1] - shugou[hook_corner_index]
    hook_angle_to_vertical = _angle_between_undirected(hook_vector, np.asarray((0.0, 1.0)))
    if not (hook_vector[0] < 0.0 and hook_vector[1] > 0.0):
        failures.append("ke shugou hook is not left-up")
    if abs(hook_angle_to_vertical - 60.0) > 0.2:
        failures.append(
            f"ke shugou angle is {hook_angle_to_vertical:.6f}, not 60 degrees from vertical"
        )
    if not (shugou[0, 1] < ke.strokes[0].points[0, 1]):
        failures.append("ke shugou does not start below the top horizontal")

    longest_medium_steps = max(
        intervals_for_length(arc_length(stroke.points), TRAIN_SPEED_MPS["medium"])
        for character in characters.values()
        for stroke in character.strokes
    )
    if longest_medium_steps != target_long_medium_steps:
        failures.append(
            f"longest medium duration is {longest_medium_steps}, expected {target_long_medium_steps}"
        )

    transitions = penup_transitions(characters)
    if len(transitions) != 12:
        failures.append(f"expected 12 pen-up transitions, got {len(transitions)}")

    cue_normalizer = cue_scale_m(characters)
    schedule_summary: Dict[str, dict] = {}
    for name, character in characters.items():
        schedule = assemble_character_schedule(
            character,
            speed_name="medium",
            cue_normalizer_m=cue_normalizer,
        )
        expected_length = len(schedule["target_xy_m"])
        aligned_lengths = {
            key: len(schedule[key])
            for key in (
                "rule_input",
                "speed_scalar",
                "go_cue",
                "spatial_goal_cue",
                "writing_mask",
                "phase",
            )
        }
        if any(length != expected_length for length in aligned_lengths.values()):
            failures.append(f"{name}: schedule arrays have inconsistent lengths")
        if schedule["rule_input"].shape[1] != RULE_DIM:
            failures.append(f"{name}: rule input is not 10-dimensional")
        if not np.allclose(schedule["rule_input"][:, UNUSED_RULE_INDEX], 0.0):
            failures.append(f"{name}: unused rule column is nonzero")
        if np.abs(schedule["spatial_goal_cue"]).max() > 1.0 + 1e-9:
            failures.append(f"{name}: normalized goal cue exceeds [-1, 1]")
        writing_rules = np.argmax(schedule["rule_input"][schedule["writing_mask"]], axis=1)
        if np.any(writing_rules == RULE_INDEX["move"]):
            failures.append(f"{name}: move samples are marked as writing")
        move_mask = np.argmax(schedule["rule_input"], axis=1) == RULE_INDEX["move"]
        if np.any(schedule["writing_mask"] & move_mask):
            failures.append(f"{name}: move and writing masks overlap")
        movement_mask = np.asarray(
            ["movement" in phase for phase in schedule["phase"]], dtype=bool
        )
        hold_mask = np.asarray(schedule["phase"]) == "final_hold"
        prepare_or_stable_mask = ~(movement_mask | hold_mask)
        if not np.allclose(schedule["go_cue"][movement_mask], 1.0):
            failures.append(f"{name}: movement go cue is not one")
        if not np.allclose(schedule["go_cue"][hold_mask], 0.0):
            failures.append(f"{name}: final-hold go cue is not zero")
        if not np.allclose(schedule["go_cue"][prepare_or_stable_mask], 0.0):
            failures.append(f"{name}: prepare/stable go cue is not zero")
        schedule_summary[name] = {
            "timesteps": expected_length,
            "writing_timesteps": int(schedule["writing_mask"].sum()),
            "move_timesteps": int(move_mask.sum()),
            "segment_count": len(schedule["segments"]),
        }

    return {
        "overall_passed": not failures,
        "failures": failures,
        "target_long_medium_steps": target_long_medium_steps,
        "global_scale_m_per_design_unit": scale,
        "cue_scale_m": cue_normalizer,
        "rule_index": dict(RULE_INDEX),
        "unused_rule_index": UNUSED_RULE_INDEX,
        "penup_transition_count": len(transitions),
        "schedule_summary": schedule_summary,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_audit_artifacts(
    output_dir: Path,
    target_long_medium_steps: int = TARGET_LONG_MEDIUM_STEPS,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    characters, scale = physical_characters(target_long_medium_steps)
    report = run_self_test(target_long_medium_steps)
    if not report["overall_passed"]:
        raise RuntimeError("self-test failed: " + "; ".join(report["failures"]))

    occurrences = primitive_occurrences(characters)
    transition_rows = penup_transitions(characters)
    move_conditions = move_training_conditions(characters)
    cue_normalizer = cue_scale_m(characters)

    with (output_dir / "hanzi_geometry_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    manifest_rows: List[dict] = []
    for occurrence in occurrences:
        row = {
            key: value
            for key, value in occurrence.items()
            if key not in {"relative_points_m", "start_xy_m", "end_xy_m"}
        }
        row.update(
            {
                "start_x_m": occurrence["start_xy_m"][0],
                "start_y_m": occurrence["start_xy_m"][1],
                "end_x_m": occurrence["end_xy_m"][0],
                "end_y_m": occurrence["end_xy_m"][1],
                "fast_intervals": intervals_for_length(
                    occurrence["length_m"], TRAIN_SPEED_MPS["fast"]
                ),
                "medium_intervals": intervals_for_length(
                    occurrence["length_m"], TRAIN_SPEED_MPS["medium"]
                ),
                "slow_intervals": intervals_for_length(
                    occurrence["length_m"], TRAIN_SPEED_MPS["slow"]
                ),
            }
        )
        manifest_rows.append(row)
    _write_csv(output_dir / "stroke_manifest.csv", manifest_rows)

    transition_csv_rows: List[dict] = []
    for transition in transition_rows:
        transition_csv_rows.append(
            {
                "character": transition["character"],
                "transition_index": transition["transition_index"],
                "from_rule": transition["from_rule"],
                "to_rule": transition["to_rule"],
                "start_x_m": transition["start_xy_m"][0],
                "start_y_m": transition["start_xy_m"][1],
                "goal_x_m": transition["goal_xy_m"][0],
                "goal_y_m": transition["goal_xy_m"][1],
                "delta_x_m": transition["delta_xy_m"][0],
                "delta_y_m": transition["delta_xy_m"][1],
                "distance_m": transition["distance_m"],
                "medium_intervals": intervals_for_length(
                    transition["distance_m"], TRAIN_SPEED_MPS["medium"]
                ),
            }
        )
    _write_csv(output_dir / "penup_transitions.csv", transition_csv_rows)

    geometry_payload = {
        "dt_s": DT_S,
        "train_speed_mps": dict(TRAIN_SPEED_MPS),
        "train_speed_scalar": dict(TRAIN_SPEED_SCALAR),
        "target_long_medium_steps": target_long_medium_steps,
        "global_scale_m_per_design_unit": scale,
        "cue_scale_m": cue_normalizer,
        "rule_index": dict(RULE_INDEX),
        "unused_rule_index": UNUSED_RULE_INDEX,
        "characters": {
            name: {
                "strokes": [
                    {
                        "rule": stroke.label,
                        "points_m": stroke.points.tolist(),
                        "length_m": arc_length(stroke.points),
                    }
                    for stroke in character.strokes
                ]
            }
            for name, character in characters.items()
        },
        "primitive_occurrences": occurrences,
        "move_training_conditions": move_conditions,
    }
    with (output_dir / "hanzi_geometry_final.json").open("w", encoding="utf-8") as handle:
        json.dump(geometry_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    schedules = {
        name: assemble_character_schedule(
            character,
            speed_name="medium",
            cue_normalizer_m=cue_normalizer,
        )
        for name, character in characters.items()
    }
    schedule_payload = {}
    for name, schedule in schedules.items():
        schedule_payload[name] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in schedule.items()
        }
    with (output_dir / "character_schedules_medium.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # Plot only authoritative writing strokes. Pen-up moves are shown separately as dashed lines.
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (name, character) in zip(axes, characters.items()):
        for stroke_index, stroke in enumerate(character.strokes):
            axis.plot(stroke.points[:, 0], stroke.points[:, 1], linewidth=2.4)
            axis.scatter(stroke.points[0, 0], stroke.points[0, 1], s=25)
            axis.text(
                stroke.points[0, 0],
                stroke.points[0, 1],
                f"{stroke_index}:{stroke.label}",
                fontsize=8,
            )
            if stroke_index < len(character.strokes) - 1:
                next_start = character.strokes[stroke_index + 1].points[0]
                axis.plot(
                    [stroke.points[-1, 0], next_start[0]],
                    [stroke.points[-1, 1], next_start[1]],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.45,
                )
        axis.set_title(name)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.suptitle("Final Hanzi trajectories; dashed lines are pen-up moves")
    figure.tight_layout()
    figure.savefig(output_dir / "final_hanzi_trajectories_physical.png", dpi=220)
    plt.close(figure)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_hanzi_geometry"))
    parser.add_argument(
        "--target-long-medium-steps",
        type=int,
        default=TARGET_LONG_MEDIUM_STEPS,
    )
    arguments = parser.parse_args()

    if arguments.self_test:
        report = write_audit_artifacts(
            arguments.output_dir,
            target_long_medium_steps=arguments.target_long_medium_steps,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        parser.error("use --self-test to generate and validate the authoritative artifacts")


if __name__ == "__main__":
    main()
