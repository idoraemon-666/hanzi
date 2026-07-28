"""Server-only pre-training audit for the Hanzi protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from digit_writing.geometry_audit import FK_TOLERANCE_M, _kinematic_metrics
from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.envs import HanziCharacterEnv, HanziComponentEnv
from hanzi_writing.geometry import (
    PROJECT,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    characters,
    checkpoint_stroke_groups,
    load_geometry_config,
    move_conditions,
    stroke_conditions,
    training_conditions_by_rule,
)
from hanzi_writing.motornet_support import baseline_anchor
from hanzi_writing.training import (
    _assert_state_equal,
    _character_rollout,
    _hp_from_config,
    _l1_rows,
    _make_effector,
    _rollout,
    _segment_metrics,
    _state_clone,
    _write_json,
    validate_training_config,
)
from losses import l1_muscle_act, l1_rate, l1_weight, position_l1_metrics, simple_dynamics
from train import _build_policy


def _condition_points(condition: StrokeCondition | MoveCondition) -> np.ndarray:
    if isinstance(condition, StrokeCondition):
        return condition.relative_points_m + condition.start_xy_m
    return authority.straight_line(condition.start_xy_m, condition.goal_xy_m)


def _workspace_audit(geometry: GeometryConfig) -> dict[str, Any]:
    effector = _make_effector()
    skeleton = effector.skeleton
    anchor = baseline_anchor(effector)
    lower = np.asarray(effector.pos_lower_bound, dtype=np.float64)
    upper = np.asarray(effector.pos_upper_bound, dtype=np.float64)
    categorized: list[tuple[str, str, np.ndarray]] = []
    for name, character in characters(geometry).items():
        for stroke_index, stroke in enumerate(character.strokes):
            categorized.append(
                ("character", f"{name}_stroke_{stroke_index:02d}_{stroke.label}", stroke.points)
            )
    for condition in stroke_conditions(geometry, include_jitter=True):
        categorized.append(("isolated_stroke", condition.condition_id, _condition_points(condition)))
    for condition in move_conditions(geometry, include_jitter=True):
        categorized.append(("move", condition.condition_id, _condition_points(condition)))

    rows = []
    all_world_points = []
    for category, condition_id, local_points in categorized:
        world_points = local_points + anchor
        metrics, _ = _kinematic_metrics(
            world_points,
            skeleton,
            float(skeleton.L1),
            float(skeleton.L2),
            lower,
            upper,
        )
        rows.append({"category": category, "condition_id": condition_id, **metrics})
        all_world_points.append(world_points)
    combined = np.concatenate(all_world_points)
    passed = all(
        row["all_inverse_solutions_reachable"]
        and row["minimum_inner_radial_margin_m"] > 0.0
        and row["minimum_outer_radial_margin_m"] > 0.0
        and row["minimum_joint_margin_rad"] > 0.0
        and row["maximum_motor_fk_error_m"] <= FK_TOLERANCE_M
        for row in rows
    )
    most_dangerous = min(
        rows,
        key=lambda row: min(
            row["minimum_inner_radial_margin_m"],
            row["minimum_outer_radial_margin_m"],
            row["minimum_joint_margin_rad"],
        ),
    )
    occurrences = authority.primitive_occurrences(characters(geometry))
    longest_heng = max(
        (row for row in occurrences if row["rule"] == "heng"),
        key=lambda row: row["length_m"],
    )
    deepest_shu = min(
        (row for row in occurrences if row["rule"] == "shu"),
        key=lambda row: row["end_xy_m"][1],
    )
    longest_move = max(
        move_conditions(geometry, include_jitter=False),
        key=lambda row: float(np.linalg.norm(row.goal_xy_m - row.start_xy_m)),
    )
    return {
        "device": "cpu",
        "anchor_m": anchor.tolist(),
        "first_stroke_centered_at_anchor": bool(
            all(np.allclose(character.strokes[0].points[0], 0.0) for character in characters(geometry).values())
        ),
        "character_stroke_count": sum(len(character.strokes) for character in characters(geometry).values()),
        "isolated_stroke_placement_count": len(stroke_conditions(geometry, include_jitter=True)),
        "move_condition_count": len(move_conditions(geometry, include_jitter=True)),
        "cartesian_bbox_m": {
            "min_x": float(combined[:, 0].min()),
            "max_x": float(combined[:, 0].max()),
            "min_y": float(combined[:, 1].min()),
            "max_y": float(combined[:, 1].max()),
        },
        "minimum_inner_radial_margin_m": min(row["minimum_inner_radial_margin_m"] for row in rows),
        "minimum_outer_radial_margin_m": min(row["minimum_outer_radial_margin_m"] for row in rows),
        "minimum_joint_margin_rad": min(row["minimum_joint_margin_rad"] for row in rows),
        "maximum_motor_fk_error_m": max(row["maximum_motor_fk_error_m"] for row in rows),
        "fk_tolerance_m": FK_TOLERANCE_M,
        "most_dangerous_condition": most_dangerous,
        "named_risk_geometry": {
            "longest_heng": {
                "character": longest_heng["character"],
                "stroke_index": longest_heng["stroke_index"],
                "length_m": longest_heng["length_m"],
            },
            "deepest_shu": {
                "character": deepest_shu["character"],
                "stroke_index": deepest_shu["stroke_index"],
                "end_xy_m": deepest_shu["end_xy_m"],
            },
            "longest_exact_move": {
                "condition_id": longest_move.condition_id,
                "distance_m": float(
                    np.linalg.norm(longest_move.goal_xy_m - longest_move.start_xy_m)
                ),
            },
        },
        "conditions": rows,
        "passed": bool(passed),
        "failure_policy": "report_only_no_automatic_rescaling",
    }


def _select_smoke_conditions(geometry: GeometryConfig) -> list[tuple[str, StrokeCondition | MoveCondition, str]]:
    exact_strokes = [
        condition
        for condition in stroke_conditions(geometry, include_jitter=False)
        if condition.variant == "exact" and np.allclose(condition.start_xy_m, 0.0)
    ]

    def longest(rule: str) -> StrokeCondition:
        matches = [condition for condition in exact_strokes if condition.rule == rule]
        return max(matches, key=lambda condition: authority.arc_length(condition.relative_points_m))

    def first(rule: str) -> StrokeCondition:
        return next(condition for condition in exact_strokes if condition.rule == rule)

    exact_moves = move_conditions(geometry, include_jitter=False)
    all_moves = move_conditions(geometry, include_jitter=True)
    longest_move = max(
        exact_moves,
        key=lambda condition: float(np.linalg.norm(condition.goal_xy_m - condition.start_xy_m)),
    )
    jitter_move = next(condition for condition in all_moves if condition.variant != "exact")
    return [
        ("heng_longest_fast", longest("heng"), "fast"),
        ("heng_longest_slow", longest("heng"), "slow"),
        ("shu_longest_fast", longest("shu"), "fast"),
        ("shu_longest_slow", longest("shu"), "slow"),
        ("na_medium", first("na"), "medium"),
        ("hengzhe_medium", first("hengzhe"), "medium"),
        ("shugou_medium", first("shugou"), "medium"),
        ("move_longest_exact_medium", longest_move, "medium"),
        ("move_one_jitter_medium", jitter_move, "medium"),
    ]


def _component_smoke(
    policy, hp: dict[str, Any], geometry: GeometryConfig, geometry_path: str
) -> dict[str, Any]:
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=geometry_path,
        action_frame_stacking=0,
    )
    rows = []
    for label, condition, speed_name in _select_smoke_conditions(geometry):
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            speed_name,
            geometry.validation_delay_steps,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=False,
        )
        position = position_l1_metrics(
            result["xy"], result["target"], result["epoch_bounds"]
        )
        components = {
            "position": float(position["phase_normalized_position_l1"]),
            "rate": float(l1_rate(result["hidden"], hp["l1_rate"])),
            "weight": float(l1_weight(policy, hp["l1_weight"])),
            "muscle": float(l1_muscle_act(result["muscle"], hp["l1_muscle_act"])),
            "simple_dynamics": float(
                simple_dynamics(
                    result["hidden"], policy.mrnn, weight=hp["simple_dynamics_weight"]
                )
            ),
        }
        finite = bool(
            np.isfinite(list(components.values())).all()
            and torch.isfinite(result["xy"]).all()
            and torch.isfinite(result["target"]).all()
            and torch.isfinite(result["hidden"]).all()
            and torch.isfinite(result["muscle"]).all()
        )
        rows.append(
            {
                "label": label,
                "condition_id": condition.condition_id,
                "rule": "move" if isinstance(condition, MoveCondition) else condition.rule,
                "speed": speed_name,
                "timesteps": result["timesteps"],
                "loss_components": components,
                "total_loss": float(sum(components.values())),
                "all_values_finite": finite,
                "episode_terminated_exactly": result["timesteps"] == env.max_ep_duration + 1,
            }
        )
    return {
        "cases": rows,
        "passed": all(row["all_values_finite"] and row["episode_terminated_exactly"] for row in rows),
    }


def _character_smoke(policy, hp: dict[str, Any], geometry_path: str) -> dict[str, Any]:
    before = _state_clone(policy)
    rows = {}
    for name in ("mu", "jiang", "ke"):
        env = HanziCharacterEnv(
            effector=_make_effector(),
            geometry_config_path=geometry_path,
            action_frame_stacking=0,
        )
        result = _character_rollout(policy, env, hp, name, "medium")
        error = _l1_rows(result["actual"], result["target"])
        move_rows = _segment_metrics(result, "move_movement_")
        move_mask = np.asarray(
            [phase.startswith("move_movement_") for phase in result["phase"]],
            dtype=bool,
        )
        hold_start = len(error) - authority.FINAL_HOLD_STEPS
        hold_drift = np.abs(
            result["actual"][hold_start:] - result["actual"][hold_start - 1]
        ).sum(axis=1)
        rows[name] = {
            "timesteps": len(error),
            "episode_terminated_exactly": result["episode_terminated_exactly"],
            "all_values_finite": bool(
                np.isfinite(result["actual"]).all()
                and np.isfinite(result["target"]).all()
                and np.isfinite(error).all()
            ),
            "writing_only_mean_l1_m": float(error[result["writing_mask"]].mean()),
            "stroke_metrics": _segment_metrics(result, "stroke_movement_"),
            "move_metrics": move_rows,
            "move_only_mean_l1_m": float(error[move_mask].mean()),
            "final_hold_target_mean_l1_m": float(error[hold_start:].mean()),
            "final_hold_drift_mean_l1_m": float(hold_drift.mean()),
            "writing_mask_count": int(result["writing_mask"].sum()),
            "segment_count": len(result["segments"]),
        }
    _assert_state_equal(before, policy)
    return {
        "characters": rows,
        "network_state_dict_bitwise_unchanged": True,
        "optimizer_created": False,
        "external_trainable_coefficients": False,
        "motornet_optimizer_created": False,
        "motornet_parameter_updates": 0,
        "passed": all(
            row["all_values_finite"]
            and row["episode_terminated_exactly"]
            and 800 <= row["timesteps"] <= 1000
            for row in rows.values()
        ),
    }


def _condition_tables(geometry: GeometryConfig) -> dict[str, Any]:
    moves = move_conditions(geometry, include_jitter=True)
    strokes = stroke_conditions(geometry, include_jitter=True)
    return {
        "stroke_conditions": [
            {
                "condition_id": condition.condition_id,
                "primitive_id": condition.primitive_id,
                "rule": condition.rule,
                "source_character_metadata_only": condition.character,
                "source_stroke_index": condition.stroke_index,
                "variant": condition.variant,
                "start_xy_m": condition.start_xy_m.tolist(),
                "length_m": authority.arc_length(condition.relative_points_m),
            }
            for condition in strokes
        ],
        "move_conditions": [
            {
                "condition_id": condition.condition_id,
                "source_character_metadata_only": condition.character,
                "transition_index": condition.transition_index,
                "variant": condition.variant,
                "start_xy_m": condition.start_xy_m.tolist(),
                "goal_xy_m": condition.goal_xy_m.tolist(),
            }
            for condition in moves
        ],
    }


def run_audit(
    geometry_path: str | Path,
    training_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    geometry = load_geometry_config(geometry_path)
    with Path(training_config_path).open("r", encoding="utf-8") as handle:
        training_config = json.load(handle)
    validate_training_config(training_config)
    hp = _hp_from_config(training_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    authority_artifacts = authority.write_audit_artifacts(output / "authority")
    self_test = authority.run_self_test()
    condition_tables = _condition_tables(geometry)
    _write_json(output / "condition_tables.json", condition_tables)
    grouped = training_conditions_by_rule(geometry)
    sampler = {
        "active_rules": list(grouped),
        "condition_count_by_rule": {rule: len(values) for rule, values in grouped.items()},
        "uniform_rule_probability": 1.0 / len(grouped),
        "contains_complete_character_sequence": False,
        "character_name_enters_observation": False,
        "checkpoint_stroke_group_count": len(checkpoint_stroke_groups(geometry)),
        "checkpoint_exact_move_count": len(move_conditions(geometry, include_jitter=False)),
        "checkpoint_rollout_group_count_three_speeds": (
            len(checkpoint_stroke_groups(geometry))
            + len(move_conditions(geometry, include_jitter=False))
        )
        * 3,
    }
    workspace = _workspace_audit(geometry)

    torch.manual_seed(training_config["seed"])
    np.random.seed(training_config["seed"])
    policy = _build_policy(hp, 6, torch.device("cpu"))
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    component_smoke = _component_smoke(policy, hp, geometry, str(geometry_path))
    character_smoke = _character_smoke(policy, hp, str(geometry_path))
    overall = bool(
        self_test["overall_passed"]
        and workspace["passed"]
        and component_smoke["passed"]
        and character_smoke["passed"]
        and sampler["checkpoint_stroke_group_count"] == 15
        and sampler["checkpoint_exact_move_count"] == 12
        and sampler["checkpoint_rollout_group_count_three_speeds"] == 81
    )
    report = {
        "project": PROJECT,
        "device": "cpu",
        "formal_training_started": False,
        "authority_self_test": self_test,
        "authority_artifacts": authority_artifacts,
        "sampler_and_checkpoint_grid": sampler,
        "workspace_audit": workspace,
        "component_closed_loop_smoke": component_smoke,
        "frozen_character_smoke": character_smoke,
        "overall_passed": overall,
    }
    _write_json(output / "audit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_audit(args.geometry_config, args.training_config, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
