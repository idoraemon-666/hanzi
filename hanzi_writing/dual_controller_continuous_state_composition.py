"""Compose three characters while carrying the full MotorNet physical state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.dual_controller_composition import (
    CHARACTERS,
    METRIC_FIELDS,
    PROJECT,
    _git_value,
    _load_frozen_policy,
    _load_json,
    _require_keys,
    _segment_row,
    _validate_source,
    _write_json,
    character_sequences,
)
from hanzi_writing.dual_rule_envs import DualFixedRuleEnv
from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    FIXED_SPEED_NAME,
    FixedRuleCondition,
    component_trajectory,
)
from hanzi_writing.dual_rule_training import (
    JOINT_GRADIENT_VARIANT,
    load_config as load_dual_config,
)
from hanzi_writing.training import _make_effector, _state_clone


VARIANT = "frozen_joint_gradient_onset_window_continuous_physical_state_v1"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "hanzi_stroke_temporal_composition_frozen_dual_controller_continuous_state_v1.json"
)
PHYSICAL_STATE_KEYS = ("joint", "cartesian", "muscle", "geometry")
EXPECTED_BOUNDARIES = 24


def load_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    config = _load_json(path)
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "enabled",
            "seed",
            "device",
            "joint_gradient_config",
            "source",
            "composition",
            "render",
            "output",
        },
        "continuous-state composition configuration",
    )
    if (
        config["project"],
        config["run_kind"],
        config["variant"],
        config["enabled"],
        config["seed"],
        config["device"],
    ) != (
        PROJECT,
        "frozen_dual_controller_continuous_physical_state_composition",
        VARIANT,
        True,
        42,
        "cpu",
    ):
        raise ValueError("continuous-state composition identity differs")
    if config["composition"] != {
        "characters": list(CHARACTERS),
        "controller_order": "stroke_then_alternating_move_stroke",
        "complete_trial_per_task": True,
        "character_initial_effector_state": "canonical_start_standard_reset",
        "intertrial_effector_operation": "none_shared_instance",
        "physical_state_carryover": list(PHYSICAL_STATE_KEYS),
        "physical_state_continuity_requirement": (
            "bitwise_equal_before_next_action"
        ),
        "fingertip_endpoint_continuity": True,
        "recurrent_state_reset_each_trial": True,
        "feedback_buffers": (
            "reinitialize_from_current_continuous_sensory_state"
        ),
        "translate_canonical_target": False,
        "network_noise": False,
        "deterministic_observation": True,
    }:
        raise ValueError("continuous physical-state boundary contract differs")
    if config["render"] != {
        "trajectory": "actual_complete_trial_including_physical_start_state",
        "stroke_linestyle": "solid",
        "move_linestyle": "dashed",
        "equal_aspect": True,
        "postprocessing": False,
    }:
        raise ValueError("continuous-state rendering contract differs")
    if config["output"] != {
        "directory": (
            "runs/hanzi_stroke_temporal_composition/"
            "frozen_dual_controller_continuous_physical_state/dev42"
        )
    }:
        raise ValueError("continuous-state output differs")
    dual_config, geometry = load_dual_config(config["joint_gradient_config"])
    if dual_config["variant"] != JOINT_GRADIENT_VARIANT:
        raise ValueError("continuous composition requires the joint-gradient config")
    return config, dual_config, geometry


class ContinuousStateDualFixedRuleEnv(DualFixedRuleEnv):
    """Configure the next trial without resetting the shared MotorNet effector."""

    def continue_from_current_state(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        del testing
        self._set_generator(seed=seed)
        supplied_options = {} if options is None else options
        supplied = supplied_options.get("conditions")
        if not isinstance(supplied, Sequence) or not supplied:
            raise ValueError("options.conditions must be a non-empty sequence")
        if not all(isinstance(item, FixedRuleCondition) for item in supplied):
            raise TypeError("dual conditions must contain FixedRuleCondition values")
        conditions = tuple(supplied)
        if len(conditions) != 1:
            raise ValueError("continuous composition requires batch size one")
        if any(condition.model_kind != self.model_kind for condition in conditions):
            raise ValueError("condition model kind differs from the environment")
        if supplied_options.get("speed_name") != FIXED_SPEED_NAME:
            raise ValueError("continuous composition requires the frozen slow cue")
        if int(supplied_options.get("delay_steps")) != FIXED_DELAY_STEPS:
            raise ValueError("continuous composition requires delay_steps=50")
        deterministic = bool(supplied_options.get("deterministic", False))
        trajectories = tuple(component_trajectory(condition) for condition in conditions)
        self._require_batch_compatible(trajectories)
        local = np.stack([trajectory.points_m for trajectory in trajectories])
        self.traj = torch.as_tensor(
            local + self.anchor_m,
            dtype=torch.float32,
            device=self.device,
        )
        self.conditions = conditions
        self.component_trajectories = trajectories
        self.delay_time = FIXED_DELAY_STEPS
        self.movement_intervals = trajectories[0].movement_intervals
        self.base_movement_intervals = trajectories[0].base_movement_intervals
        self.movement_subphase = trajectories[0].movement_subphase
        self._build_trial_inputs(conditions, FIXED_DELAY_STEPS)
        return self._initialize_observation_buffers(1, deterministic=deterministic)


def _clone_physical_state(effector: Any) -> dict[str, torch.Tensor]:
    states = effector.states
    if not isinstance(states, dict):
        raise RuntimeError("MotorNet effector states must be a dictionary")
    if not set(PHYSICAL_STATE_KEYS).issubset(states):
        raise RuntimeError("MotorNet physical state keys differ")
    output = {}
    for key in PHYSICAL_STATE_KEYS:
        value = states[key]
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise RuntimeError(f"MotorNet {key} state is not a finite tensor")
        output[key] = value.detach().clone()
    return output


def _compare_physical_states(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if tuple(expected) != PHYSICAL_STATE_KEYS or tuple(actual) != PHYSICAL_STATE_KEYS:
        raise RuntimeError("physical state audit keys differ")
    maximum_differences = {}
    for key in PHYSICAL_STATE_KEYS:
        if expected[key].shape != actual[key].shape:
            raise RuntimeError(f"physical state shape changed: {key}")
        maximum_differences[key] = float(
            torch.max(torch.abs(expected[key] - actual[key])).item()
        )
        if not torch.equal(expected[key], actual[key]):
            raise RuntimeError(f"physical state changed at trial switch: {key}")
    return {
        "state_keys": list(PHYSICAL_STATE_KEYS),
        "bitwise_equal": True,
        "maximum_absolute_difference": maximum_differences,
    }


def _make_shared_environments(
    dual_config: dict[str, Any],
) -> dict[str, ContinuousStateDualFixedRuleEnv]:
    effector = _make_effector()
    environments = {
        model_kind: ContinuousStateDualFixedRuleEnv(
            effector=effector,
            model_kind=model_kind,
            geometry_config_path=dual_config["geometry_config"],
            action_frame_stacking=0,
        )
        for model_kind in ("stroke", "move")
    }
    if environments["stroke"].effector is not environments["move"].effector:
        raise RuntimeError("Stroke and Move environments do not share one effector")
    if not np.array_equal(
        environments["stroke"].anchor_m, environments["move"].anchor_m
    ):
        raise RuntimeError("Stroke and Move MotorNet anchors differ")
    return environments


def _trial_options(condition: FixedRuleCondition) -> dict[str, Any]:
    return {
        "conditions": (condition,),
        "speed_name": FIXED_SPEED_NAME,
        "delay_steps": FIXED_DELAY_STEPS,
        "deterministic": True,
    }


def _rollout_trial(
    policy: Any,
    env: ContinuousStateDualFixedRuleEnv,
    hp: dict[str, Any],
    condition: FixedRuleCondition,
    *,
    initialize_character: bool,
    expected_start_state: dict[str, torch.Tensor] | None,
    expected_start_xy_m: np.ndarray,
) -> dict[str, Any]:
    x = torch.zeros((1, hp["hid_size"]), dtype=torch.float32)
    h = torch.zeros_like(x)
    options = _trial_options(condition)
    if initialize_character:
        if expected_start_state is not None:
            raise RuntimeError("character initialization cannot receive prior state")
        obs, info = env.reset(options=options)
        boundary_audit = None
    else:
        if expected_start_state is None:
            raise RuntimeError("continuous trial requires prior physical state")
        obs, info = env.continue_from_current_state(options=options)
        boundary_audit = _compare_physical_states(
            expected_start_state,
            _clone_physical_state(env.effector),
        )
    start_state = _clone_physical_state(env.effector)
    start_xy = (
        info["states"]["fingertip"].detach().cpu().numpy()[0]
        - env.anchor_m
    )
    endpoint_to_start_difference = float(
        np.linalg.norm(start_xy - expected_start_xy_m)
    )
    if not np.allclose(start_xy, expected_start_xy_m, rtol=0.0, atol=1e-6):
        raise RuntimeError("continuous physical start differs from expected endpoint")

    xy = []
    targets = []
    timestep = 0
    terminated = False
    with torch.no_grad():
        while not terminated:
            x, h, action = policy(obs, x, h, noise=False)
            obs, _, terminated, info = env.step(timestep, action=action)
            xy.append(info["states"]["fingertip"][:, None, :])
            targets.append(info["goal"][:, None, :])
            timestep += 1
    actual = torch.cat(xy, dim=1)[0].detach().cpu().numpy() - env.anchor_m
    target = torch.cat(targets, dim=1)[0].detach().cpu().numpy() - env.anchor_m
    stable_bounds = env.epoch_bounds["stable"]
    delay_bounds = env.epoch_bounds["delay"]
    movement_bounds = env.epoch_bounds["movement"]
    hold_bounds = env.epoch_bounds["hold"]
    if (
        stable_bounds[0] != 0
        or stable_bounds[1] != delay_bounds[0]
        or delay_bounds[1] != movement_bounds[0]
        or movement_bounds[1] != hold_bounds[0]
        or hold_bounds[1] != timestep
        or actual.shape[0] != timestep
        or target.shape[0] != timestep
    ):
        raise RuntimeError("continuous complete-trial coverage differs")
    movement_start, movement_end = movement_bounds
    movement_target = target[movement_start:movement_end]
    if movement_target.shape != condition.points_m.shape or not np.allclose(
        movement_target, condition.points_m, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("continuous composition target was translated or changed")
    return {
        "actual": actual,
        "actual_full_with_start": np.concatenate((start_xy[None, :], actual), axis=0),
        "target": target,
        "movement_actual": actual[movement_start:movement_end],
        "movement_target": movement_target,
        "reset_start": start_xy,
        "actual_endpoint": actual[-1].copy(),
        "epoch_bounds": dict(env.epoch_bounds),
        "timesteps": timestep,
        "physical_start_state": start_state,
        "physical_end_state": _clone_physical_state(env.effector),
        "boundary_audit": boundary_audit,
        "endpoint_to_start_difference_m": endpoint_to_start_difference,
    }


def _plot_character(path: Path, character: str, segments: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    labels = set()
    for segment in segments:
        model_kind = segment["condition"].model_kind
        style = "-" if model_kind == "stroke" else "--"
        color = "#1f4e79" if model_kind == "stroke" else "#888888"
        label = model_kind.capitalize() if model_kind not in labels else None
        labels.add(model_kind)
        points = segment["result"]["actual_full_with_start"]
        axis.plot(points[:, 0], points[:, 1], style, color=color, linewidth=2.4, label=label)
    axis.set_title(f"{character}: continuous physical-state composition")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    axis.legend()
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_overview(path: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, character in zip(axes, CHARACTERS):
        labels = set()
        for segment in results[character]:
            model_kind = segment["condition"].model_kind
            style = "-" if model_kind == "stroke" else "--"
            color = "#1f4e79" if model_kind == "stroke" else "#888888"
            label = model_kind.capitalize() if model_kind not in labels else None
            labels.add(model_kind)
            points = segment["result"]["actual_full_with_start"]
            axis.plot(points[:, 0], points[:, 1], style, color=color, linewidth=2.2, label=label)
        axis.set_title(character)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Frozen Stroke/Move composition with continuous physical state")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_composition(config_path: str | Path, source_results: str | Path) -> dict[str, Any]:
    config, dual_config, geometry = load_config(config_path)
    source_root = Path(source_results).resolve()
    manifests = _validate_source(config, dual_config, geometry, source_root)
    policies = {}
    hps = {}
    checkpoint_hashes = {}
    initial_policy_states = {}
    for model_kind in ("stroke", "move"):
        policy, hp, digest = _load_frozen_policy(
            config, dual_config, source_root, model_kind, manifests[model_kind]
        )
        policies[model_kind] = policy
        hps[model_kind] = hp
        checkpoint_hashes[model_kind] = digest
        initial_policy_states[model_kind] = _state_clone(policy)
    environments = _make_shared_environments(dual_config)

    output = Path(config["output"]["directory"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    plot_directory = output / "plots"
    plot_directory.mkdir()
    sequence_library = character_sequences(geometry)
    all_results: dict[str, list[dict[str, Any]]] = {}
    rows = []
    boundary_rows = []
    for character in CHARACTERS:
        character_results = []
        arrays: dict[str, np.ndarray] = {}
        previous_state = None
        previous_endpoint = sequence_library[character][0].points_m[0].copy()
        previous_condition = None
        for segment_index, condition in enumerate(sequence_library[character]):
            result = _rollout_trial(
                policies[condition.model_kind],
                environments[condition.model_kind],
                hps[condition.model_kind],
                condition,
                initialize_character=segment_index == 0,
                expected_start_state=previous_state,
                expected_start_xy_m=previous_endpoint,
            )
            prefix = f"segment_{segment_index:02d}_{condition.model_kind}_{condition.rule}"
            arrays[f"{prefix}__actual_full"] = result["actual"]
            arrays[f"{prefix}__actual_full_with_start"] = result[
                "actual_full_with_start"
            ]
            arrays[f"{prefix}__target_full"] = result["target"]
            arrays[f"{prefix}__actual_movement"] = result["movement_actual"]
            arrays[f"{prefix}__target_movement"] = result["movement_target"]
            for state_key in PHYSICAL_STATE_KEYS:
                arrays[f"{prefix}__physical_start_{state_key}"] = (
                    result["physical_start_state"][state_key].cpu().numpy()
                )
                arrays[f"{prefix}__physical_end_{state_key}"] = (
                    result["physical_end_state"][state_key].cpu().numpy()
                )
            if result["boundary_audit"] is not None:
                boundary_rows.append(
                    {
                        "character": character,
                        "boundary_index": segment_index - 1,
                        "from_model_kind": previous_condition.model_kind,
                        "from_rule": previous_condition.rule,
                        "to_model_kind": condition.model_kind,
                        "to_rule": condition.rule,
                        "endpoint_to_start_difference_m": result[
                            "endpoint_to_start_difference_m"
                        ],
                        **result["boundary_audit"],
                    }
                )
            rows.append(_segment_row(character, segment_index, condition, result))
            character_results.append({"condition": condition, "result": result})
            previous_state = result["physical_end_state"]
            previous_endpoint = result["actual_endpoint"].copy()
            previous_condition = condition
        np.savez_compressed(output / f"{character}_composition_trajectories.npz", **arrays)
        _plot_character(
            plot_directory / f"{character}_continuous_physical_state.png",
            character,
            character_results,
        )
        all_results[character] = character_results
    _plot_overview(
        plot_directory / "three_characters_continuous_physical_state.png",
        all_results,
    )

    if len(boundary_rows) != EXPECTED_BOUNDARIES:
        raise RuntimeError("continuous composition boundary count differs")
    if not all(row["bitwise_equal"] for row in boundary_rows):
        raise RuntimeError("continuous physical-state audit failed")
    with (output / "character_segment_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output / "physical_state_boundary_audit.json",
        {"physical_state_keys": list(PHYSICAL_STATE_KEYS), "boundaries": boundary_rows},
    )

    policy_unchanged = {
        model_kind: all(
            torch.equal(value, policies[model_kind].state_dict()[name])
            for name, value in initial_policy_states[model_kind].items()
        )
        for model_kind in ("stroke", "move")
    }
    if not all(policy_unchanged.values()):
        raise RuntimeError("frozen policy state changed during continuous composition")
    numeric_values = [
        float(row[field])
        for row in rows
        for field in METRIC_FIELDS[7:]
    ]
    if not np.isfinite(numeric_values).all():
        raise RuntimeError("continuous composition metrics contain non-finite values")
    maximum_state_difference = max(
        difference
        for row in boundary_rows
        for difference in row["maximum_absolute_difference"].values()
    )
    maximum_endpoint_difference = max(
        row["endpoint_to_start_difference_m"] for row in boundary_rows
    )
    summary = {
        "characters": list(CHARACTERS),
        "character_trial_counts": {
            character: len(all_results[character]) for character in CHARACTERS
        },
        "total_trials": len(rows),
        "stroke_trials": sum(row["model_kind"] == "stroke" for row in rows),
        "move_trials": sum(row["model_kind"] == "move" for row in rows),
        "character_initial_effector_resets": len(CHARACTERS),
        "intertrial_effector_resets": 0,
        "continuous_physical_state_boundaries": len(boundary_rows),
        "physical_state_bitwise_continuity": True,
        "maximum_physical_state_boundary_difference": maximum_state_difference,
        "maximum_endpoint_to_next_start_difference_m": maximum_endpoint_difference,
        "shared_effector_instance": True,
        "full_trial_render_verified": all(
            segment["result"]["actual_full_with_start"].shape[0]
            == segment["result"]["timesteps"] + 1
            for segments in all_results.values()
            for segment in segments
        ),
        "maximum_start_offset_euclidean_m": max(
            row["start_offset_euclidean_m"] for row in rows
        ),
        "maximum_trial_endpoint_euclidean_m": max(
            row["trial_endpoint_euclidean_m"] for row in rows
        ),
    }
    _write_json(output / "composition_summary.json", summary)
    report = [
        "# Frozen dual-controller continuous physical-state composition",
        "",
        "The selected joint-gradient Stroke/onset_window and Move/onset_window",
        "macro-best checkpoints were frozen and alternated over complete trials.",
        "Stroke and Move environments shared one MotorNet effector instance.",
        "No inter-trial effector reset or physical-state replacement occurred.",
        "Joint, Cartesian, muscle, and geometry states were bitwise continuous",
        "across all 24 within-character boundaries before the next action.",
        "Controller x/h and sensory buffers were reset at every trial switch.",
        "",
        "Complete Stroke trials are solid; complete Move trials are dashed.",
        "No trajectory postprocessing or training was performed.",
    ]
    (output / "FINAL_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )
    provenance = {
        "project": PROJECT,
        "variant": VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "source_results": str(source_root),
        "source_variant": config["source"]["variant"],
        "source_git_head": config["source"]["git_head"],
        "selected_checkpoint_sha256": checkpoint_hashes,
        "selected_loss_arms": {"stroke": "onset_window", "move": "onset_window"},
        "boundary_contract": config["composition"],
        "render_contract": config["render"],
        "policy_state_bitwise_unchanged": policy_unchanged,
        "training_started": False,
        "optimizer_created": False,
        "backward_executed": False,
        "trajectory_postprocessing_performed": False,
        "behavioral_pass_fail_defined": False,
        "integrity": {
            **summary,
            "physical_state_keys": list(PHYSICAL_STATE_KEYS),
            "plots": 4,
            "trajectory_archives": 3,
            "metric_rows": len(rows),
            "boundary_audit_rows": len(boundary_rows),
            "all_numeric_values_finite": True,
        },
        "completed": True,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--source-results", required=True)
    arguments = parser.parse_args()
    result = run_composition(arguments.config, arguments.source_results)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
