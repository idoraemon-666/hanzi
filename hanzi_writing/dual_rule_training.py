"""Train and audit the two fixed-rule RNNs under three matched loss arms."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.canonical_overfit import _atomic_torch_save, _movement_metrics
from hanzi_writing.dual_rule_envs import DualFixedRuleEnv
from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    FIXED_SPEED_NAME,
    FIXED_SPEED_SCALAR,
    LOSS_ARMS,
    MODEL_KINDS,
    VARIANT,
    FixedRuleCondition,
    condition_manifest,
    conditions,
    input_size,
    rule_names,
)
from hanzi_writing.geometry import (
    CANONICAL_GEOMETRY_VARIANT,
    CANONICAL_TIMING_MODE,
    PROJECT,
    GeometryConfig,
    load_geometry_config,
)
from hanzi_writing.training import (
    _environment_generator_states,
    _generator_state_equal,
    _gradient_state,
    _make_effector,
    _nested_equal,
    _restore_generator_state,
    _restore_gradients,
    _rollout,
    _sha256_file,
    _state_clone,
)
from losses import (
    detached_position_metrics,
    l1_muscle_act,
    l1_rate,
    l1_weight,
    onset_window_position_l1,
    position_l1_metrics,
    simple_dynamics,
)
from train import _build_policy, _checkpoint_payload, _fixed_rng


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "hanzi_stroke_temporal_composition_dual_fixed_rule_rnn_v1.json"
)
JOINT_GRADIENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "hanzi_stroke_temporal_composition_dual_fixed_rule_joint_gradient_v1.json"
)
JOINT_GRADIENT_VARIANT = "dual_fixed_rule_joint_gradient_loss_comparison_v1"
CHECKPOINT_PROFILES = (
    "best_macro_then_worst",
    "best_worst_then_macro",
)
CANDIDATE_NAMES = ("best_macro", "best_worst", "final")
METRIC_FIELDS = (
    "model_kind",
    "loss_arm",
    "candidate",
    "checkpoint_update",
    "rule",
    "condition_id",
    "source_character",
    "source_component_index",
    "movement_intervals",
    "movement_samples",
    "start_error_euclidean_m",
    "movement_mean_euclidean_m",
    "endpoint_euclidean_m",
    "target_path_length_m",
    "actual_path_length_m",
    "path_length_ratio",
    "max_euclidean_m",
    "pre_straight_mean_euclidean_m",
    "transition_mean_euclidean_m",
    "post_straight_mean_euclidean_m",
    "transition_max_euclidean_m",
    "exit_direction_error_deg",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("dual-RNN configuration must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def load_config(path: str | Path = CONFIG_PATH) -> tuple[dict[str, Any], GeometryConfig]:
    config = _load_json(path)
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "enabled",
            "seed",
            "validation_seed",
            "device",
            "geometry_config",
            "timing",
            "model_common",
            "models",
            "optimizer",
            "training",
            "loss_arms",
            "checkpoint_selection",
            "position_loss",
            "regularization",
            "output",
        },
        "dual-RNN configuration",
    )
    variant = config["variant"]
    if variant == VARIANT:
        expected_run_kind = "dual_fixed_rule_multitask_loss_comparison"
        expected_models = {
            "stroke": {
                "rule_dim": 15,
                "input_size": 33,
                "max_updates": 120000,
                "phase1_updates": 90000,
                "phase2_updates": 30000,
                "log_interval": 150,
            },
            "move": {
                "rule_dim": 12,
                "input_size": 30,
                "max_updates": 96000,
                "phase1_updates": 72000,
                "phase2_updates": 24000,
                "log_interval": 120,
            },
        }
        expected_training = {
            "batch_size": 1,
            "scheduler": "fixed_round_robin",
            "updates_per_rule": 8000,
            "validation_interval": 600,
            "network_noise": False,
            "deterministic_observation": True,
            "parallel_processes": 6,
        }
        expected_output = (
            "runs/hanzi_stroke_temporal_composition/"
            "dual_fixed_rule_loss_comparison/dev42"
        )
    elif variant == JOINT_GRADIENT_VARIANT:
        expected_run_kind = (
            "dual_fixed_rule_multitask_joint_gradient_loss_comparison"
        )
        expected_models = {
            "stroke": {
                "rule_dim": 15,
                "input_size": 33,
                "max_updates": 8000,
                "phase1_updates": 6000,
                "phase2_updates": 2000,
                "log_interval": 10,
                "validation_interval": 40,
            },
            "move": {
                "rule_dim": 12,
                "input_size": 30,
                "max_updates": 8000,
                "phase1_updates": 6000,
                "phase2_updates": 2000,
                "log_interval": 10,
                "validation_interval": 50,
            },
        }
        expected_training = {
            "microbatch_size_per_rule": 1,
            "optimizer_step_mode": "all_rules_mean_gradient",
            "scheduler": "all_rules_every_optimizer_step",
            "updates_per_rule": 8000,
            "effective_rules_per_optimizer_step": {"stroke": 15, "move": 12},
            "network_noise": False,
            "deterministic_observation": True,
            "parallel_processes": 6,
        }
        expected_output = (
            "runs/hanzi_stroke_temporal_composition/"
            "dual_fixed_rule_joint_gradient_loss_comparison/dev42"
        )
    else:
        raise ValueError("dual-RNN experiment variant differs")
    identity = (
        config["project"],
        config["run_kind"],
        config["enabled"],
        config["seed"],
        config["validation_seed"],
        config["device"],
    )
    if identity != (
        PROJECT,
        expected_run_kind,
        True,
        42,
        1042,
        "cpu",
    ):
        raise ValueError("dual-RNN experiment identity differs")
    if config["geometry_config"] != (
        "configurations/hanzi_stroke_temporal_composition_canonical_geometry.json"
    ):
        raise ValueError("dual-RNN geometry configuration differs")
    if config["timing"] != {
        "stable_steps": 25,
        "delay_steps": FIXED_DELAY_STEPS,
        "hold_steps": 25,
        "speed_name": FIXED_SPEED_NAME,
        "speed_scalar": FIXED_SPEED_SCALAR,
        "direction_augmentation": False,
    }:
        raise ValueError("dual-RNN fixed timing differs")
    if config["model_common"] != {
        "network": "rnn",
        "hidden_size": 256,
        "output_size": 6,
        "activation": "softplus",
        "recurrent_noise_std": 0.1,
        "input_noise_std": 0.01,
        "constrained": False,
        "rnn_dt_ms": 10,
        "rnn_tau_ms": 20,
        "batch_first": True,
    }:
        raise ValueError("dual-RNN model core differs")
    if config["models"] != expected_models:
        raise ValueError("dual-RNN model schedules differ")
    if config["optimizer"] != {
        "name": "Adam",
        "phase1_learning_rate": 0.001,
        "phase2_learning_rate": 0.0001,
        "grad_clip_norm": 1.0,
    }:
        raise ValueError("dual-RNN optimizer differs")
    if config["training"] != expected_training:
        raise ValueError("dual-RNN training contract differs")
    if config["loss_arms"] != {
        "baseline": "phase_normalized_l1",
        "full_trial": "full_trial_l1",
        "onset_window": "onset_window_phase_normalized_l1",
    }:
        raise ValueError("dual-RNN loss arms differ")
    if config["checkpoint_selection"] != {
        "common_validation_objective": "macro_mean_phase_normalized_position_l1",
        "profiles": list(CHECKPOINT_PROFILES),
        "behavioral_pass_fail_threshold": None,
    }:
        raise ValueError("dual-RNN checkpoint selection differs")
    if config["position_loss"] != {
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
        "onset_delay_last_steps": 10,
        "onset_movement_first_steps": 5,
    }:
        raise ValueError("dual-RNN position loss constants differ")
    if config["regularization"] != {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }:
        raise ValueError("dual-RNN regularization differs")
    if config["output"] != {"directory": expected_output}:
        raise ValueError("dual-RNN output contract differs")
    geometry = load_geometry_config(config["geometry_config"])
    if (
        geometry.timing_mode != CANONICAL_TIMING_MODE
        or geometry.geometry_variant != CANONICAL_GEOMETRY_VARIANT
        or geometry.validation_seed != config["validation_seed"]
    ):
        raise ValueError("dual-RNN authority geometry differs")
    for model_kind in MODEL_KINDS:
        model = config["models"][model_kind]
        count = len(rule_names(model_kind))
        common_invalid = (
            model["rule_dim"] != count
            or model["input_size"] != input_size(model_kind)
            or model["phase1_updates"] + model["phase2_updates"]
            != model["max_updates"]
        )
        if variant == VARIANT:
            schedule_invalid = (
                model["max_updates"]
                != count * config["training"]["updates_per_rule"]
                or model["phase1_updates"] % count
                or model["phase2_updates"] % count
                or config["training"]["validation_interval"] % count
                or model["log_interval"] % count
            )
        else:
            schedule_invalid = (
                model["max_updates"] != config["training"]["updates_per_rule"]
                or config["training"]["effective_rules_per_optimizer_step"][
                    model_kind
                ]
                != count
                or model["max_updates"] % model["validation_interval"]
                or model["max_updates"] % model["log_interval"]
            )
        if common_invalid or schedule_invalid:
            raise ValueError(f"dual-RNN cycle arithmetic differs for {model_kind}")
    return config, geometry


def _runtime_hp(config: dict[str, Any], model_kind: str) -> dict[str, Any]:
    common = config["model_common"]
    model = config["models"][model_kind]
    return {
        "network": common["network"],
        "inp_size": model["input_size"],
        "hid_size": common["hidden_size"],
        "activation_name": common["activation"],
        "noise_level_act": common["recurrent_noise_std"],
        "noise_level_inp": common["input_noise_std"],
        "constrained": common["constrained"],
        "dt": common["rnn_dt_ms"],
        "t_const": common["rnn_tau_ms"],
        "batch_first": common["batch_first"],
        "lr": config["optimizer"]["phase1_learning_rate"],
        "grad_clip_norm": config["optimizer"]["grad_clip_norm"],
        "batch_size": 1,
        "epochs": model["max_updates"],
        "save_iter": _validation_interval(config, model_kind),
        **config["regularization"],
        "seed": config["seed"],
        "validation_seed": config["validation_seed"],
        "variant": config["variant"],
        "project": PROJECT,
        "protocol_config": config,
    }


def _optimizer_step_mode(config: dict[str, Any]) -> str:
    if config["variant"] == VARIANT:
        return "single_rule_round_robin"
    return config["training"]["optimizer_step_mode"]


def _validation_interval(config: dict[str, Any], model_kind: str) -> int:
    if config["variant"] == VARIANT:
        return int(config["training"]["validation_interval"])
    return int(config["models"][model_kind]["validation_interval"])


def _updates_per_rule(config: dict[str, Any], model_kind: str) -> int:
    if config["variant"] == VARIANT:
        return int(config["models"][model_kind]["max_updates"]) // len(
            rule_names(model_kind)
        )
    return int(config["models"][model_kind]["max_updates"])


def scheduled_rule(model_kind: str, update: int) -> str:
    names = rule_names(model_kind)
    if update < 0:
        raise ValueError("dual-RNN update must be non-negative")
    return names[update % len(names)]


def _make_env(config: dict[str, Any], model_kind: str) -> DualFixedRuleEnv:
    return DualFixedRuleEnv(
        effector=_make_effector(),
        model_kind=model_kind,
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )


def _position_objective(
    arm: str,
    result: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    position = position_l1_metrics(
        result["xy"], result["target"], result["epoch_bounds"]
    )
    if arm == "baseline":
        objective = position["phase_normalized_position_l1"]
    elif arm == "full_trial":
        objective = position["full_trial_position_l1"]
    elif arm == "onset_window":
        onset = onset_window_position_l1(
            result["xy"], result["target"], result["epoch_bounds"]
        )
        objective = onset.pop("objective")
        position.update(onset)
    else:
        raise ValueError(f"unknown loss arm: {arm}")
    return objective, position


def _validation_rollouts(
    policy,
    env: DualFixedRuleEnv,
    hp: dict[str, Any],
    library: tuple[FixedRuleCondition, ...],
) -> dict[str, Any]:
    rows = []
    for condition in library:
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            FIXED_SPEED_NAME,
            FIXED_DELAY_STEPS,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=False,
        )
        metrics = detached_position_metrics(
            position_l1_metrics(
                result["xy"], result["target"], result["epoch_bounds"]
            )
        )
        rows.append(
            {
                "rule": condition.rule,
                "condition_id": condition.condition_id,
                **metrics,
            }
        )
    values = [float(row["phase_normalized_position_l1"]) for row in rows]
    return {
        "aggregation": "equal_mean_over_rules",
        "rule_count": len(rows),
        "macro_mean_phase_normalized_position_l1": float(np.mean(values)),
        "worst_rule_phase_normalized_position_l1": float(np.max(values)),
        "worst_rule": rows[int(np.argmax(values))]["rule"],
        "per_rule": rows,
    }


def readonly_validation(
    policy,
    optimizer,
    training_env: DualFixedRuleEnv,
    hp: dict[str, Any],
    config: dict[str, Any],
    model_kind: str,
    library: tuple[FixedRuleCondition, ...],
    checkpoint_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    policy_mode = bool(policy.training)
    policy_state = _state_clone(policy)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    gradients = _gradient_state(policy)
    python_rng = copy.deepcopy(random.getstate())
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    generator_states = _environment_generator_states(training_env)
    checkpoint_hashes = {
        str(Path(path)): _sha256_file(path) for path in checkpoint_paths
    }
    validation = None
    checks: dict[str, bool] = {}
    try:
        policy.eval()
        with _fixed_rng(config["validation_seed"]):
            validation_env = _make_env(config, model_kind)
            with torch.no_grad():
                validation = _validation_rollouts(
                    policy, validation_env, hp, library
                )
        checks = {
            "policy_state_bitwise_unchanged": all(
                torch.equal(value, policy.state_dict()[name])
                for name, value in policy_state.items()
            ),
            "optimizer_state_bitwise_unchanged": _nested_equal(
                optimizer_state, optimizer.state_dict()
            ),
            "gradient_state_bitwise_unchanged": _nested_equal(
                gradients, _gradient_state(policy)
            ),
            "python_rng_unchanged": _nested_equal(python_rng, random.getstate()),
            "numpy_rng_unchanged": _nested_equal(numpy_rng, np.random.get_state()),
            "torch_rng_unchanged": bool(torch.equal(torch_rng, torch.get_rng_state())),
            "environment_generators_unchanged": all(
                _generator_state_equal(generator, kind, state)
                for generator, kind, state, _ in generator_states
            ),
            "checkpoint_files_unchanged": checkpoint_hashes
            == {str(Path(path)): _sha256_file(path) for path in checkpoint_paths},
        }
    finally:
        policy.load_state_dict(policy_state)
        optimizer.load_state_dict(optimizer_state)
        _restore_gradients(policy, gradients)
        policy.train(policy_mode)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        for generator, kind, state, _ in generator_states:
            _restore_generator_state(generator, kind, state)
    checks["policy_training_mode_restored"] = policy.training == policy_mode
    if validation is None or not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("dual-RNN read-only validation failed: " + ",".join(failed))
    return {"validation": validation, "read_only_checks": checks}


def _checkpoint_identity(
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    variant: str = VARIANT,
) -> dict[str, Any]:
    return {
        "project": PROJECT,
        "variant": variant,
        "model_kind": model_kind,
        "loss_arm": arm,
        "rule_dim": len(rule_names(model_kind)),
        "input_size": input_size(model_kind),
        "rule_names": list(rule_names(model_kind)),
        "condition_manifest_sha256": manifest["condition_manifest_sha256"],
    }


def validate_checkpoint_identity(
    checkpoint: dict[str, Any],
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    variant: str = VARIANT,
) -> None:
    expected = _checkpoint_identity(model_kind, arm, manifest, variant)
    actual = {name: checkpoint.get(name) for name in expected}
    if actual != expected:
        raise ValueError("dual-RNN checkpoint identity differs")
    hp = checkpoint.get("hp")
    if not isinstance(hp, dict) or hp.get("inp_size") != input_size(model_kind):
        raise ValueError("dual-RNN checkpoint input size differs")


def _base_checkpoint(
    policy,
    optimizer,
    hp: dict[str, Any],
    update: int,
    validation: dict[str, Any],
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    checkpoint_kind: str,
) -> dict[str, Any]:
    macro = validation["macro_mean_phase_normalized_position_l1"]
    payload = _checkpoint_payload(policy, optimizer, hp, update, macro)
    payload.update(
        {
            **_checkpoint_identity(model_kind, arm, manifest, hp["variant"]),
            "checkpoint_kind": checkpoint_kind,
            "common_validation": validation,
        }
    )
    return payload


def _continuation_payload(
    policy,
    optimizer,
    hp: dict[str, Any],
    update: int,
    validation: dict[str, Any],
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    env: DualFixedRuleEnv,
    loss_tail: list[float],
    selection_state: dict[str, Any],
) -> dict[str, Any]:
    payload = _base_checkpoint(
        policy,
        optimizer,
        hp,
        update,
        validation,
        model_kind,
        arm,
        manifest,
        "dual_fixed_rule_continuation",
    )
    payload["continuation_state"] = {
        "next_update": update + 1,
        "loss_tail": loss_tail,
        "selection_state": selection_state,
        "environment_rng_states": [
            {"kind": kind, "state": state, "path": path}
            for _, kind, state, path in _environment_generator_states(env)
        ],
    }
    if not payload["continuation_state"]["environment_rng_states"]:
        raise RuntimeError("dual-RNN continuation environment RNG state is empty")
    return payload


def _restore_continuation(
    checkpoint: dict[str, Any],
    policy,
    optimizer,
    env: DualFixedRuleEnv,
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    variant: str,
) -> tuple[int, list[float], dict[str, Any]]:
    validate_checkpoint_identity(checkpoint, model_kind, arm, manifest, variant)
    if checkpoint.get("checkpoint_kind") != "dual_fixed_rule_continuation":
        raise ValueError("dual-RNN continuation checkpoint kind differs")
    policy.load_state_dict(checkpoint["agent_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict) or set(rng_state) != {
        "python",
        "numpy",
        "torch",
    }:
        raise ValueError("dual-RNN continuation global RNG state differs")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    state = checkpoint.get("continuation_state")
    if not isinstance(state, dict) or set(state) != {
        "next_update",
        "loss_tail",
        "selection_state",
        "environment_rng_states",
    }:
        raise ValueError("dual-RNN continuation state differs")
    available = {
        path: (generator, kind)
        for generator, kind, _, path in _environment_generator_states(env)
    }
    saved = state["environment_rng_states"]
    if {entry.get("path") for entry in saved} != set(available):
        raise ValueError("dual-RNN continuation environment RNG paths differ")
    for entry in saved:
        generator, kind = available[entry["path"]]
        if entry.get("kind") != kind:
            raise ValueError("dual-RNN continuation environment RNG kind differs")
        _restore_generator_state(generator, kind, entry["state"])
    next_update = state["next_update"]
    if next_update != checkpoint.get("update", -2) + 1:
        raise ValueError("dual-RNN continuation update differs")
    return (
        int(next_update),
        [float(value) for value in state["loss_tail"]],
        copy.deepcopy(state["selection_state"]),
    )


def _selection_key(profile: str, validation: dict[str, Any]) -> tuple[float, float]:
    macro = float(validation["macro_mean_phase_normalized_position_l1"])
    worst = float(validation["worst_rule_phase_normalized_position_l1"])
    if profile == "best_macro_then_worst":
        return macro, worst
    if profile == "best_worst_then_macro":
        return worst, macro
    raise ValueError(f"unknown checkpoint profile: {profile}")


def _save_selected_checkpoints(
    policy,
    optimizer,
    hp: dict[str, Any],
    update: int,
    validation: dict[str, Any],
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    output: Path,
    selection_state: dict[str, Any],
) -> None:
    filenames = {
        "best_macro_then_worst": "best_macro_checkpoint.pt",
        "best_worst_then_macro": "best_worst_checkpoint.pt",
    }
    for profile in CHECKPOINT_PROFILES:
        key = _selection_key(profile, validation)
        previous = tuple(selection_state[profile]["key"])
        if key < previous:
            payload = _base_checkpoint(
                policy,
                optimizer,
                hp,
                update,
                validation,
                model_kind,
                arm,
                manifest,
                "dual_fixed_rule_selected",
            )
            payload["selection_profile"] = profile
            payload["selection_key"] = list(key)
            _atomic_torch_save(payload, output / filenames[profile])
            selection_state[profile] = {"key": list(key), "update": update}


def _initial_selection_state() -> dict[str, Any]:
    return {
        profile: {"key": [float("inf"), float("inf")], "update": None}
        for profile in CHECKPOINT_PROFILES
    }


def _training_conditions(
    config: dict[str, Any],
    model_kind: str,
    update: int,
    library: tuple[FixedRuleCondition, ...],
    by_rule: dict[str, FixedRuleCondition],
) -> tuple[FixedRuleCondition, ...]:
    mode = _optimizer_step_mode(config)
    if mode == "single_rule_round_robin":
        return (by_rule[scheduled_rule(model_kind, update)],)
    if mode == "all_rules_mean_gradient":
        return library
    raise ValueError("dual-RNN optimizer step mode differs")


def _training_step(
    policy,
    optimizer,
    env: DualFixedRuleEnv,
    hp: dict[str, Any],
    arm: str,
    step_conditions: tuple[FixedRuleCondition, ...],
) -> dict[str, Any]:
    if not step_conditions:
        raise ValueError("dual-RNN optimizer step has no rules")
    optimizer.zero_grad()
    task_losses = []
    position_rows = []
    scale = 1.0 / len(step_conditions)
    for condition in step_conditions:
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            FIXED_SPEED_NAME,
            FIXED_DELAY_STEPS,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=True,
        )
        position_objective, position = _position_objective(arm, result)
        task_loss = position_objective
        task_loss = task_loss + l1_rate(result["hidden"], hp["l1_rate"])
        task_loss = task_loss + l1_weight(policy, hp["l1_weight"])
        task_loss = task_loss + l1_muscle_act(
            result["muscle"], hp["l1_muscle_act"]
        )
        task_loss = task_loss + simple_dynamics(
            result["hidden"], policy.mrnn, weight=hp["simple_dynamics_weight"]
        )
        (task_loss * scale).backward()
        task_losses.append(float(task_loss.detach().cpu()))
        position_rows.append(detached_position_metrics(position))
    torch.nn.utils.clip_grad_norm_(policy.parameters(), hp["grad_clip_norm"])
    optimizer.step()
    metric_names = set(position_rows[0])
    if any(set(row) != metric_names for row in position_rows[1:]):
        raise RuntimeError("dual-RNN per-rule training metrics differ")
    return {
        "mean_total_loss": float(np.mean(task_losses)),
        "mean_position_metrics": {
            name: float(np.mean([row[name] for row in position_rows]))
            for name in sorted(metric_names)
        },
    }


def train_worker(
    config_path: str | Path,
    model_kind: str,
    arm: str,
) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, geometry = load_config(config_path)
    if model_kind not in MODEL_KINDS or arm not in LOSS_ARMS:
        raise ValueError("dual-RNN worker identity differs")
    output_root = Path(config["output"]["directory"])
    manifest_path = output_root / f"{model_kind}_condition_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("dual-RNN prepared condition manifest is missing")
    manifest = _load_json(manifest_path)
    expected_manifest = condition_manifest(model_kind, geometry)
    if manifest != expected_manifest:
        raise RuntimeError("dual-RNN prepared condition manifest differs")
    output = output_root / model_kind / arm
    continuation_path = output / "continuation_checkpoint.pt"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    hp = _runtime_hp(config, model_kind)
    policy = _build_policy(
        hp, config["model_common"]["output_size"], torch.device("cpu")
    )
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config["optimizer"]["phase1_learning_rate"]
    )
    env = _make_env(config, model_kind)
    library = conditions(model_kind, geometry)
    by_rule = {condition.rule: condition for condition in library}
    if len(by_rule) != len(library):
        raise RuntimeError("each dual-RNN one-hot must map to exactly one condition")
    if output.exists():
        if not continuation_path.is_file():
            raise FileExistsError("worker output exists without a continuation checkpoint")
        first = library[0]
        env.reset(
            seed=config["seed"],
            options={
                "conditions": (first,),
                "speed_name": FIXED_SPEED_NAME,
                "delay_steps": FIXED_DELAY_STEPS,
                "deterministic": True,
            },
        )
        checkpoint = torch.load(
            continuation_path, map_location="cpu", weights_only=False
        )
        start_update, losses, selection_state = _restore_continuation(
            checkpoint,
            policy,
            optimizer,
            env,
            model_kind,
            arm,
            manifest,
            config["variant"],
        )
    else:
        output.mkdir(parents=False, exist_ok=False)
        start_update = 0
        losses = []
        selection_state = _initial_selection_state()
        first = library[0]
        env.reset(
            seed=config["seed"],
            options={
                "conditions": (first,),
                "speed_name": FIXED_SPEED_NAME,
                "delay_steps": FIXED_DELAY_STEPS,
                "deterministic": True,
            },
        )
        readonly = readonly_validation(
            policy,
            optimizer,
            env,
            hp,
            config,
            model_kind,
            library,
        )
        validation = readonly["validation"]
        _append_jsonl(
            output / "validation_metrics.jsonl",
            {
                "update": -1,
                "completed_updates": 0,
                "validation_kind": "initial",
                **readonly,
            },
        )
        _save_selected_checkpoints(
            policy,
            optimizer,
            hp,
            -1,
            validation,
            model_kind,
            arm,
            manifest,
            output,
            selection_state,
        )
        payload = _continuation_payload(
            policy,
            optimizer,
            hp,
            -1,
            validation,
            model_kind,
            arm,
            manifest,
            env,
            [],
            selection_state,
        )
        _atomic_torch_save(payload, continuation_path)
    schedule = config["models"][model_kind]
    max_updates = schedule["max_updates"]
    phase1_updates = schedule["phase1_updates"]
    log_interval = schedule["log_interval"]
    validation_interval = _validation_interval(config, model_kind)
    if start_update > max_updates:
        raise ValueError("dual-RNN continuation exceeds the configured updates")
    for update in range(start_update, max_updates):
        learning_rate = (
            config["optimizer"]["phase1_learning_rate"]
            if update < phase1_updates
            else config["optimizer"]["phase2_learning_rate"]
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        step_conditions = _training_conditions(
            config, model_kind, update, library, by_rule
        )
        step_result = _training_step(
            policy,
            optimizer,
            env,
            hp,
            arm,
            step_conditions,
        )
        losses.append(step_result["mean_total_loss"])
        completed = update + 1
        if completed % log_interval == 0:
            joint_gradient = len(step_conditions) > 1
            condition = step_conditions[-1]
            row = {
                "update": update,
                "completed_updates": completed,
                "model_kind": model_kind,
                "loss_arm": arm,
                "optimizer_step_mode": _optimizer_step_mode(config),
                "rules_per_optimizer_step": len(step_conditions),
                "rule": "all_rules" if joint_gradient else condition.rule,
                "condition_id": None if joint_gradient else condition.condition_id,
                "delay_steps": FIXED_DELAY_STEPS,
                "learning_rate": learning_rate,
                "mean_total_loss": float(np.mean(losses[-log_interval:])),
                **step_result["mean_position_metrics"],
            }
            _append_jsonl(output / "training_metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % validation_interval == 0:
            protected = [
                path
                for path in (
                    output / "best_macro_checkpoint.pt",
                    output / "best_worst_checkpoint.pt",
                    continuation_path,
                )
                if path.is_file()
            ]
            readonly = readonly_validation(
                policy,
                optimizer,
                env,
                hp,
                config,
                model_kind,
                library,
                protected,
            )
            validation = readonly["validation"]
            _append_jsonl(
                output / "validation_metrics.jsonl",
                {
                    "update": update,
                    "completed_updates": completed,
                    "validation_kind": "scheduled",
                    **readonly,
                },
            )
            _save_selected_checkpoints(
                policy,
                optimizer,
                hp,
                update,
                validation,
                model_kind,
                arm,
                manifest,
                output,
                selection_state,
            )
            tail = losses[-max(log_interval - 1, 0) :]
            payload = _continuation_payload(
                policy,
                optimizer,
                hp,
                update,
                validation,
                model_kind,
                arm,
                manifest,
                env,
                tail,
                selection_state,
            )
            _atomic_torch_save(payload, continuation_path)
    if not all(
        (output / name).is_file()
        for name in (
            "best_macro_checkpoint.pt",
            "best_worst_checkpoint.pt",
            "continuation_checkpoint.pt",
        )
    ):
        raise RuntimeError("dual-RNN worker did not produce all selected checkpoints")
    final_payload = torch.load(
        continuation_path, map_location="cpu", weights_only=False
    )
    final_payload["checkpoint_kind"] = "dual_fixed_rule_final"
    final_payload.pop("continuation_state", None)
    _atomic_torch_save(final_payload, output / "final_checkpoint.pt")
    diagnostic = write_worker_diagnostics(
        config, geometry, model_kind, arm, manifest, output
    )
    summary = {
        "model_kind": model_kind,
        "loss_arm": arm,
        "completed_updates": max_updates,
        "updates_per_rule": _updates_per_rule(config, model_kind),
        "optimizer_step_mode": _optimizer_step_mode(config),
        "rules_per_optimizer_step": len(
            _training_conditions(config, model_kind, 0, library, by_rule)
        ),
        "phase1_updates": phase1_updates,
        "phase2_updates": schedule["phase2_updates"],
        "elapsed_seconds": time.monotonic() - started,
        "selection_state": selection_state,
        **diagnostic,
    }
    _write_json(output / "worker_summary.json", summary)
    return summary


def _load_policy(
    path: Path,
    config: dict[str, Any],
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(
        checkpoint, model_kind, arm, manifest, config["variant"]
    )
    policy = _build_policy(
        checkpoint["hp"], config["model_common"]["output_size"], torch.device("cpu")
    )
    policy.load_state_dict(checkpoint["agent_state_dict"])
    policy.eval()
    return policy, checkpoint


def _candidate_curves(
    path: Path,
    candidate_name: str,
    config: dict[str, Any],
    geometry: GeometryConfig,
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    policy, checkpoint = _load_policy(path, config, model_kind, arm, manifest)
    hp = checkpoint["hp"]
    env = _make_env(config, model_kind)
    curves = []
    for condition in conditions(model_kind, geometry):
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            FIXED_SPEED_NAME,
            FIXED_DELAY_STEPS,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=False,
        )
        movement_start, movement_end = result["epoch_bounds"]["movement"]
        actual = (
            result["xy"][0, movement_start:movement_end].detach().cpu().numpy()
            - env.anchor_m
        )
        target = (
            result["target"][0, movement_start:movement_end]
            .detach()
            .cpu()
            .numpy()
            - env.anchor_m
        )
        if actual.shape != condition.points_m.shape or np.max(
            np.abs(target - condition.points_m)
        ) > 1e-7:
            raise RuntimeError("dual-RNN diagnostic target alignment differs")
        row = {
            "model_kind": model_kind,
            "loss_arm": arm,
            "candidate": candidate_name,
            "checkpoint_update": int(checkpoint["update"]),
            "rule": condition.rule,
            "condition_id": condition.condition_id,
            "source_character": condition.source_character,
            "source_component_index": condition.source_component_index,
            "movement_intervals": condition.movement_intervals,
            "movement_samples": len(condition.points_m),
            **_movement_metrics(
                actual, target, condition.canonical_rounding
            ),
        }
        curves.append({"row": row, "actual": actual, "target": target})
    return curves


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_overview(path: Path, curves: list[dict[str, Any]], title: str) -> None:
    columns = 3
    rows = int(np.ceil(len(curves) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.6 * rows))
    flat = np.asarray(axes).reshape(-1)
    for axis, curve in zip(flat, curves):
        target = curve["target"]
        actual = curve["actual"]
        axis.plot(target[:, 0], target[:, 1], "--", color="0.5", label="target")
        axis.plot(actual[:, 0], actual[:, 1], color="#4c78a8", label="actual")
        axis.scatter(target[0, 0], target[0, 1], color="green", s=18)
        axis.scatter(target[-1, 0], target[-1, 1], color="black", s=18)
        axis.set_title(curve["row"]["rule"], fontsize=9)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
    for axis in flat[len(curves) :]:
        axis.axis("off")
    flat[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_worker_diagnostics(
    config: dict[str, Any],
    geometry: GeometryConfig,
    model_kind: str,
    arm: str,
    manifest: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    paths = {
        "best_macro": output / "best_macro_checkpoint.pt",
        "best_worst": output / "best_worst_checkpoint.pt",
        "final": output / "final_checkpoint.pt",
    }
    all_rows = []
    arrays: dict[str, np.ndarray] = {}
    plot_directory = output / "plots"
    plot_directory.mkdir(exist_ok=False)
    for name, path in paths.items():
        curves = _candidate_curves(
            path, name, config, geometry, model_kind, arm, manifest
        )
        all_rows.extend(curve["row"] for curve in curves)
        for curve in curves:
            prefix = f"{name}__{curve['row']['rule']}"
            arrays[f"{prefix}__target"] = curve["target"]
            arrays[f"{prefix}__actual"] = curve["actual"]
        _plot_overview(
            plot_directory / f"{name}.png",
            curves,
            f"{model_kind} / {arm} / {name}",
        )
    _write_metrics(output / "candidate_metrics.csv", all_rows)
    np.savez_compressed(output / "candidate_trajectories.npz", **arrays)
    expected_rows = len(rule_names(model_kind)) * len(CANDIDATE_NAMES)
    if len(all_rows) != expected_rows or len(arrays) != expected_rows * 2:
        raise RuntimeError("dual-RNN worker diagnostic row count differs")
    return {
        "candidate_metric_rows": len(all_rows),
        "trajectory_arrays": len(arrays),
        "plot_count": len(paths),
    }


def _plot_target_audit(
    path: Path,
    libraries: dict[str, tuple[FixedRuleCondition, ...]],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 13))
    for axis, model_kind in zip(axes, MODEL_KINDS):
        for condition in libraries[model_kind]:
            points = condition.points_m
            axis.plot(points[:, 0], points[:, 1], label=condition.rule)
            axis.scatter(points[0, 0], points[0, 1], s=10)
        axis.set_title(f"{model_kind}: frozen target trajectories")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def prepare_experiment(config_path: str | Path) -> dict[str, Any]:
    config, geometry = load_config(config_path)
    output = Path(config["output"]["directory"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    (output / "stroke").mkdir()
    (output / "move").mkdir()
    (output / "worker_logs").mkdir()
    _write_json(output / "resolved_config.json", config)
    libraries = {kind: conditions(kind, geometry) for kind in MODEL_KINDS}
    manifests = {}
    arrays: dict[str, np.ndarray] = {}
    for model_kind in MODEL_KINDS:
        manifest = condition_manifest(model_kind, geometry)
        manifests[model_kind] = manifest
        _write_json(output / f"{model_kind}_condition_manifest.json", manifest)
        for condition in libraries[model_kind]:
            arrays[f"{model_kind}__{condition.rule}"] = condition.points_m
    np.savez_compressed(output / "target_trajectories.npz", **arrays)
    _plot_target_audit(output / "target_audit.png", libraries)
    joint_gradient = config["variant"] == JOINT_GRADIENT_VARIANT
    report = [
        "# Dual fixed-rule RNN preflight",
        "",
        f"- variant: `{config['variant']}`",
        "- Stroke RNN: 15 independent one-hot rules, input size 33",
        "- Move RNN: 12 independent one-hot rules, input size 30",
        "- fixed delay: 50 steps",
        (
            "- optimizer step: mean gradient over every rule"
            if joint_gradient
            else "- batch size: 1"
        ),
        (
            "- scheduler: all rules once per optimizer step"
            if joint_gradient
            else "- scheduler: deterministic fixed round-robin"
        ),
        "- loss arms: baseline, full_trial, onset_window",
        "- behavioral pass/fail threshold: none",
        "",
        "No complete-character or composed-controller training is included.",
    ]
    (output / "PREFLIGHT_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "output_directory": str(output),
        "parallel_workers": 6,
        "stroke_rule_count": len(libraries["stroke"]),
        "move_rule_count": len(libraries["move"]),
        "stroke_manifest_sha256": manifests["stroke"][
            "condition_manifest_sha256"
        ],
        "move_manifest_sha256": manifests["move"]["condition_manifest_sha256"],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def finalize_experiment(config_path: str | Path) -> dict[str, Any]:
    config, geometry = load_config(config_path)
    output = Path(config["output"]["directory"])
    all_rows: list[dict[str, str]] = []
    summaries = []
    plot_count = 0
    for model_kind in MODEL_KINDS:
        manifest = _load_json(output / f"{model_kind}_condition_manifest.json")
        if manifest != condition_manifest(model_kind, geometry):
            raise RuntimeError("dual-RNN final manifest differs")
        for arm in LOSS_ARMS:
            worker = output / model_kind / arm
            summary = _load_json(worker / "worker_summary.json")
            expected_rows = len(rule_names(model_kind)) * len(CANDIDATE_NAMES)
            if (
                summary["completed_updates"]
                != config["models"][model_kind]["max_updates"]
                or summary["updates_per_rule"] != 8000
                or summary["candidate_metric_rows"] != expected_rows
                or summary["plot_count"] != 3
            ):
                raise RuntimeError("dual-RNN worker summary differs")
            rows = _read_csv(worker / "candidate_metrics.csv")
            if len(rows) != expected_rows:
                raise RuntimeError("dual-RNN worker metrics row count differs")
            for row in rows:
                for field in METRIC_FIELDS[10:]:
                    if row[field] and not np.isfinite(float(row[field])):
                        raise RuntimeError("dual-RNN worker metric is non-finite")
            all_rows.extend(rows)
            summaries.append(summary)
            plot_count += summary["plot_count"]
    _write_metrics(output / "all_candidate_metrics.csv", all_rows)
    expected_total_rows = 3 * (15 * 3 + 12 * 3)
    if len(all_rows) != expected_total_rows or plot_count != 18:
        raise RuntimeError("dual-RNN final artifact counts differ")
    report = [
        "# Dual fixed-rule RNN loss comparison",
        "",
        "All six matched workers completed. This report defines no behavioral threshold.",
        "",
        "| Model | Loss arm | Updates | Updates/rule | Macro-best update | Worst-best update |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        selection = summary["selection_state"]
        report.append(
            "| {model_kind} | {loss_arm} | {completed_updates} | "
            "{updates_per_rule} | {macro} | {worst} |".format(
                **summary,
                macro=selection["best_macro_then_worst"]["update"],
                worst=selection["best_worst_then_macro"]["update"],
            )
        )
    report.extend(
        [
            "",
            "The macro-best, worst-rule-best and final checkpoints are retained for manual review.",
            "No complete-character, chained stroke/move, or automatic timing adjustment was run.",
        ]
    )
    (output / "FINAL_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )
    provenance = {
        "project": PROJECT,
        "variant": config["variant"],
        "completed": True,
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "parallel_workers": 6,
        "models": list(MODEL_KINDS),
        "loss_arms": list(LOSS_ARMS),
        "fixed_delay_steps": FIXED_DELAY_STEPS,
        "automatic_checkpoint_selection_performed": True,
        "checkpoint_profiles": list(CHECKPOINT_PROFILES),
        "behavioral_pass_fail_defined": False,
        "complete_character_started": False,
        "chained_controller_started": False,
        "integrity": {
            "workers_completed": len(summaries),
            "candidate_metric_rows": len(all_rows),
            "plots": plot_count,
            "all_numeric_values_finite": True,
        },
    }
    if config["variant"] == VARIANT:
        provenance["batch_size"] = 1
    else:
        provenance.update(
            {
                "microbatch_size_per_rule": 1,
                "optimizer_step_mode": _optimizer_step_mode(config),
                "effective_rules_per_optimizer_step": {
                    model_kind: len(rule_names(model_kind))
                    for model_kind in MODEL_KINDS
                },
                "optimizer_steps_per_model": {
                    model_kind: config["models"][model_kind]["max_updates"]
                    for model_kind in MODEL_KINDS
                },
                "rule_exposures_per_rule": config["training"]["updates_per_rule"],
            }
        )
    _write_json(output / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("prepare")
    worker = modes.add_parser("train-worker")
    worker.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    worker.add_argument("--arm", choices=LOSS_ARMS, required=True)
    modes.add_parser("finalize")
    arguments = parser.parse_args()
    if arguments.mode == "prepare":
        result = prepare_experiment(arguments.config)
    elif arguments.mode == "train-worker":
        result = train_worker(arguments.config, arguments.model_kind, arguments.arm)
    else:
        result = finalize_experiment(arguments.config)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
