"""Nine independent canonical single-task overfit runs and minimal diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.canonical_protocol import (
    COMPOUND_RULES,
    EXPECTED_TARGET_ROWS,
    STAGE0_ARTIFACT_NAMES,
    canonical_stroke_conditions,
    write_stage0_artifacts,
)
from hanzi_writing.envs import HanziComponentEnv
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    CANONICAL_GEOMETRY_VARIANT,
    CANONICAL_TIMING_MODE,
    PROJECT,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    load_geometry_config,
    move_conditions,
)
from hanzi_writing.training import (
    _environment_generator_states,
    _generator_state_equal,
    _gradient_state,
    _hp_from_config,
    _make_effector,
    _nested_equal,
    _restore_generator_state,
    _restore_gradients,
    _rollout,
    _sha256_file,
    _state_clone,
)
from losses import (
    compound_subphase_equal_position_l1,
    detached_position_metrics,
    l1_muscle_act,
    l1_rate,
    l1_weight,
    onset_window_position_l1,
    position_l1_metrics,
    simple_dynamics,
)
from train import _build_policy, _checkpoint_payload, _fixed_rng


CANONICAL_VARIANT = "canonical_single_duration_overfit_v1"
SHARED_VARIANT = "canonical_shared_9task_v1"
EXPECTED_METRIC_ROWS = 40
EXPECTED_PLOTS = 9
METRIC_NAMES = (
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
METRIC_FIELDS = (
    "task",
    "category",
    "rule",
    "condition_id",
    "source_character",
    "source_component_index",
    "checkpoint",
    "checkpoint_update",
    "checkpoint_validation_loss",
    "movement_intervals",
    "movement_samples",
    *METRIC_NAMES,
)


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("canonical configuration must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True))
        handle.write("\n")


def _baseline_model() -> dict[str, Any]:
    return {
        "network": "rnn",
        "input_size": 28,
        "hidden_size": 256,
        "output_size": 6,
        "activation": "softplus",
        "recurrent_noise_std": 0.1,
        "input_noise_std": 0.01,
        "constrained": False,
        "rnn_dt_ms": 10,
        "rnn_tau_ms": 20,
        "batch_first": True,
    }


def _baseline_optimizer() -> dict[str, Any]:
    return {"name": "Adam", "learning_rate": 0.001, "grad_clip_norm": 1.0}


def _baseline_position_loss() -> dict[str, Any]:
    return {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }


def _baseline_regularization() -> dict[str, Any]:
    return {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }


def load_canonical_overfit_config(path: str | Path) -> tuple[dict[str, Any], GeometryConfig]:
    config = _load_json(path)
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "timing_mode",
            "geometry_variant",
            "enabled",
            "seed",
            "validation_seed",
            "device",
            "geometry_config",
            "active_rules",
            "rule_dim_total",
            "unused_rule_index",
            "model",
            "optimizer",
            "training",
            "position_loss",
            "regularization",
            "diagnostic_contract",
            "output",
        },
        "canonical overfit configuration",
    )
    identity = (
        config["project"],
        config["run_kind"],
        config["variant"],
        config["timing_mode"],
        config["geometry_variant"],
        config["enabled"],
        config["seed"],
        config["validation_seed"],
        config["device"],
    )
    if identity != (
        PROJECT,
        "canonical_single_task_overfit",
        CANONICAL_VARIANT,
        CANONICAL_TIMING_MODE,
        CANONICAL_GEOMETRY_VARIANT,
        True,
        42,
        1042,
        "cpu",
    ):
        raise ValueError("canonical overfit identity differs")
    if config["geometry_config"] != (
        "configurations/hanzi_stroke_temporal_composition_canonical_geometry.json"
    ):
        raise ValueError("canonical geometry path differs")
    if tuple(config["active_rules"]) != ACTIVE_RULES:
        raise ValueError("canonical active rules differ")
    if config["rule_dim_total"] != 10 or config["unused_rule_index"] != 9:
        raise ValueError("canonical rule dimensions differ")
    if config["model"] != _baseline_model():
        raise ValueError("canonical model differs from the frozen baseline")
    if config["optimizer"] != _baseline_optimizer():
        raise ValueError("canonical optimizer differs from the frozen baseline")
    if config["training"] != {
        "batch_size": 1,
        "initial_review_updates": 6000,
        "validation_interval": 250,
        "log_interval": 100,
        "speed_name": "slow",
        "delay_steps": [25, 50, 75],
        "delay_sampler": "seeded_random_choice_legacy_grid",
        "network_noise": False,
        "deterministic_observation": True,
        "stroke_sampler": "repeat_one_canonical_exact_condition",
        "move_sampler": "deterministic_round_robin_12_exact_conditions",
    }:
        raise ValueError("canonical training schedule differs")
    if config["position_loss"] != _baseline_position_loss():
        raise ValueError("canonical position loss differs")
    if config["regularization"] != _baseline_regularization():
        raise ValueError("canonical regularization differs")
    if config["diagnostic_contract"] != {
        "metrics_rows": EXPECTED_METRIC_ROWS,
        "best_trajectory_rows": EXPECTED_TARGET_ROWS,
        "plots": EXPECTED_PLOTS,
        "behavioral_pass_fail": "not_defined",
        "complete_character_rollout": False,
    }:
        raise ValueError("canonical diagnostic contract differs")
    if config["output"] != {
        "directory": (
            "runs/hanzi_stroke_temporal_composition/"
            "canonical_single_task_overfit/dev42"
        )
    }:
        raise ValueError("canonical output path differs")
    geometry = load_geometry_config(config["geometry_config"])
    if (
        geometry.timing_mode != CANONICAL_TIMING_MODE
        or geometry.geometry_variant != CANONICAL_GEOMETRY_VARIANT
    ):
        raise ValueError("canonical overfit references the wrong geometry")
    return config, geometry


def validate_frozen_shared_config(path: str | Path) -> dict[str, Any]:
    config = _load_json(path)
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "timing_mode",
            "geometry_variant",
            "enabled",
            "seed",
            "validation_seed",
            "device",
            "geometry_config",
            "active_rules",
            "training",
            "authorization",
        },
        "frozen canonical shared configuration",
    )
    if config != {
        "project": PROJECT,
        "run_kind": "canonical_shared_9task",
        "variant": SHARED_VARIANT,
        "timing_mode": CANONICAL_TIMING_MODE,
        "geometry_variant": CANONICAL_GEOMETRY_VARIANT,
        "enabled": False,
        "seed": 42,
        "validation_seed": 1042,
        "device": "cpu",
        "geometry_config": (
            "configurations/hanzi_stroke_temporal_composition_canonical_geometry.json"
        ),
        "active_rules": list(ACTIVE_RULES),
        "training": {
            "batch_size": 32,
            "max_updates": 10000,
            "speed_name": "slow",
            "delay_steps": [25, 50, 75],
            "delay_sampler": "seeded_random_choice_legacy_grid",
            "network_noise": False,
            "deterministic_observation": True,
            "sampler": "uniform_rule_then_uniform_move_condition",
        },
        "authorization": "requires_explicit_user_approval_after_stage1_review",
    }:
        raise ValueError("frozen canonical shared configuration differs")
    return config


def _task_conditions(
    geometry: GeometryConfig,
) -> tuple[tuple[str, tuple[StrokeCondition | MoveCondition, ...]], ...]:
    tasks = [
        (condition.rule, (condition,))
        for condition in canonical_stroke_conditions(geometry)
    ]
    tasks.append(("move", move_conditions(geometry, include_jitter=False)))
    if len(tasks) != 9 or len(tasks[-1][1]) != 12:
        raise RuntimeError("canonical Stage 1 requires eight stroke tasks and one move task")
    return tuple(tasks)


def _condition_for_update(
    task: str,
    conditions: tuple[StrokeCondition | MoveCondition, ...],
    update: int,
) -> StrokeCondition | MoveCondition:
    if update < 0 or not conditions:
        raise ValueError("canonical training update and conditions must be valid")
    return conditions[0] if task != "move" else conditions[update % len(conditions)]


def _canonical_validation(
    policy,
    env: HanziComponentEnv,
    hp: dict[str, Any],
    conditions: tuple[StrokeCondition | MoveCondition, ...],
    geometry: GeometryConfig,
) -> dict[str, Any]:
    rows = []
    for condition in conditions:
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            geometry.canonical_speed_name or "slow",
            geometry.validation_delay_steps,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=False,
        )
        rows.append(
            {
                "condition_id": condition.condition_id,
                **detached_position_metrics(
                    position_l1_metrics(
                        result["xy"], result["target"], result["epoch_bounds"]
                    )
                ),
            }
        )
    metric_names = tuple(name for name in rows[0] if name != "condition_id")
    aggregate = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    return {
        "aggregation": "single_condition" if len(rows) == 1 else "equal_mean_over_12_moves",
        "aggregate": aggregate,
        "per_condition": rows,
        "condition_count": len(rows),
    }


def canonical_readonly_validation(
    policy,
    optimizer,
    env: HanziComponentEnv,
    hp: dict[str, Any],
    conditions: tuple[StrokeCondition | MoveCondition, ...],
    geometry: GeometryConfig,
    checkpoint_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    policy_mode = bool(policy.training)
    policy_state = _state_clone(policy)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    gradients = _gradient_state(policy)
    python_rng = copy.deepcopy(random.getstate())
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    generator_states = _environment_generator_states(env)
    if not generator_states:
        raise RuntimeError("canonical validation environment RNG snapshot is empty")
    checkpoint_hashes = {
        str(Path(path)): _sha256_file(path) for path in checkpoint_paths
    }
    validation = None
    checks: dict[str, bool] = {}
    try:
        policy.eval()
        with _fixed_rng(geometry.validation_seed):
            validation_env = HanziComponentEnv(
                effector=_make_effector(),
                geometry_config_path=hp["protocol_config"]["geometry_config"],
                action_frame_stacking=0,
            )
            with torch.no_grad():
                validation = _canonical_validation(
                    policy, validation_env, hp, conditions, geometry
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
        raise RuntimeError(
            "canonical read-only validation failed: "
            + ",".join(sorted(name for name, passed in checks.items() if not passed))
        )
    return {
        "validation": validation,
        "read_only_checks": {
            **checks,
            "environment_generator_capture_nonempty": bool(generator_states),
            "optimizer_not_created_during_validation": True,
            "backward_not_executed_during_validation": True,
            "captured_environment_generator_paths": [
                path for _, _, _, path in generator_states
            ],
        },
    }


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _continuation_payload(
    policy,
    optimizer,
    hp: dict[str, Any],
    update: int,
    validation_loss: float,
    task: str,
    checkpoint_kind: str,
    env: HanziComponentEnv,
    losses: list[float],
    log_interval: int,
) -> dict[str, Any]:
    tail_length = max(log_interval - 1, 0)
    payload = _checkpoint_payload(
        policy, optimizer, hp, update, validation_loss
    )
    payload.update(
        {
            "project": PROJECT,
            "checkpoint_kind": checkpoint_kind,
            "task": task,
            "continuation_state": {
                "next_update": update + 1,
                "loss_tail": losses[-tail_length:] if tail_length else [],
                "environment_rng_states": [
                    {"kind": kind, "state": state, "path": path}
                    for _, kind, state, path in _environment_generator_states(env)
                ],
            },
        }
    )
    if not payload["continuation_state"]["environment_rng_states"]:
        raise RuntimeError("canonical continuation environment RNG state is empty")
    return payload


def _restore_continuation_state(
    checkpoint: dict[str, Any],
    policy,
    optimizer,
    env: HanziComponentEnv,
    hp: dict[str, Any],
    task: str,
    expected_variant: str = CANONICAL_VARIANT,
    checkpoint_prefix: str = "canonical_single_task",
) -> tuple[int, list[float]]:
    if (
        checkpoint.get("project") != PROJECT
        or checkpoint.get("variant") != expected_variant
        or checkpoint.get("checkpoint_kind")
        != f"{checkpoint_prefix}_continuation"
        or checkpoint.get("task") != task
        or not _nested_equal(checkpoint.get("hp"), hp)
    ):
        raise ValueError("canonical continuation checkpoint identity differs")
    policy.load_state_dict(checkpoint["agent_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict) or set(rng_state) != {
        "python",
        "numpy",
        "torch",
    }:
        raise ValueError("canonical continuation global RNG state differs")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    continuation = checkpoint.get("continuation_state")
    if not isinstance(continuation, dict) or set(continuation) != {
        "next_update",
        "loss_tail",
        "environment_rng_states",
    }:
        raise ValueError("canonical continuation state differs")
    expected_states = {
        path: (generator, kind)
        for generator, kind, _, path in _environment_generator_states(env)
    }
    saved_states = continuation["environment_rng_states"]
    if (
        not isinstance(saved_states, list)
        or {entry.get("path") for entry in saved_states} != set(expected_states)
    ):
        raise ValueError("canonical continuation environment RNG paths differ")
    for entry in saved_states:
        generator, kind = expected_states[entry["path"]]
        if entry.get("kind") != kind:
            raise ValueError("canonical continuation environment RNG kinds differ")
        _restore_generator_state(generator, kind, entry["state"])
    next_update = continuation["next_update"]
    loss_tail = continuation["loss_tail"]
    if (
        not isinstance(next_update, int)
        or next_update != checkpoint.get("update", -2) + 1
        or not isinstance(loss_tail, list)
        or not all(np.isfinite(float(value)) for value in loss_tail)
    ):
        raise ValueError("canonical continuation update or loss tail differs")
    return next_update, [float(value) for value in loss_tail]


def _train_task(
    task: str,
    conditions: tuple[StrokeCondition | MoveCondition, ...],
    config: dict[str, Any],
    geometry: GeometryConfig,
    output: Path,
    target_updates: int | None = None,
    initial_checkpoint_path: Path | None = None,
    checkpoint_variant: str = CANONICAL_VARIANT,
    checkpoint_prefix: str = "canonical_single_task",
    initial_checkpoint_variant: str = CANONICAL_VARIANT,
    initial_checkpoint_prefix: str = "canonical_single_task",
    training_position_objective: str | None = None,
    save_scheduled_checkpoints: bool = False,
) -> dict[str, Any]:
    training = config["training"]
    if target_updates is None:
        target_updates = training["initial_review_updates"]
    if not isinstance(target_updates, int) or isinstance(target_updates, bool):
        raise ValueError("canonical target updates must be an integer")
    if target_updates < 1:
        raise ValueError("canonical target updates must be positive")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    hp = _hp_from_config(config)
    policy = _build_policy(hp, config["model"]["output_size"], torch.device("cpu"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    best_path = output / "best_checkpoint.pt"
    continuation_path = output / "continuation_checkpoint.pt"
    review_path = output / "review_checkpoint.pt"
    candidate_directory = output / "scheduled_checkpoints"
    if output.exists():
        if not continuation_path.is_file():
            raise FileExistsError(
                "canonical task directory exists without a continuation checkpoint"
            )
        env.reset(
            seed=config["seed"],
            options={
                "conditions": (conditions[0],),
                "speed_name": training["speed_name"],
                "delay_steps": geometry.validation_delay_steps,
                "deterministic": True,
            },
        )
        continuation_checkpoint = torch.load(
            continuation_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if (
            training_position_objective is not None
            and continuation_checkpoint.get("training_position_objective")
            != training_position_objective
        ):
            raise ValueError("canonical continuation position objective differs")
        start_update, losses = _restore_continuation_state(
            continuation_checkpoint,
            policy,
            optimizer,
            env,
            hp,
            task,
            checkpoint_variant,
            checkpoint_prefix,
        )
        if not best_path.is_file():
            raise FileNotFoundError("canonical best checkpoint is missing")
        best_checkpoint = torch.load(
            best_path, map_location=torch.device("cpu"), weights_only=False
        )
        if (
            best_checkpoint.get("project") != PROJECT
            or best_checkpoint.get("variant") != checkpoint_variant
            or best_checkpoint.get("checkpoint_kind")
            != f"{checkpoint_prefix}_best"
            or best_checkpoint.get("task") != task
            or (
                training_position_objective is not None
                and best_checkpoint.get("training_position_objective")
                != training_position_objective
            )
        ):
            raise ValueError("canonical best checkpoint identity differs")
        best_validation = float(best_checkpoint["validation_loss"])
        best_update = int(best_checkpoint["update"])
    else:
        output.mkdir(parents=False, exist_ok=False)
        if initial_checkpoint_path is not None:
            initial_checkpoint = torch.load(
                initial_checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            if (
                initial_checkpoint.get("project") != PROJECT
                or initial_checkpoint.get("variant") != initial_checkpoint_variant
                or initial_checkpoint.get("checkpoint_kind")
                not in {
                    f"{initial_checkpoint_prefix}_best",
                    f"{initial_checkpoint_prefix}_review",
                }
                or initial_checkpoint.get("task") != task
            ):
                raise ValueError("canonical refinement source checkpoint differs")
            policy.load_state_dict(initial_checkpoint["agent_state_dict"])
            optimizer.load_state_dict(initial_checkpoint["optimizer_state_dict"])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = hp["lr"]
            rng_state = initial_checkpoint.get("rng_state")
            if not isinstance(rng_state, dict) or set(rng_state) != {
                "python",
                "numpy",
                "torch",
            }:
                raise ValueError("canonical refinement source RNG state differs")
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
        if save_scheduled_checkpoints:
            candidate_directory.mkdir()
        losses = []
        best_validation = np.inf
        best_update = None
        start_update = 0
    if start_update > target_updates:
        raise ValueError("canonical target updates precede the continuation checkpoint")
    if start_update == target_updates and review_path.is_file():
        review_checkpoint = torch.load(
            review_path, map_location=torch.device("cpu"), weights_only=False
        )
        if (
            review_checkpoint.get("checkpoint_kind")
            != f"{checkpoint_prefix}_review"
            or review_checkpoint.get("task") != task
            or review_checkpoint.get("update") != target_updates - 1
        ):
            raise ValueError("canonical review checkpoint identity differs")
        return {
            "task": task,
            "best_update": best_update,
            "best_validation_loss": float(best_validation),
            "review_update": target_updates - 1,
            "review_validation_loss": float(
                review_checkpoint["validation_loss"]
            ),
            "resumed_from_update": start_update,
            "target_updates": target_updates,
        }
    for update in range(start_update, target_updates):
        condition = _condition_for_update(task, conditions, update)
        delay_steps = random.choice(geometry.delay_steps)
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            training["speed_name"],
            delay_steps,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=True,
        )
        position = position_l1_metrics(
            result["xy"], result["target"], result["epoch_bounds"]
        )
        if training_position_objective in (None, "phase_normalized_l1"):
            position_objective = position["phase_normalized_position_l1"]
        elif training_position_objective == "full_trial_l1":
            position_objective = position["full_trial_position_l1"]
        elif training_position_objective == "onset_window_phase_normalized_l1":
            onset_position = onset_window_position_l1(
                result["xy"], result["target"], result["epoch_bounds"]
            )
            position_objective = onset_position.pop("objective")
            position.update(onset_position)
        elif training_position_objective == "compound_subphase_equal_l1":
            subphase_position = compound_subphase_equal_position_l1(
                result["xy"],
                result["target"],
                result["epoch_bounds"],
                env.component_trajectories[0].movement_subphase,
            )
            position_objective = subphase_position.pop("objective")
            position.update(subphase_position)
        else:
            raise ValueError("canonical training position objective differs")
        loss = position_objective
        loss = loss + l1_rate(result["hidden"], hp["l1_rate"])
        loss = loss + l1_weight(policy, hp["l1_weight"])
        loss = loss + l1_muscle_act(result["muscle"], hp["l1_muscle_act"])
        loss = loss + simple_dynamics(
            result["hidden"], policy.mrnn, weight=hp["simple_dynamics_weight"]
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), hp["grad_clip_norm"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update % training["log_interval"] == 0:
            interval = min(update + 1, training["log_interval"])
            row = {
                "update": update,
                "task": task,
                "condition_id": condition.condition_id,
                "sampled_delay_steps": delay_steps,
                "mean_total_loss": float(np.mean(losses[-interval:])),
                **detached_position_metrics(position),
            }
            _append_jsonl(output / "training_metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if update % training["validation_interval"] == 0:
            readonly = canonical_readonly_validation(
                policy,
                optimizer,
                env,
                hp,
                conditions,
                geometry,
                (best_path,) if best_path.is_file() else (),
            )
            validation = readonly["validation"]
            value = validation["aggregate"]["phase_normalized_position_l1"]
            _append_jsonl(
                output / "validation_metrics.jsonl",
                {"update": update, "validation_kind": "scheduled", **validation},
            )
            if value <= best_validation:
                best_validation = value
                best_update = update
                payload = _checkpoint_payload(policy, optimizer, hp, update, value)
                payload.update(
                    {
                        "project": PROJECT,
                        "checkpoint_kind": f"{checkpoint_prefix}_best",
                        "task": task,
                    }
                )
                if training_position_objective is not None:
                    payload["training_position_objective"] = (
                        training_position_objective
                    )
                _atomic_torch_save(payload, best_path)
            if save_scheduled_checkpoints:
                candidate_payload = _checkpoint_payload(
                    policy, optimizer, hp, update, value
                )
                candidate_payload.update(
                    {
                        "project": PROJECT,
                        "checkpoint_kind": f"{checkpoint_prefix}_candidate",
                        "task": task,
                        "training_position_objective": training_position_objective,
                    }
                )
                _atomic_torch_save(
                    candidate_payload,
                    candidate_directory / f"update_{update:06d}.pt",
                )
            continuation_payload = _continuation_payload(
                policy,
                optimizer,
                hp,
                update,
                value,
                task,
                f"{checkpoint_prefix}_continuation",
                env,
                losses,
                training["log_interval"],
            )
            if training_position_objective is not None:
                continuation_payload["training_position_objective"] = (
                    training_position_objective
                )
            _atomic_torch_save(continuation_payload, continuation_path)
    review_readonly = canonical_readonly_validation(
        policy,
        optimizer,
        env,
        hp,
        conditions,
        geometry,
        (best_path, continuation_path),
    )
    review_value = review_readonly["validation"]["aggregate"][
        "phase_normalized_position_l1"
    ]
    _append_jsonl(
        output / "validation_metrics.jsonl",
        {
            "update": target_updates - 1,
            "validation_kind": "review_read_only",
            **review_readonly,
        },
    )
    continuation_payload = _continuation_payload(
        policy,
        optimizer,
        hp,
        target_updates - 1,
        review_value,
        task,
        f"{checkpoint_prefix}_continuation",
        env,
        losses,
        training["log_interval"],
    )
    if training_position_objective is not None:
        continuation_payload["training_position_objective"] = (
            training_position_objective
        )
    _atomic_torch_save(continuation_payload, continuation_path)
    review_payload = _continuation_payload(
        policy,
        optimizer,
        hp,
        target_updates - 1,
        review_value,
        task,
        f"{checkpoint_prefix}_review",
        env,
        losses,
        training["log_interval"],
    )
    if training_position_objective is not None:
        review_payload["training_position_objective"] = training_position_objective
    _atomic_torch_save(review_payload, review_path)
    return {
        "task": task,
        "best_update": best_update,
        "best_validation_loss": float(best_validation),
        "review_update": target_updates - 1,
        "review_validation_loss": float(review_value),
        "review_validation_read_only_checks": review_readonly["read_only_checks"],
        "resumed_from_update": start_update,
        "target_updates": target_updates,
    }


def _load_task_checkpoint(
    path: Path,
    task: str,
    checkpoint_name: str,
    *,
    expected_variant: str = CANONICAL_VARIANT,
    checkpoint_prefix: str = "canonical_single_task",
):
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    expected_kind = f"{checkpoint_prefix}_{checkpoint_name}"
    if (
        checkpoint.get("project") != PROJECT
        or checkpoint.get("variant") != expected_variant
        or checkpoint.get("checkpoint_kind") != expected_kind
        or checkpoint.get("task") != task
    ):
        raise ValueError("canonical single-task checkpoint identity differs")
    hp = checkpoint.get("hp")
    if not isinstance(hp, dict):
        raise ValueError("canonical checkpoint is missing hyperparameters")
    policy = _build_policy(hp, 6, torch.device("cpu"))
    policy.load_state_dict(checkpoint["agent_state_dict"])
    policy.eval()
    return policy, checkpoint


def _angle_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    actual_norm = float(np.linalg.norm(actual))
    target_norm = float(np.linalg.norm(target))
    if actual_norm <= 0.0 or target_norm <= 0.0:
        raise RuntimeError("exit-direction chord has zero length")
    cosine = float(np.clip(actual @ target / (actual_norm * target_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _movement_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    rounding: dict[str, Any] | None,
) -> dict[str, float | None]:
    if (
        actual.shape != target.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(target).all()
    ):
        raise RuntimeError("canonical diagnostic movement arrays differ or are non-finite")
    errors = np.linalg.norm(actual - target, axis=1)
    target_length = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    actual_length = float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum())
    values: dict[str, float | None] = {
        "start_error_euclidean_m": float(errors[0]),
        "movement_mean_euclidean_m": float(errors.mean()),
        "endpoint_euclidean_m": float(errors[-1]),
        "target_path_length_m": target_length,
        "actual_path_length_m": actual_length,
        "path_length_ratio": actual_length / target_length,
        "max_euclidean_m": float(errors.max()),
        "pre_straight_mean_euclidean_m": None,
        "transition_mean_euclidean_m": None,
        "post_straight_mean_euclidean_m": None,
        "transition_max_euclidean_m": None,
        "exit_direction_error_deg": None,
    }
    if rounding is not None:
        pre_start, pre_end = rounding["pre_indices"]
        transition_start, transition_end = rounding["transition_indices"]
        post_start, post_end = rounding["post_indices"]
        values.update(
            {
                "pre_straight_mean_euclidean_m": float(errors[pre_start:pre_end].mean()),
                "transition_mean_euclidean_m": float(
                    errors[transition_start:transition_end].mean()
                ),
                "post_straight_mean_euclidean_m": float(errors[post_start:post_end].mean()),
                "transition_max_euclidean_m": float(
                    errors[transition_start:transition_end].max()
                ),
                "exit_direction_error_deg": _angle_error_deg(
                    actual[-1] - actual[post_start],
                    target[-1] - target[post_start],
                ),
            }
        )
    return values


def _checkpoint_curves(
    task: str,
    conditions: tuple[StrokeCondition | MoveCondition, ...],
    checkpoint_name: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    geometry: GeometryConfig,
    *,
    checkpoint_identity_name: str | None = None,
    expected_variant: str = CANONICAL_VARIANT,
    checkpoint_prefix: str = "canonical_single_task",
) -> list[dict[str, Any]]:
    policy, checkpoint = _load_task_checkpoint(
        checkpoint_path,
        task,
        checkpoint_identity_name or checkpoint_name,
        expected_variant=expected_variant,
        checkpoint_prefix=checkpoint_prefix,
    )
    hp = checkpoint["hp"]
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    curves = []
    anchor = None
    for condition in conditions:
        result = _rollout(
            policy,
            env,
            hp,
            (condition,),
            geometry.canonical_speed_name or "slow",
            geometry.validation_delay_steps,
            network_noise=False,
            deterministic_observation=True,
            track_gradients=False,
        )
        movement_start, movement_end = result["epoch_bounds"]["movement"]
        if anchor is None:
            anchor = np.asarray(env.anchor_m, dtype=np.float64)
        actual = (
            result["xy"][0, movement_start:movement_end].detach().cpu().numpy()
            - anchor
        )
        target = (
            result["target"][0, movement_start:movement_end].detach().cpu().numpy()
            - anchor
        )
        trajectory = env.component_trajectories[0]
        if actual.shape != trajectory.points_m.shape:
            raise RuntimeError("canonical diagnostic movement alignment differs")
        if float(np.max(np.abs(target - trajectory.points_m))) > 1e-7:
            raise RuntimeError("executed canonical target differs from the frozen trajectory")
        row = {
            "task": task,
            "category": "stroke" if isinstance(condition, StrokeCondition) else "move",
            "rule": condition.rule if isinstance(condition, StrokeCondition) else "move",
            "condition_id": condition.condition_id,
            "source_character": condition.character,
            "source_component_index": (
                condition.stroke_index
                if isinstance(condition, StrokeCondition)
                else condition.transition_index
            ),
            "checkpoint": checkpoint_name,
            "checkpoint_update": int(checkpoint["update"]),
            "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
            "movement_intervals": trajectory.movement_intervals,
            "movement_samples": len(trajectory.points_m),
            **_movement_metrics(actual, target, trajectory.canonical_rounding),
        }
        curves.append(
            {
                "row": row,
                "actual": actual,
                "target": target,
                "subphase": np.asarray(trajectory.movement_subphase, dtype="U16"),
            }
        )
    return curves


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _best_trajectory_arrays(curves: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    selected = [curve for curve in curves if curve["row"]["checkpoint"] == "best"]
    arrays: dict[str, list[np.ndarray]] = {
        "task": [],
        "category": [],
        "rule": [],
        "condition_id": [],
        "sample_index": [],
        "target_xy_m": [],
        "actual_xy_m": [],
        "euclidean_error_m": [],
        "subphase": [],
    }
    for curve in selected:
        row = curve["row"]
        samples = len(curve["target"])
        arrays["task"].append(np.full(samples, row["task"], dtype="U12"))
        arrays["category"].append(np.full(samples, row["category"], dtype="U8"))
        arrays["rule"].append(np.full(samples, row["rule"], dtype="U12"))
        arrays["condition_id"].append(
            np.full(samples, row["condition_id"], dtype="U64")
        )
        arrays["sample_index"].append(np.arange(samples, dtype=np.int64))
        arrays["target_xy_m"].append(curve["target"])
        arrays["actual_xy_m"].append(curve["actual"])
        arrays["euclidean_error_m"].append(
            np.linalg.norm(curve["actual"] - curve["target"], axis=1)
        )
        arrays["subphase"].append(curve["subphase"])
    output = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    if len(output["sample_index"]) != EXPECTED_TARGET_ROWS:
        raise RuntimeError("canonical best trajectory row count must equal 3120")
    return output


def _plot_stroke(path: Path, curve: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(curve["target"][:, 0], curve["target"][:, 1], color="0.65", label="target")
    axis.plot(curve["actual"][:, 0], curve["actual"][:, 1], color="#e45756", label="actual")
    axis.set_title(curve["row"]["rule"])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_moves(path: Path, curves: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for axis, curve in zip(axes.flat, curves):
        axis.plot(curve["target"][:, 0], curve["target"][:, 1], color="0.65")
        axis.plot(curve["actual"][:, 0], curve["actual"][:, 1], color="#4c78a8")
        axis.set_title(curve["row"]["condition_id"], fontsize=8)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_plots(output: Path, curves: list[dict[str, Any]]) -> list[Path]:
    plot_directory = output / "plots"
    plot_directory.mkdir(exist_ok=True)
    best = [curve for curve in curves if curve["row"]["checkpoint"] == "best"]
    strokes = [curve for curve in best if curve["row"]["category"] == "stroke"]
    moves = [curve for curve in best if curve["row"]["category"] == "move"]
    for curve in strokes:
        _plot_stroke(plot_directory / f"{curve['row']['rule']}.png", curve)
    _plot_moves(plot_directory / "move.png", moves)
    plots = sorted(plot_directory.glob("*.png"))
    if len(plots) != EXPECTED_PLOTS:
        raise RuntimeError("canonical plot count must equal 9")
    return plots


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.9g}"


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    best = [row for row in rows if row["checkpoint"] == "best"]
    review = [row for row in rows if row["checkpoint"] == "review"]
    lines = [
        "# Canonical single-task overfit report",
        "",
        (
            "These are deterministic single-seed isolated-task descriptions. "
            "No behavioral pass/fail threshold, complete-character claim, "
            "shared-model claim, or MotorNet-only capacity claim is defined."
        ),
        "",
        "## Best checkpoint metrics",
        "",
        "| task | condition | mean error (m) | endpoint error (m) | path ratio | max error (m) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in best:
        if row["rule"] == "move":
            continue
        lines.append(
            f"| {row['task']} | {row['condition_id']} | "
            f"{_format_metric(row['movement_mean_euclidean_m'])} | "
            f"{_format_metric(row['endpoint_euclidean_m'])} | "
            f"{_format_metric(row['path_length_ratio'])} | "
            f"{_format_metric(row['max_euclidean_m'])} |"
        )
    lines.extend(
        [
            "",
            "## Rounded compound strokes",
            "",
            (
                "| checkpoint | rule | pre mean (m) | transition mean/max (m) | "
                "post mean (m) | exit direction error (deg) |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["rule"] not in COMPOUND_RULES:
            continue
        lines.append(
            f"| {row['checkpoint']} | {row['rule']} | "
            f"{_format_metric(row['pre_straight_mean_euclidean_m'])} | "
            f"{_format_metric(row['transition_mean_euclidean_m'])} / "
            f"{_format_metric(row['transition_max_euclidean_m'])} | "
            f"{_format_metric(row['post_straight_mean_euclidean_m'])} | "
            f"{_format_metric(row['exit_direction_error_deg'])} |"
        )
    move_best = [row for row in best if row["rule"] == "move"]
    lines.extend(["", "## Move spread", ""])
    for metric in (
        "movement_mean_euclidean_m",
        "endpoint_euclidean_m",
        "path_length_ratio",
        "max_euclidean_m",
    ):
        values = np.asarray([float(row[metric]) for row in move_best])
        minimum = min(move_best, key=lambda row: float(row[metric]))
        maximum = max(move_best, key=lambda row: float(row[metric]))
        lines.append(
            f"- {metric}: min={values.min():.9g}, median={np.median(values):.9g}, "
            f"mean={values.mean():.9g}, max={values.max():.9g}; "
            f"min condition={minimum['condition_id']}; "
            f"max condition={maximum['condition_id']}."
        )
    review_by_key = {(row["task"], row["condition_id"]): row for row in review}
    lines.extend(["", "## Best versus current review", ""])
    for task in ACTIVE_RULES:
        task_best = [row for row in best if row["task"] == task]
        differences = np.asarray(
            [
                float(review_by_key[(task, row["condition_id"])]["movement_mean_euclidean_m"])
                - float(row["movement_mean_euclidean_m"])
                for row in task_best
            ]
        )
        if len(differences) == 1:
            summary = f"{differences[0]:+.9g} m"
        else:
            summary = (
                f"min={differences.min():+.9g}, median={np.median(differences):+.9g}, "
                f"mean={differences.mean():+.9g}, max={differences.max():+.9g} m"
            )
        lines.append(f"- {task}: review-best movement mean error {summary}.")
    lines.extend(
        [
            "",
            (
                "Shape agreement versus under-travel is intentionally not "
                "auto-classified: the target/actual plots and path ratios are "
                "provided for manual review, without a post-hoc threshold or "
                "automatic interval change."
            ),
            "",
            "The shared eight-stroke-plus-move model has not been run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def require_approved_stage0(
    generated_directory: str | Path,
    approved_directory: str | Path,
) -> dict[str, str]:
    generated = Path(generated_directory)
    approved = Path(approved_directory)
    if not approved.is_dir():
        raise FileNotFoundError("approved canonical Stage-0 directory is missing")
    approved_hashes = {}
    for name in STAGE0_ARTIFACT_NAMES:
        generated_path = generated / name
        approved_path = approved / name
        if not generated_path.is_file() or not approved_path.is_file():
            raise FileNotFoundError(f"canonical Stage-0 artifact is missing: {name}")
        generated_hash = _sha256_file(generated_path)
        approved_hash = _sha256_file(approved_path)
        if generated_hash != approved_hash:
            raise RuntimeError(f"approved canonical Stage-0 artifact differs: {name}")
        approved_hashes[name] = approved_hash
    return approved_hashes


def prepare_canonical_single_task_overfit(
    config_path: str | Path,
    approved_stage0_directory: str | Path,
) -> dict[str, Any]:
    config, geometry = load_canonical_overfit_config(config_path)
    validate_frozen_shared_config(
        "configurations/hanzi_stroke_temporal_composition_canonical_shared_9task_v1.json"
    )
    output = Path(config["output"]["directory"])
    stage0 = write_stage0_artifacts(geometry, output)
    approved_stage0_sha256 = require_approved_stage0(
        output, approved_stage0_directory
    )
    model_root = output / "models"
    model_root.mkdir(exist_ok=False)
    return {
        "output_directory": str(output),
        "stage0": stage0,
        "approved_stage0_sha256": approved_stage0_sha256,
        "initial_review_updates": config["training"]["initial_review_updates"],
        "parallel_task_count": len(_task_conditions(geometry)),
    }


def train_canonical_task_to_review(
    config_path: str | Path,
    task: str,
    target_updates: int,
) -> dict[str, Any]:
    config, geometry = load_canonical_overfit_config(config_path)
    validate_frozen_shared_config(
        "configurations/hanzi_stroke_temporal_composition_canonical_shared_9task_v1.json"
    )
    task_conditions = dict(_task_conditions(geometry))
    if task not in task_conditions:
        raise ValueError(f"unknown canonical single task: {task}")
    output = Path(config["output"]["directory"])
    for name in STAGE0_ARTIFACT_NAMES:
        if not (output / name).is_file():
            raise FileNotFoundError(f"prepared canonical Stage-0 artifact is missing: {name}")
    model_root = output / "models"
    if not model_root.is_dir():
        raise FileNotFoundError("prepared canonical model root is missing")
    return _train_task(
        task,
        task_conditions[task],
        config,
        geometry,
        model_root / task,
        target_updates=target_updates,
    )


def _task_summary_from_artifacts(task: str, output: Path) -> dict[str, Any]:
    best = torch.load(
        output / "best_checkpoint.pt",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    review = torch.load(
        output / "review_checkpoint.pt",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    continuation = torch.load(
        output / "continuation_checkpoint.pt",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    checkpoints = (best, review, continuation)
    if (
        best.get("checkpoint_kind") != "canonical_single_task_best"
        or review.get("checkpoint_kind") != "canonical_single_task_review"
        or continuation.get("checkpoint_kind")
        != "canonical_single_task_continuation"
        or any(value.get("project") != PROJECT for value in checkpoints)
        or any(value.get("variant") != CANONICAL_VARIANT for value in checkpoints)
        or any(value.get("task") != task for value in checkpoints)
        or review.get("update") != continuation.get("update")
        or (continuation.get("continuation_state") or {}).get("next_update")
        != review.get("update", -2) + 1
    ):
        raise ValueError(f"canonical task checkpoint identities differ: {task}")
    validation_rows = []
    with (output / "validation_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("validation_kind") == "review_read_only":
                validation_rows.append(row)
    read_only_checks = (
        validation_rows[-1].get("read_only_checks", {})
        if validation_rows
        else {}
    )
    if (
        not validation_rows
        or validation_rows[-1].get("update") != review.get("update")
        or not isinstance(read_only_checks, dict)
        or not read_only_checks
        or not all(read_only_checks.values())
    ):
        raise ValueError(f"canonical review validation evidence differs: {task}")
    return {
        "task": task,
        "best_update": int(best["update"]),
        "best_validation_loss": float(best["validation_loss"]),
        "review_update": int(review["update"]),
        "review_validation_loss": float(review["validation_loss"]),
        "review_validation_read_only_checks": read_only_checks,
        "completed_updates": int(review["update"]) + 1,
    }


def finalize_canonical_single_task_review(
    config_path: str | Path,
    approved_stage0_directory: str | Path,
) -> dict[str, Any]:
    config, geometry = load_canonical_overfit_config(config_path)
    validate_frozen_shared_config(
        "configurations/hanzi_stroke_temporal_composition_canonical_shared_9task_v1.json"
    )
    output = Path(config["output"]["directory"])
    approved_stage0_sha256 = require_approved_stage0(
        output, approved_stage0_directory
    )
    model_root = output / "models"
    task_summaries = []
    task_conditions = _task_conditions(geometry)
    for task, conditions in task_conditions:
        task_summaries.append(_task_summary_from_artifacts(task, model_root / task))
    expected_model_files = {
        "best_checkpoint.pt",
        "continuation_checkpoint.pt",
        "review_checkpoint.pt",
        "training_metrics.jsonl",
        "validation_metrics.jsonl",
    }
    for task, _ in task_conditions:
        task_output = model_root / task
        if {path.name for path in task_output.iterdir()} != expected_model_files:
            raise RuntimeError(f"canonical task output set differs: {task}")
    curves = []
    for task, conditions in task_conditions:
        for checkpoint_name in ("best", "review"):
            curves.extend(
                _checkpoint_curves(
                    task,
                    conditions,
                    checkpoint_name,
                    model_root / task / f"{checkpoint_name}_checkpoint.pt",
                    config,
                    geometry,
                )
            )
    rows = [curve["row"] for curve in curves]
    combinations = {
        (row["checkpoint"], row["condition_id"]) for row in rows
    }
    if len(rows) != EXPECTED_METRIC_ROWS or len(combinations) != EXPECTED_METRIC_ROWS:
        raise RuntimeError("canonical metric rows are missing or duplicated")
    numeric = [
        float(value)
        for row in rows
        for name, value in row.items()
        if name in METRIC_NAMES and value is not None
    ]
    if not np.isfinite(numeric).all():
        raise RuntimeError("canonical diagnostic metrics contain NaN or Inf")
    _write_metrics(output / "single_task_overfit_metrics.csv", rows)
    arrays = _best_trajectory_arrays(curves)
    np.savez_compressed(output / "single_task_overfit_trajectories.npz", **arrays)
    plots = _write_plots(output, curves)
    _write_report(output / "SINGLE_TASK_OVERFIT_REPORT.md", rows)
    checkpoint_sha256 = {
        str(path.relative_to(output)): _sha256_file(path)
        for path in sorted(model_root.glob("*/*_checkpoint.pt"))
    }
    if len(checkpoint_sha256) != 27:
        raise RuntimeError("canonical checkpoint count must equal 27")
    stage0 = {
        "manifest": _load_json(output / "canonical_condition_manifest.json"),
        "target_rows": EXPECTED_TARGET_ROWS,
        "artifacts": [
            str(output / name) for name in sorted(STAGE0_ARTIFACT_NAMES)
        ],
    }
    provenance = {
        "project": PROJECT,
        "variant": CANONICAL_VARIANT,
        "timing_mode": CANONICAL_TIMING_MODE,
        "geometry_variant": CANONICAL_GEOMETRY_VARIANT,
        "seed": config["seed"],
        "validation_seed": config["validation_seed"],
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "geometry_config_sha256": _sha256_file(config["geometry_config"]),
        "approved_stage0_directory": str(Path(approved_stage0_directory)),
        "approved_stage0_sha256": approved_stage0_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "stage0": stage0,
        "task_summaries": task_summaries,
        "initial_review_updates": config["training"]["initial_review_updates"],
        "update_cap": None,
        "execution_mode": "nine_independent_parallel_processes",
        "integrity": {
            "single_tasks_completed": len(task_summaries),
            "metrics_rows": len(rows),
            "best_trajectory_rows": len(arrays["sample_index"]),
            "plots": len(plots),
            "all_numeric_values_finite": True,
        },
        "shared_9task_started": False,
        "formal_75k_started": False,
        "complete_character_rollout_started": False,
        "behavioral_pass_fail_defined": False,
        "completed": True,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--task", choices=ACTIVE_RULES)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--approved-stage0")
    parser.add_argument("--target-updates", type=int)
    arguments = parser.parse_args()
    if arguments.prepare:
        if arguments.approved_stage0 is None or arguments.target_updates is not None:
            parser.error("--prepare requires --approved-stage0 and forbids --target-updates")
        result = prepare_canonical_single_task_overfit(
            arguments.config, arguments.approved_stage0
        )
    elif arguments.task is not None:
        if arguments.approved_stage0 is not None or arguments.target_updates is None:
            parser.error("--task requires --target-updates and forbids --approved-stage0")
        result = train_canonical_task_to_review(
            arguments.config, arguments.task, arguments.target_updates
        )
    else:
        if arguments.approved_stage0 is None or arguments.target_updates is not None:
            parser.error("--finalize requires --approved-stage0 and forbids --target-updates")
        result = finalize_canonical_single_task_review(
            arguments.config, arguments.approved_stage0
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
