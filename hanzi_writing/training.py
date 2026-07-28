"""Shared-model training and frozen validation for Hanzi temporal composition."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
import torch

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.envs import HanziCharacterEnv, HanziComponentEnv
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    PROJECT,
    SPEED_NAMES,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    checkpoint_stroke_groups,
    load_geometry_config,
    move_conditions,
    training_conditions_by_rule,
)
from losses import (
    detached_position_metrics,
    l1_muscle_act,
    l1_rate,
    l1_weight,
    position_l1_metrics,
    simple_dynamics,
)
from train import _build_policy, _checkpoint_payload, _fixed_rng


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True))
        handle.write("\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("protocol configuration must be a JSON object")
    return value


def _require_cpu(config: dict[str, Any]) -> None:
    if config["device"] != "cpu":
        raise ValueError("the accepted project-2 baseline device is cpu")


def _validate_shared_model(config: dict[str, Any]) -> None:
    model = config["model"]
    if model != {
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
    }:
        raise ValueError("model does not match the accepted project-2 CPU baseline")


def validate_training_config(config: dict[str, Any]) -> GeometryConfig:
    if config.get("project") != PROJECT or config.get("run_kind") != "training":
        raise ValueError("training requires the Hanzi project and run_kind=training")
    if config.get("variant") != "shared9_dev42":
        raise ValueError("training variant must be shared9_dev42")
    _require_cpu(config)
    if config.get("seed") != 42 or config.get("validation_seed") != 1042:
        raise ValueError("training and validation seeds must be 42 and 1042")
    if tuple(config.get("active_rules", ())) != ACTIVE_RULES:
        raise ValueError("active_rules must contain the accepted nine-rule mapping")
    if config.get("rule_dim_total") != 10 or config.get("unused_rule_index") != 9:
        raise ValueError("the 10-column rule contract is invalid")
    _validate_shared_model(config)
    if config["optimizer"] != {
        "name": "Adam",
        "learning_rate": 0.001,
        "grad_clip_norm": 1.0,
    }:
        raise ValueError("optimizer differs from the accepted baseline")
    if config["training"] != {
        "batch_size": 32,
        "max_updates": 75000,
        "validation_interval": 500,
        "log_interval": 100,
        "sampler": "uniform_rule_then_uniform_condition_speed_delay",
    }:
        raise ValueError("training schedule or sampler differs from the accepted protocol")
    if config["position_loss"] != {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }:
        raise ValueError("position loss differs from the reused project-2 loss")
    if config["regularization"] != {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }:
        raise ValueError("regularization differs from the accepted baseline")
    geometry = load_geometry_config(config["geometry_config"])
    if geometry.validation_seed != config["validation_seed"]:
        raise ValueError("training and geometry validation seeds disagree")
    return geometry


def _hp_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    optimizer = config["optimizer"]
    training = config["training"]
    regularization = config["regularization"]
    return {
        "network": model["network"],
        "inp_size": model["input_size"],
        "hid_size": model["hidden_size"],
        "activation_name": model["activation"],
        "noise_level_act": model["recurrent_noise_std"],
        "noise_level_inp": model["input_noise_std"],
        "constrained": model["constrained"],
        "dt": model["rnn_dt_ms"],
        "t_const": model["rnn_tau_ms"],
        "batch_first": model["batch_first"],
        "lr": optimizer["learning_rate"],
        "grad_clip_norm": optimizer["grad_clip_norm"],
        "batch_size": training["batch_size"],
        "epochs": training["max_updates"],
        "save_iter": training["validation_interval"],
        **regularization,
        "seed": config["seed"],
        "validation_seed": config["validation_seed"],
        "variant": config["variant"],
        "project": PROJECT,
        "protocol_config": config,
    }


def _make_effector():
    return mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())


def _rollout(
    policy,
    env: HanziComponentEnv,
    hp: dict[str, Any],
    conditions: Iterable[StrokeCondition | MoveCondition],
    speed_name: str,
    delay_steps: int,
    *,
    network_noise: bool,
    deterministic_observation: bool,
    track_gradients: bool,
) -> dict[str, Any]:
    condition_tuple = tuple(conditions)
    batch_size = len(condition_tuple)
    x = torch.zeros((batch_size, hp["hid_size"]), device=torch.device("cpu"))
    h = torch.zeros_like(x)
    obs, info = env.reset(
        options={
            "conditions": condition_tuple,
            "speed_name": speed_name,
            "delay_steps": delay_steps,
            "deterministic": deterministic_observation,
        }
    )
    xy = []
    targets = []
    muscles = [info["states"]["muscle"][:, 0].unsqueeze(1)]
    hidden = [h.unsqueeze(1)]
    timestep = 0
    terminated = False
    context = torch.enable_grad() if track_gradients else torch.no_grad()
    with context:
        while not terminated:
            x, h, action = policy(obs, x, h, noise=network_noise)
            obs, _, terminated, info = env.step(timestep, action=action)
            xy.append(info["states"]["fingertip"][:, None, :])
            targets.append(info["goal"][:, None, :])
            muscles.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hidden.append(h.unsqueeze(1))
            timestep += 1
    return {
        "xy": torch.cat(xy, dim=1),
        "target": torch.cat(targets, dim=1),
        "muscle": torch.cat(muscles, dim=1),
        "hidden": torch.cat(hidden, dim=1),
        "epoch_bounds": env.epoch_bounds,
        "timesteps": timestep,
    }


def checkpoint_validation(policy, hp: dict[str, Any], geometry: GeometryConfig) -> dict[str, Any]:
    """Run the fixed 15-stroke/12-move, three-speed, exact-start grid."""

    rule_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=hp["protocol_config"]["geometry_config"],
        action_frame_stacking=0,
    )
    with _fixed_rng(geometry.validation_seed):
        for group in checkpoint_stroke_groups(geometry):
            for speed_name in SPEED_NAMES:
                result = _rollout(
                    policy,
                    env,
                    hp,
                    group,
                    speed_name,
                    geometry.validation_delay_steps,
                    network_noise=geometry.validation_network_noise,
                    deterministic_observation=False,
                    track_gradients=False,
                )
                metrics = detached_position_metrics(
                    position_l1_metrics(
                        result["xy"], result["target"], result["epoch_bounds"]
                    )
                )
                rule_rows[group[0].rule].append(metrics)
        for condition in move_conditions(geometry, include_jitter=False):
            for speed_name in SPEED_NAMES:
                result = _rollout(
                    policy,
                    env,
                    hp,
                    (condition,),
                    speed_name,
                    geometry.validation_delay_steps,
                    network_noise=geometry.validation_network_noise,
                    deterministic_observation=False,
                    track_gradients=False,
                )
                metrics = detached_position_metrics(
                    position_l1_metrics(
                        result["xy"], result["target"], result["epoch_bounds"]
                    )
                )
                rule_rows["move"].append(metrics)
    if set(rule_rows) != set(ACTIVE_RULES):
        raise RuntimeError("checkpoint grid did not cover all nine active rules")
    per_rule = {
        rule: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for rule, rows in rule_rows.items()
    }
    aggregate = {
        key: float(np.mean([per_rule[rule][key] for rule in ACTIVE_RULES]))
        for key in next(iter(per_rule.values()))
    }
    return {
        "aggregation": "equal_mean_over_9_rules",
        "aggregate": aggregate,
        "per_rule": per_rule,
        "rollout_group_count_by_rule": {
            rule: len(rule_rows[rule]) for rule in ACTIVE_RULES
        },
    }


def _condition_manifest(geometry: GeometryConfig) -> dict[str, Any]:
    grouped = training_conditions_by_rule(geometry)
    return {
        "sampler": "uniform_rule_then_uniform_condition_speed_delay",
        "active_rules_in_sampling_order": list(ACTIVE_RULES),
        "condition_count_by_rule": {rule: len(grouped[rule]) for rule in ACTIVE_RULES},
        "speed_names": list(SPEED_NAMES),
        "delay_steps": list(geometry.delay_steps),
        "stroke_jitter": {
            "distribution": "uniform_per_axis",
            "fraction_of_global_character_span": geometry.stroke_jitter_fraction,
            "copies_per_base_start": geometry.stroke_jitter_copies,
            "seed": geometry.stroke_jitter_seed,
        },
        "move_jitter_from_geometry_authority": {
            "distribution": "uniform_per_axis_for_start_and_goal",
            "fraction_of_global_character_span": authority.MOVE_JITTER_FRACTION,
            "copies_per_exact_transition": authority.MOVE_JITTER_COPIES,
            "seed": authority.RANDOM_SEED,
        },
        "checkpoint_validation": {
            "rollout_group_count": len(checkpoint_stroke_groups(geometry)) * 3
            + len(move_conditions(geometry, include_jitter=False)) * 3,
            "delay_steps": geometry.validation_delay_steps,
            "validation_seed": geometry.validation_seed,
            "network_noise": geometry.validation_network_noise,
            "aggregation": "equal_mean_over_9_rules",
            "includes_jitter": False,
            "includes_complete_characters": False,
        },
    }


def train_hanzi_shared_model(config: dict[str, Any]) -> dict[str, Any]:
    geometry = validate_training_config(config)
    hp = _hp_from_config(config)
    output = config["output"]
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    with Path(config["geometry_config"]).open("r", encoding="utf-8") as handle:
        _write_json(output_dir / "resolved_geometry_config.json", json.load(handle))
    _write_json(output_dir / "condition_manifest.json", _condition_manifest(geometry))

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    policy = _build_policy(hp, config["model"]["output_size"], torch.device("cpu"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    conditions_by_rule = training_conditions_by_rule(geometry)
    losses: list[float] = []
    best_validation = np.inf
    last_validation: float | None = None

    for update in range(hp["epochs"]):
        rule = random.choice(ACTIVE_RULES)
        condition = random.choice(conditions_by_rule[rule])
        speed_name = random.choice(SPEED_NAMES)
        delay_steps = random.choice(geometry.delay_steps)
        result = _rollout(
            policy,
            env,
            hp,
            (condition,) * hp["batch_size"],
            speed_name,
            delay_steps,
            network_noise=True,
            deterministic_observation=False,
            track_gradients=True,
        )
        position_metrics = position_l1_metrics(
            result["xy"], result["target"], result["epoch_bounds"]
        )
        loss = position_metrics["phase_normalized_position_l1"]
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

        if update and update % config["training"]["log_interval"] == 0:
            interval = config["training"]["log_interval"]
            row = {
                "update": update,
                "mean_total_loss": float(np.mean(losses[-interval:])),
                "sampled_rule": rule,
                "sampled_condition": condition.condition_id,
                "sampled_speed": speed_name,
                "sampled_delay_steps": delay_steps,
                **detached_position_metrics(position_metrics),
            }
            _append_jsonl(output_dir / "training_metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)

        if update % hp["save_iter"] == 0:
            validation = checkpoint_validation(policy, hp, geometry)
            last_validation = validation["aggregate"]["phase_normalized_position_l1"]
            _append_jsonl(
                output_dir / "checkpoint_validation_metrics.jsonl",
                {"update": update, **validation},
            )
            if last_validation <= best_validation:
                best_validation = last_validation
                payload = _checkpoint_payload(
                    policy, optimizer, hp, update, last_validation
                )
                payload.update({"project": PROJECT, "checkpoint_kind": "best_shared9"})
                torch.save(payload, output_dir / output["best_checkpoint"])

    payload = _checkpoint_payload(
        policy, optimizer, hp, hp["epochs"] - 1, last_validation
    )
    payload.update({"project": PROJECT, "checkpoint_kind": "final_continuation"})
    torch.save(payload, output_dir / output["final_checkpoint"])
    summary = {
        "project": PROJECT,
        "variant": config["variant"],
        "seed": config["seed"],
        "updates": hp["epochs"],
        "best_validation_loss": float(best_validation),
        "last_validation_loss": last_validation,
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


def load_hanzi_policy_checkpoint(checkpoint_path: str | Path):
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    if checkpoint.get("project") != PROJECT:
        raise ValueError("checkpoint does not belong to the Hanzi project")
    if checkpoint.get("variant") != "shared9_dev42":
        raise ValueError("checkpoint is not the shared nine-rule model")
    hp = checkpoint.get("hp")
    if not isinstance(hp, dict):
        raise ValueError("checkpoint does not contain hyperparameters")
    policy = _build_policy(hp, 6, torch.device("cpu"))
    policy.load_state_dict(checkpoint["agent_state_dict"])
    return policy, checkpoint


def _state_clone(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(before: dict[str, torch.Tensor], module) -> None:
    after = module.state_dict()
    for name, value in before.items():
        if not torch.equal(value, after[name]):
            raise RuntimeError(f"frozen policy state changed: {name}")


def _character_rollout(
    policy,
    env: HanziCharacterEnv,
    hp: dict[str, Any],
    character_name: str,
    speed_name: str,
) -> dict[str, Any]:
    obs, _ = env.reset(
        testing=True,
        options={
            "character": character_name,
            "speed_name": speed_name,
            "deterministic": True,
        },
    )
    x = torch.zeros((1, hp["hid_size"]), dtype=torch.float32)
    h = torch.zeros_like(x)
    xy = []
    targets = []
    timestep = 0
    terminated = False
    with torch.no_grad():
        while not terminated:
            x, h, action = policy(obs, x, h, noise=False)
            obs, _, terminated, info = env.step(timestep, action=action)
            xy.append(info["states"]["fingertip"][0].detach().cpu().numpy())
            targets.append(info["goal"][0].detach().cpu().numpy())
            timestep += 1
    return {
        "actual": np.asarray(xy),
        "target": np.asarray(targets),
        "writing_mask": env.writing_mask.detach().cpu().numpy(),
        "segments": env.segments,
        "phase": env.phase,
        "episode_terminated_exactly": timestep == env.max_ep_duration + 1,
    }


def _l1_rows(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(actual - target).sum(axis=1)


def _segment_metrics(result: dict[str, Any], phase_prefix: str) -> list[dict[str, Any]]:
    error = _l1_rows(result["actual"], result["target"])
    rows = []
    for segment in result["segments"]:
        if not str(segment["phase"]).startswith(phase_prefix):
            continue
        start = int(segment["start_index"])
        end = int(segment["end_index_exclusive"])
        rows.append(
            {
                "phase": segment["phase"],
                "rule": segment["rule"],
                "mean_l1_m": float(error[start:end].mean()),
                "endpoint_l1_m": float(error[end - 1]),
            }
        )
    return rows


def _plot_character(path: Path, name: str, result: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    target = result["target"]
    actual = result["actual"]
    axis.plot(target[:, 0], target[:, 1], color="0.7", linewidth=1.3, label="target")
    writing_label = "writing"
    move_label = "pen-up move"
    for segment in result["segments"]:
        phase = str(segment["phase"])
        if not (phase.startswith("stroke_movement_") or phase.startswith("move_movement_")):
            continue
        start = int(segment["start_index"])
        end = int(segment["end_index_exclusive"])
        writing = phase.startswith("stroke_movement_")
        axis.plot(
            actual[start:end, 0],
            actual[start:end, 1],
            color="#e45756" if writing else "#4c78a8",
            linewidth=1.5 if writing else 1.0,
            linestyle="-" if writing else "--",
            label=writing_label if writing else move_label,
        )
        if writing:
            writing_label = None
        else:
            move_label = None
    axis.set_title(name)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_frozen_characters(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("project") != PROJECT or config.get("run_kind") != "character_validation":
        raise ValueError("expected a Hanzi character-validation configuration")
    _require_cpu(config)
    validation = config["character_validation"]
    if validation != {
        "characters": ["mu", "jiang", "ke"],
        "speed": "medium",
        "prepare_steps": 25,
        "final_hold_steps": 25,
        "freeze_network": True,
        "writing_mask_enabled": True,
        "network_noise": False,
    }:
        raise ValueError("character validation differs from the accepted frozen protocol")
    geometry = load_geometry_config(config["geometry_config"])
    policy, checkpoint = load_hanzi_policy_checkpoint(config["source_checkpoint"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    before = _state_clone(policy)
    output_dir = Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    hp = checkpoint["hp"]
    reports = {}
    for name in validation["characters"]:
        env = HanziCharacterEnv(
            effector=_make_effector(),
            geometry_config_path=config["geometry_config"],
            action_frame_stacking=0,
        )
        result = _character_rollout(policy, env, hp, name, validation["speed"])
        error = _l1_rows(result["actual"], result["target"])
        writing_mask = result["writing_mask"]
        hold_start = len(error) - geometry.hold_steps
        move_mask = np.asarray(
            [phase.startswith("move_movement_") for phase in result["phase"]],
            dtype=bool,
        )
        hold_drift = np.abs(
            result["actual"][hold_start:] - result["actual"][hold_start - 1]
        ).sum(axis=1)
        reports[name] = {
            "all_values_finite": bool(
                np.isfinite(result["actual"]).all()
                and np.isfinite(result["target"]).all()
                and np.isfinite(error).all()
                and np.isfinite(hold_drift).all()
            ),
            "writing_only_mean_l1_m": float(error[writing_mask].mean()),
            "stroke_metrics": _segment_metrics(result, "stroke_movement_"),
            "move_metrics": _segment_metrics(result, "move_movement_"),
            "move_only_mean_l1_m": float(error[move_mask].mean()),
            "final_hold_target_mean_l1_m": float(error[hold_start:].mean()),
            "final_hold_drift_mean_l1_m": float(hold_drift.mean()),
            "final_hold_drift_endpoint_l1_m": float(hold_drift[-1]),
            "timesteps": len(error),
            "episode_terminated_exactly": result["episode_terminated_exactly"],
            "segments": list(result["segments"]),
        }
        _plot_character(output_dir / f"{name}_frozen_rollout.png", name, result)
    _assert_state_equal(before, policy)
    report = {
        "project": PROJECT,
        "checkpoint": str(config["source_checkpoint"]),
        "network_noise": False,
        "optimizer_created": False,
        "motornet_optimizer_created": False,
        "motornet_parameter_updates": 0,
        "policy_requires_grad_after_freeze": any(
            parameter.requires_grad for parameter in policy.parameters()
        ),
        "policy_state_bitwise_unchanged": True,
        "characters": reports,
        "overall_passed": all(
            row["all_values_finite"] and row["episode_terminated_exactly"]
            for row in reports.values()
        ),
    }
    _write_json(output_dir / "frozen_character_validation.json", report)
    return report


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("run_kind") == "training":
        return train_hanzi_shared_model(config)
    if config.get("run_kind") == "character_validation":
        return validate_frozen_characters(config)
    raise ValueError(f"unsupported Hanzi run_kind: {config.get('run_kind')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_config(_load_json(args.config))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
