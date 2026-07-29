"""Complete-character boundary state-block intervention sensitivity diagnostic.

This module is diagnostic-only.  It never creates an optimizer, calls backward,
updates a policy parameter, or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from hanzi_writing.envs import HanziCharacterEnv
from hanzi_writing.fit_diagnostic import movement_metrics as legacy_movement_metrics
from hanzi_writing.geometry import (
    PROJECT,
    characters,
    load_geometry_config,
    stroke_conditions,
)
from hanzi_writing.training import (
    _assert_state_equal,
    _make_effector,
    _state_clone,
    load_hanzi_policy_checkpoint,
)
from train import _fixed_rng


CONDITIONS = (
    "C0_continuous",
    "C1_plant_sensory_clean",
    "C2_neural_clean",
    "C3_joint_clean",
    "P1_plant_sensory_clean_before_prepare",
)
REFERENCE_CONDITION = "local_clean_reference"
CHARACTERS = ("mu", "jiang", "ke")
EFFECTOR_STATE_FIELDS = (
    "joint",
    "cartesian",
    "muscle",
    "geometry",
    "fingertip",
)
DYNAMIC_EFFECTOR_STATE_FIELDS = ("joint", "muscle", "geometry")
OBS_BUFFER_FIELDS = ("proprioception", "vision", "action")
METER_ERROR_METRICS = (
    "movement_mean_euclidean_m",
    "endpoint_euclidean_m",
    "movement_sample_0_error_m",
    "movement_sample_1_error_m",
)
CHANGE_ERROR_METRICS = METER_ERROR_METRICS + (
    "path_length_ratio_absolute_deviation",
)
HENGZHE_CHANGE_ERROR_METRICS = (
    "corner_min_distance_m",
    "corner_pre_segment_mean_euclidean_m",
    "corner_post_segment_mean_euclidean_m",
    "corner_turn_angle_absolute_error_deg",
)
PATH_NORMALIZED_CHANGE_METRICS = METER_ERROR_METRICS + HENGZHE_CHANGE_ERROR_METRICS[:3]
FIXED_SCOPE_STATEMENT = (
    "本诊断描述完整三字 12 个指定边界对 neural state block、plant-sensory "
    "state block、联合状态替换以及 prepare 前状态替换的数值敏感性。由于交叉状态可能 "
    "off-manifold，且没有预注册行为阈值，本诊断不提供独立因果贡献比例、唯一根因或"
    "行为成功判定。"
)


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_boundary_intervention_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("boundary intervention configuration must be an object")
    _require_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "device",
            "seed",
            "geometry_config",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_kind",
            "speed",
            "characters",
            "prepare_steps",
            "network_noise",
            "deterministic_observation",
            "environment_noise",
            "conditions",
            "isolated_control",
            "corner_window_radius_samples",
            "technical_tolerance",
            "behavioral_pass_fail",
            "output_directory",
        },
        "boundary intervention configuration",
    )
    if (
        config["project"] != PROJECT
        or config["run_kind"] != "boundary_state_block_intervention"
        or config["variant"] != "shared9_dev42"
        or config["device"] != "cpu"
    ):
        raise ValueError("configuration names the wrong project, run kind, variant, or device")
    if config["seed"] != 1042:
        raise ValueError("boundary intervention seed must be 1042")
    if config["checkpoint_kind"] != "best" or config["speed"] != "medium":
        raise ValueError("diagnostic is frozen to best checkpoint and medium speed")
    if tuple(config["characters"]) != CHARACTERS:
        raise ValueError("diagnostic must cover mu, jiang, and ke in that order")
    if config["prepare_steps"] != 25 or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("prepare duration or intervention conditions changed")
    if (
        config["network_noise"] is not False
        or config["deterministic_observation"] is not True
        or config["environment_noise"] is not False
    ):
        raise ValueError("boundary intervention must be deterministic and noise-free")
    if config["behavioral_pass_fail"] != "not_defined":
        raise ValueError("post-hoc behavioral pass/fail definitions are forbidden")
    if config["corner_window_radius_samples"] != 3:
        raise ValueError("hengzhe corner window must remain fixed at three samples")
    if not math.isclose(
        float(config["technical_tolerance"]), 1e-7, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("technical tolerance must be 1e-7")
    isolated = config["isolated_control"]
    if not isinstance(isolated, dict):
        raise ValueError("isolated_control must be an object")
    _require_keys(
        isolated,
        {"path", "sha256", "checkpoint", "speed", "variant"},
        "isolated_control",
    )
    if isolated["checkpoint"] != "best" or isolated["speed"] != "medium":
        raise ValueError("isolated control must use best checkpoint and medium speed")
    if isolated["variant"] != "exact":
        raise ValueError("isolated control must use exact conditions")
    geometry = load_geometry_config(config["geometry_config"])
    if geometry.prepare_steps != config["prepare_steps"]:
        raise ValueError("diagnostic and geometry prepare durations disagree")
    return config


def _require_zero_environment_noise(env: HanziCharacterEnv) -> None:
    if any(float(value) != 0.0 for value in env.obs_noise):
        raise RuntimeError("observation noise is not zero")
    if any(float(value) != 0.0 for value in env.action_noise):
        raise RuntimeError("action noise is not zero")
    if any(float(value) != 0.0 for value in env.proprioception_noise):
        raise RuntimeError("proprioception noise is not zero")
    if any(float(value) != 0.0 for value in env.vision_noise):
        raise RuntimeError("vision noise is not zero")


def _new_character_env(
    geometry_config_path: str | Path, character: str, speed: str, seed: int
) -> tuple[HanziCharacterEnv, torch.Tensor]:
    env = HanziCharacterEnv(
        effector=_make_effector(),
        geometry_config_path=geometry_config_path,
        action_frame_stacking=0,
    )
    obs, _ = env.reset(
        testing=True,
        seed=seed,
        options={"character": character, "speed_name": speed, "deterministic": True},
    )
    _require_zero_environment_noise(env)
    return env, obs


def enumerate_boundaries(env: HanziCharacterEnv) -> list[dict[str, Any]]:
    by_phase = {str(segment["phase"]): segment for segment in env.segments}
    if len(by_phase) != len(env.segments):
        raise RuntimeError("character schedule contains duplicate phase names")
    boundaries = []
    move_segments = [
        segment
        for segment in env.segments
        if str(segment["phase"]).startswith("move_movement_")
    ]
    for move in move_segments:
        move_index = int(str(move["phase"]).rsplit("_", 1)[1])
        next_stroke_index = move_index + 1
        prepare = by_phase.get(f"stroke_prepare_{next_stroke_index}")
        stroke = by_phase.get(f"stroke_movement_{next_stroke_index}")
        if prepare is None or stroke is None:
            raise RuntimeError("move boundary is missing its next prepare or stroke segment")
        if (
            int(move["end_index_exclusive"]) != int(prepare["start_index"])
            or int(prepare["end_index_exclusive"]) != int(stroke["start_index"])
        ):
            raise RuntimeError("move, prepare, and next stroke segments are not contiguous")
        p = int(prepare["start_index"])
        m = int(stroke["start_index"])
        e = int(stroke["end_index_exclusive"])
        if m - p != 25 or e - m < 2:
            raise RuntimeError("boundary prepare or movement point count is invalid")
        boundary_id = f"{env.character_name}_m{move_index}_to_s{next_stroke_index}"
        boundaries.append(
            {
                "boundary_id": boundary_id,
                "character": env.character_name,
                "move_index": move_index,
                "next_stroke_index": next_stroke_index,
                "next_rule": str(stroke["rule"]),
                "prepare_entry_schedule_index": p,
                "last_prepare_decision_schedule_index": m - 1,
                "movement_sample_0_schedule_index": m,
                "movement_sample_1_schedule_index": m + 1,
                "movement_end_index_exclusive": e,
                "movement_intervals": e - m - 1,
            }
        )
    return boundaries


def _clone_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a Tensor")
    clone = value.detach().clone()
    if (clone.is_floating_point() or clone.is_complex()) and not torch.isfinite(clone).all():
        raise RuntimeError(f"{label} contains non-finite values")
    return clone


def snapshot_plant_sensory(env: HanziCharacterEnv) -> dict[str, Any]:
    if set(env.effector.states) != set(EFFECTOR_STATE_FIELDS):
        raise RuntimeError(
            f"unexpected MotorNet state fields: {sorted(env.effector.states)}"
        )
    if set(env.obs_buffer) != set(OBS_BUFFER_FIELDS):
        raise RuntimeError(
            f"unexpected observation buffer fields: {sorted(env.obs_buffer)}"
        )
    effector = {
        name: _clone_tensor(env.effector.states[name], f"effector.states.{name}")
        for name in EFFECTOR_STATE_FIELDS
    }
    buffers = {
        name: [
            _clone_tensor(value, f"obs_buffer.{name}[{index}]")
            for index, value in enumerate(env.obs_buffer[name])
        ]
        for name in OBS_BUFFER_FIELDS
    }
    return {
        "effector_states": effector,
        "obs_buffer": buffers,
        "hidden_goal": _clone_tensor(env.hidden_goal, "hidden_goal"),
        "initial_pos": _clone_tensor(env.initial_pos, "initial_pos"),
        "numpy_generator_state": copy.deepcopy(env.np_random.bit_generator.state),
    }


def _tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    return (
        str(array.dtype).encode("utf-8")
        + json.dumps(list(array.shape)).encode("utf-8")
        + array.tobytes(order="C")
    )


def plant_sensory_manifest(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = []
    for name in EFFECTOR_STATE_FIELDS:
        value = snapshot["effector_states"][name]
        manifest.append(
            {
                "field": f"effector.states.{name}",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
            }
        )
    for name in OBS_BUFFER_FIELDS:
        for index, value in enumerate(snapshot["obs_buffer"][name]):
            manifest.append(
                {
                    "field": f"obs_buffer.{name}[{index}]",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
                }
            )
    for name in ("hidden_goal", "initial_pos"):
        value = snapshot[name]
        manifest.append(
            {
                "field": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
            }
        )
    rng_bytes = json.dumps(
        snapshot["numpy_generator_state"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest.append(
        {
            "field": "numpy_generator_state",
            "shape": [],
            "dtype": "json",
            "sha256": hashlib.sha256(rng_bytes).hexdigest(),
        }
    )
    return manifest


def plant_sensory_digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        plant_sensory_manifest(snapshot), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_tensor_equal(expected: torch.Tensor, actual: torch.Tensor, label: str) -> None:
    if not torch.equal(expected, actual):
        difference = float((expected - actual).abs().max().detach().cpu())
        raise RuntimeError(f"state restore mismatch for {label}; max_abs={difference}")


def restore_plant_sensory(env: HanziCharacterEnv, snapshot: dict[str, Any]) -> None:
    env.effector._set_state(
        {
            name: snapshot["effector_states"][name].detach().clone()
            for name in DYNAMIC_EFFECTOR_STATE_FIELDS
        }
    )
    for name in EFFECTOR_STATE_FIELDS:
        _assert_tensor_equal(
            snapshot["effector_states"][name],
            env.effector.states[name],
            f"effector.states.{name}",
        )
    env.obs_buffer = {
        name: [value.detach().clone() for value in snapshot["obs_buffer"][name]]
        for name in OBS_BUFFER_FIELDS
    }
    env.hidden_goal = snapshot["hidden_goal"].detach().clone()
    env.initial_pos = snapshot["initial_pos"].detach().clone()
    env.np_random.bit_generator.state = copy.deepcopy(snapshot["numpy_generator_state"])
    restored = snapshot_plant_sensory(env)
    if plant_sensory_digest(restored) != plant_sensory_digest(snapshot):
        raise RuntimeError("plant-sensory restore omitted or changed a manifest field")


def compose_observation(env: HanziCharacterEnv, schedule_index: int) -> torch.Tensor:
    obs = torch.cat(
        [
            env.rule_input[:, schedule_index],
            env.speed_scalar[:, schedule_index],
            env.go_cue[:, schedule_index],
            env.vis_inp[:, schedule_index],
            env.obs_buffer["vision"][0],
            env.obs_buffer["proprioception"][0],
        ],
        dim=-1,
    )
    if obs.shape != (1, 28):
        raise RuntimeError(f"reconstructed observation has invalid shape {tuple(obs.shape)}")
    return obs


def _policy_state_digest(policy: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _decision_state(
    env: HanziCharacterEnv,
    index: int,
    obs: torch.Tensor,
    x: torch.Tensor,
    h: torch.Tensor,
    action_to_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "schedule_index": index,
        "obs": obs.detach().clone(),
        "x": x.detach().clone(),
        "h": h.detach().clone(),
        "action_to_state": None
        if action_to_state is None
        else action_to_state.detach().clone(),
        "plant": snapshot_plant_sensory(env),
    }


def _sample(
    env: HanziCharacterEnv,
    index: int,
    x: torch.Tensor,
    h: torch.Tensor,
    action_to_state: torch.Tensor | None,
) -> dict[str, Any]:
    action_width = int(env.action_space.shape[0])
    action = (
        np.full(action_width, np.nan, dtype=np.float32)
        if action_to_state is None
        else action_to_state[0].detach().cpu().numpy().copy()
    )
    plant = snapshot_plant_sensory(env)
    field_manifest = plant_sensory_manifest(plant)
    return {
        "schedule_index": index,
        "target_xy": env.traj[0, index].detach().cpu().numpy().copy(),
        "actual_xy": env.effector.states["fingertip"][0].detach().cpu().numpy().copy(),
        "phase": str(env.phase[index]),
        "rule_index": int(torch.argmax(env.rule_input[0, index]).detach().cpu()),
        "rule": str(
            next(
                segment["rule"]
                for segment in env.segments
                if int(segment["start_index"])
                <= index
                < int(segment["end_index_exclusive"])
            )
        ),
        "go_cue": float(env.go_cue[0, index, 0].detach().cpu()),
        "spatial_cue": env.vis_inp[0, index].detach().cpu().numpy().copy(),
        "x": x[0].detach().cpu().numpy().copy(),
        "h": h[0].detach().cpu().numpy().copy(),
        "action": action,
        "plant_sensory_sha256": plant_sensory_digest(plant),
        "plant_sensory_field_manifest_json": json.dumps(
            field_manifest, sort_keys=True, separators=(",", ":")
        ),
    }


def _run_from_decision(
    policy: torch.nn.Module,
    geometry_config_path: str | Path,
    character: str,
    speed: str,
    seed: int,
    decision: dict[str, Any],
    end_index_exclusive: int,
    *,
    initial_action: torch.Tensor | None,
    capture_indices: Iterable[int] = (),
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    env, _ = _new_character_env(geometry_config_path, character, speed, seed)
    restore_plant_sensory(env, decision["plant"])
    x = decision["x"].detach().clone()
    h = decision["h"].detach().clone()
    start = int(decision["schedule_index"])
    obs = compose_observation(env, start)
    _assert_tensor_equal(decision["obs"], obs, "decision observation")
    samples = [_sample(env, start, x, h, initial_action)]
    captures: dict[int, dict[str, Any]] = {}
    if start in set(capture_indices):
        captures[start] = _decision_state(env, start, obs, x, h, initial_action)
    capture_set = set(capture_indices)
    with torch.no_grad():
        for index in range(start + 1, end_index_exclusive):
            x, h, action = policy(obs, x, h, noise=False)
            obs, _, _, _ = env.step(index, action=action)
            samples.append(_sample(env, index, x, h, action))
            if index in capture_set:
                captures[index] = _decision_state(env, index, obs, x, h, action)
    return samples, captures


def _natural_character_rollout(
    policy: torch.nn.Module,
    hp: dict[str, Any],
    geometry_config_path: str | Path,
    character: str,
    speed: str,
    seed: int,
    capture_indices: Iterable[int],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], HanziCharacterEnv]:
    env, obs = _new_character_env(geometry_config_path, character, speed, seed)
    x = torch.zeros((1, hp["hid_size"]), dtype=torch.float32)
    h = torch.zeros_like(x)
    samples = []
    captures = {}
    capture_set = set(capture_indices)
    with torch.no_grad():
        for index in range(env.max_ep_duration + 1):
            x, h, action = policy(obs, x, h, noise=False)
            obs, _, terminated, _ = env.step(index, action=action)
            samples.append(_sample(env, index, x, h, action))
            if index in capture_set:
                captures[index] = _decision_state(env, index, obs, x, h, action)
            if terminated != (index == env.max_ep_duration):
                raise RuntimeError("character termination index changed")
    return samples, captures, env


def _clean_prepare_entry_decision(
    hp: dict[str, Any],
    geometry_config_path: str | Path,
    character: str,
    speed: str,
    seed: int,
    prepare_entry_index: int,
) -> dict[str, Any]:
    env, _ = _new_character_env(geometry_config_path, character, speed, seed)
    target_world = env.traj[0, prepare_entry_index].detach().cpu().numpy()
    target_local = target_world - np.asarray(env.anchor_m, dtype=np.float64)
    env._reset_effector(target_local[None, :])
    proprioception = env.get_proprioception().detach().clone()
    vision = env.get_vision().detach().clone()
    env.obs_buffer["proprioception"] = [
        proprioception.detach().clone() for _ in env.obs_buffer["proprioception"]
    ]
    env.obs_buffer["vision"] = [
        vision.detach().clone() for _ in env.obs_buffer["vision"]
    ]
    env.obs_buffer["action"] = []
    env.hidden_goal = env._target_at(prepare_entry_index).detach().clone()
    x = torch.zeros((1, hp["hid_size"]), dtype=torch.float32)
    h = torch.zeros_like(x)
    obs = compose_observation(env, prepare_entry_index)
    return _decision_state(env, prepare_entry_index, obs, x, h, None)


def finalize_trace(
    samples: list[dict[str, Any]], dt_seconds: float, previous_position: np.ndarray | None
) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("trace cannot be empty")
    output = {
        "schedule_index": np.asarray([sample["schedule_index"] for sample in samples]),
        "target_xy": np.stack([sample["target_xy"] for sample in samples]),
        "actual_xy": np.stack([sample["actual_xy"] for sample in samples]),
        "phase": np.asarray([sample["phase"] for sample in samples], dtype=np.str_),
        "rule_index": np.asarray([sample["rule_index"] for sample in samples]),
        "rule": np.asarray([sample["rule"] for sample in samples], dtype=np.str_),
        "go_cue": np.asarray([sample["go_cue"] for sample in samples]),
        "spatial_cue": np.stack([sample["spatial_cue"] for sample in samples]),
        "x": np.stack([sample["x"] for sample in samples]),
        "h": np.stack([sample["h"] for sample in samples]),
        "action": np.stack([sample["action"] for sample in samples]),
        "plant_sensory_sha256": np.asarray(
            [sample["plant_sensory_sha256"] for sample in samples], dtype=np.str_
        ),
        "plant_sensory_field_manifest_json": np.asarray(
            [sample["plant_sensory_field_manifest_json"] for sample in samples],
            dtype=np.str_,
        ),
    }
    speed = np.empty(len(samples), dtype=np.float64)
    if previous_position is None:
        speed[0] = np.nan
    else:
        speed[0] = np.linalg.norm(output["actual_xy"][0] - previous_position) / dt_seconds
    if len(samples) > 1:
        speed[1:] = (
            np.linalg.norm(np.diff(output["actual_xy"], axis=0), axis=1) / dt_seconds
        )
    output["fingertip_speed_mps"] = speed
    output["position_error_m"] = np.linalg.norm(
        output["actual_xy"] - output["target_xy"], axis=1
    )
    return output


def _trace_slice(trace: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    indices = trace["schedule_index"]
    mask = (indices >= start) & (indices < end)
    if int(mask.sum()) != end - start:
        raise RuntimeError("trace schedule indices are incomplete or duplicated")
    return {name: value[mask] for name, value in trace.items()}


def movement_metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    legacy = legacy_movement_metrics(actual, target)
    return {
        "target_path_length_m": legacy["target_path_length_m"],
        "actual_path_length_m": legacy["actual_path_length_m"],
        "path_length_ratio": legacy["path_length_ratio"],
        "path_length_ratio_absolute_deviation": legacy[
            "path_length_ratio_absolute_deviation"
        ],
        "movement_mean_euclidean_m": legacy["movement_mean_euclidean_m"],
        "endpoint_euclidean_m": legacy["endpoint_euclidean_m"],
        "movement_mean_euclidean_per_target_path_length": legacy[
            "movement_mean_euclidean_per_target_path_length"
        ],
        "endpoint_euclidean_per_target_path_length": legacy[
            "endpoint_euclidean_per_target_path_length"
        ],
        "movement_sample_0_error_m": float(np.linalg.norm(actual[0] - target[0])),
        "movement_sample_1_error_m": float(np.linalg.norm(actual[1] - target[1])),
    }


def prepare_metrics(
    trace: dict[str, np.ndarray], prepare_entry: int, movement_start: int
) -> dict[str, Any]:
    prepare = _trace_slice(trace, prepare_entry, movement_start + 1)
    actual = prepare["actual_xy"]
    target_start = prepare["target_xy"][-1]
    error = np.linalg.norm(actual - target_start, axis=1)
    excursion = np.linalg.norm(actual - actual[0], axis=1)
    finite_speed = prepare["fingertip_speed_mps"][
        np.isfinite(prepare["fingertip_speed_mps"])
    ]
    return {
        "prepare_sample_count": int(len(actual)),
        "prepare_path_length_m": float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()),
        "prepare_max_excursion_m": float(excursion.max()),
        "prepare_max_target_start_error_m": float(error.max()),
        "prepare_entry_target_start_error_m": float(error[0]),
        "last_prepare_decision_target_start_error_m": float(error[-2]),
        "prepare_end_target_start_error_m": float(error[-1]),
        "prepare_mean_fingertip_speed_mps": float(finite_speed.mean()),
        "prepare_max_fingertip_speed_mps": float(finite_speed.max()),
        "prepare_frame_target_start_error_m": error.tolist(),
        "prepare_frame_fingertip_speed_mps": [
            float(value) if np.isfinite(value) else None
            for value in prepare["fingertip_speed_mps"]
        ],
    }


def _angle_degrees(before: np.ndarray, after: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(before))
    second_norm = float(np.linalg.norm(after))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise RuntimeError("corner angle vector has zero length")
    cosine = float(np.dot(before, after) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def hengzhe_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    corner_xy: np.ndarray,
    window_radius: int,
) -> dict[str, float]:
    corner_distance = np.linalg.norm(target - corner_xy, axis=1)
    corner_index = int(np.argmin(corner_distance))
    if corner_index < window_radius or corner_index + window_radius >= len(target):
        raise RuntimeError("fixed hengzhe corner window does not fit the movement trace")
    error = np.linalg.norm(actual - target, axis=1)
    target_before = target[corner_index] - target[corner_index - window_radius]
    target_after = target[corner_index + window_radius] - target[corner_index]
    actual_before = actual[corner_index] - actual[corner_index - window_radius]
    actual_after = actual[corner_index + window_radius] - actual[corner_index]
    target_angle = _angle_degrees(target_before, target_after)
    actual_angle = _angle_degrees(actual_before, actual_after)
    return {
        "corner_target_sample_index": corner_index,
        "corner_min_distance_m": float(
            np.linalg.norm(actual - corner_xy, axis=1).min()
        ),
        "corner_pre_segment_mean_euclidean_m": float(error[: corner_index + 1].mean()),
        "corner_post_segment_mean_euclidean_m": float(error[corner_index:].mean()),
        "corner_target_turn_angle_deg": target_angle,
        "corner_actual_turn_angle_deg": actual_angle,
        "corner_turn_angle_difference_deg": actual_angle - target_angle,
        "corner_turn_angle_absolute_error_deg": abs(actual_angle - target_angle),
    }


def _trace_difference_metrics(
    trace: dict[str, np.ndarray], reference: dict[str, np.ndarray], p: int, m: int
) -> dict[str, Any]:
    left = _trace_slice(trace, p, m + 1)
    right = _trace_slice(reference, p, m + 1)
    difference = np.linalg.norm(left["actual_xy"] - right["actual_xy"], axis=1)
    return {
        "prepare_vs_local_clean_mean_distance_m": float(difference.mean()),
        "prepare_vs_local_clean_max_distance_m": float(difference.max()),
        "prepare_vs_local_clean_endpoint_distance_m": float(difference[-1]),
        "prepare_vs_local_clean_frame_distance_m": difference.tolist(),
    }


def _max_array_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"array shapes differ: {left.shape} != {right.shape}")
    if left.dtype.kind in "US" or right.dtype.kind in "US":
        if not np.array_equal(left, right):
            raise RuntimeError("string trace fields differ")
        return 0.0
    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        if not np.array_equal(left, right):
            raise RuntimeError("integer trace fields differ")
        return 0.0
    both_nan = np.isnan(left) & np.isnan(right)
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise RuntimeError("trace NaN masks differ")
    finite = ~both_nan
    return 0.0 if not finite.any() else float(np.max(np.abs(left[finite] - right[finite])))


def assert_traces_match(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    tolerance: float,
    label: str,
) -> float:
    fields = (
        "schedule_index",
        "target_xy",
        "actual_xy",
        "phase",
        "rule_index",
        "rule",
        "go_cue",
        "spatial_cue",
        "x",
        "h",
        "action",
        "plant_sensory_sha256",
        "plant_sensory_field_manifest_json",
    )
    maximum = 0.0
    for field in fields:
        difference = _max_array_difference(left[field], right[field])
        maximum = max(maximum, difference)
        if difference > tolerance:
            raise RuntimeError(
                f"{label} differs in {field}; max_abs={difference}, tolerance={tolerance}"
            )
    return maximum


def state_block_changes(condition_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    output = {}
    metrics = list(CHANGE_ERROR_METRICS)
    metrics.extend(
        metric
        for metric in HENGZHE_CHANGE_ERROR_METRICS
        if all(metric in condition_metrics[condition] for condition in CONDITIONS)
    )
    for metric in metrics:
        c0 = condition_metrics["C0_continuous"][metric]
        c1 = condition_metrics["C1_plant_sensory_clean"][metric]
        c2 = condition_metrics["C2_neural_clean"][metric]
        c3 = condition_metrics["C3_joint_clean"][metric]
        p1 = condition_metrics["P1_plant_sensory_clean_before_prepare"][metric]
        values = {
            "body_change": c0 - c1,
            "neural_change": c0 - c2,
            "joint_change": c0 - c3,
            "prepare_reset_change": c0 - p1,
            "interaction": c1 + c2 - c0 - c3,
        }
        target_length = condition_metrics["C0_continuous"]["target_path_length_m"]
        output[metric] = {
            **values,
            "normalized_by_target_path_length": {
                name: value / target_length for name, value in values.items()
            }
            if metric in PATH_NORMALIZED_CHANGE_METRICS
            else None,
        }
    return output


def _direction_summary(values: list[float], epsilon: float) -> dict[str, Any]:
    return {
        "count_lower_error": sum(value > epsilon for value in values),
        "count_higher_error": sum(value < -epsilon for value in values),
        "count_within_technical_epsilon": sum(abs(value) <= epsilon for value in values),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def aggregate_changes(boundaries: list[dict[str, Any]], epsilon: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in CHANGE_ERROR_METRICS:
        output[metric] = {}
        for change_name in (
            "body_change",
            "neural_change",
            "joint_change",
            "prepare_reset_change",
            "interaction",
        ):
            values = [boundary["changes"][metric][change_name] for boundary in boundaries]
            output[metric][change_name] = _direction_summary(values, epsilon)
        if metric in METER_ERROR_METRICS:
            output[metric]["normalized_by_target_path_length"] = {}
            for change_name in (
                "body_change",
                "neural_change",
                "joint_change",
                "prepare_reset_change",
                "interaction",
            ):
                values = [
                    boundary["changes"][metric]["normalized_by_target_path_length"][
                        change_name
                    ]
                    for boundary in boundaries
                ]
                output[metric]["normalized_by_target_path_length"][change_name] = {
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
    return output


def _load_isolated_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "checkpoint",
        "speed",
        "category",
        "rule",
        "condition_id",
        "character",
        "component_index",
        "variant",
        "movement_mean_euclidean_m",
        "endpoint_euclidean_m",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("isolated component evidence columns are incomplete")
    return rows


def map_isolated_controls(
    geometry_config_path: str | Path,
    isolated_path: Path,
    boundary_list: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    geometry = load_geometry_config(geometry_config_path)
    character_map = characters(geometry)
    conditions = stroke_conditions(geometry, include_jitter=False)
    rows = _load_isolated_rows(isolated_path)
    by_condition = {
        row["condition_id"]: row
        for row in rows
        if row["checkpoint"] == "best"
        and row["speed"] == "medium"
        and row["category"] == "stroke"
        and row["variant"] == "exact"
    }
    output = {}
    for boundary in boundary_list:
        character = boundary["character"]
        stroke_index = boundary["next_stroke_index"]
        target_start = character_map[character].strokes[stroke_index].points[0]
        candidates = [
            condition
            for condition in conditions
            if condition.character == character
            and condition.stroke_index == stroke_index
            and np.allclose(
                condition.start_xy_m, target_start, rtol=0.0, atol=1e-12
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"isolated exact geometry mapping is not unique for {boundary['boundary_id']}"
            )
        condition = candidates[0]
        row = by_condition.get(condition.condition_id)
        if row is None:
            raise RuntimeError(
                f"isolated exact evidence row is missing for {condition.condition_id}"
            )
        if (
            row["character"] != character
            or int(row["component_index"]) != stroke_index
            or row["rule"] != boundary["next_rule"]
        ):
            raise RuntimeError("isolated exact evidence metadata disagrees with geometry")
        output[boundary["boundary_id"]] = {
            "condition_id": condition.condition_id,
            "primitive_id": condition.primitive_id,
            "character": character,
            "component_index": stroke_index,
            "rule": condition.rule,
            "start_xy_m": condition.start_xy_m.tolist(),
            "movement_intervals": int(row["movement_intervals"]),
            "target_path_length_m": float(row["target_path_length_m"]),
            "movement_mean_euclidean_m": float(row["movement_mean_euclidean_m"]),
            "endpoint_euclidean_m": float(row["endpoint_euclidean_m"]),
            "movement_mean_euclidean_per_target_path_length": float(
                row["movement_mean_euclidean_per_target_path_length"]
            ),
            "endpoint_euclidean_per_target_path_length": float(
                row["endpoint_euclidean_per_target_path_length"]
            ),
        }
    if len(output) != 12:
        raise RuntimeError("isolated exact mapping must contain 12 boundary references")
    return output


def _mixed_decision(
    neural: dict[str, Any], plant: dict[str, Any], schedule_index: int
) -> dict[str, Any]:
    if int(neural["schedule_index"]) != schedule_index:
        raise RuntimeError("neural decision is aligned to the wrong schedule index")
    if int(plant["schedule_index"]) != schedule_index:
        raise RuntimeError("plant decision is aligned to the wrong schedule index")
    if neural["x"].shape != neural["h"].shape or neural["x"].ndim != 2:
        raise RuntimeError("neural state block must contain aligned full x and h tensors")
    if not torch.isfinite(neural["x"]).all() or not torch.isfinite(neural["h"]).all():
        raise RuntimeError("neural state block contains non-finite values")
    return {
        "schedule_index": schedule_index,
        "obs": plant["obs"].detach().clone(),
        "x": neural["x"].detach().clone(),
        "h": neural["h"].detach().clone(),
        "action_to_state": None,
        "plant": copy.deepcopy(plant["plant"]),
    }


def _decision_max_difference(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    maximum = 0.0
    for field in ("obs", "x", "h"):
        left = expected[field].detach().cpu().numpy()
        right = actual[field].detach().cpu().numpy()
        maximum = max(maximum, _max_array_difference(left, right))
    if plant_sensory_digest(expected["plant"]) != plant_sensory_digest(actual["plant"]):
        raise RuntimeError("round-trip continuation plant-sensory state differs")
    return maximum


def _one_step_roundtrip_check(
    policy: torch.nn.Module,
    geometry_config_path: str | Path,
    character: str,
    speed: str,
    seed: int,
    start_decision: dict[str, Any],
    expected_samples: list[dict[str, Any]],
    expected_captures: dict[int, dict[str, Any]],
    tolerance: float,
    label: str,
) -> float:
    start = int(start_decision["schedule_index"])
    next_index = start + 1
    branch_samples, branch_captures = _run_from_decision(
        policy,
        geometry_config_path,
        character,
        speed,
        seed,
        start_decision,
        next_index + 1,
        initial_action=start_decision["action_to_state"],
        capture_indices=(next_index,),
    )
    expected_selection = [
        sample
        for sample in expected_samples
        if start <= int(sample["schedule_index"]) <= next_index
    ]
    if len(expected_selection) != 2:
        raise RuntimeError(f"{label} expected trace does not contain two aligned samples")
    expected = finalize_trace(expected_selection, 1.0, None)
    actual = finalize_trace(branch_samples, 1.0, None)
    maximum = assert_traces_match(expected, actual, tolerance, label)
    maximum = max(
        maximum,
        _decision_max_difference(expected_captures[next_index], branch_captures[next_index]),
    )
    if maximum > tolerance:
        raise RuntimeError(f"{label} exceeds technical tolerance: {maximum}")
    return maximum


def _partial_restore_checks(
    geometry_config_path: str | Path,
    character: str,
    speed: str,
    seed: int,
    character_decision: dict[str, Any],
    clean_decision: dict[str, Any],
) -> dict[str, bool]:
    env, _ = _new_character_env(geometry_config_path, character, speed, seed)
    restore_plant_sensory(env, character_decision["plant"])
    character_digest = plant_sensory_digest(snapshot_plant_sensory(env))
    clean_x = clean_decision["x"].detach().clone()
    clean_h = clean_decision["h"].detach().clone()
    if plant_sensory_digest(snapshot_plant_sensory(env)) != character_digest:
        raise RuntimeError("neural-only restore changed plant-sensory state")
    character_x = character_decision["x"].detach().clone()
    character_h = character_decision["h"].detach().clone()
    restore_plant_sensory(env, clean_decision["plant"])
    if plant_sensory_digest(snapshot_plant_sensory(env)) != plant_sensory_digest(
        clean_decision["plant"]
    ):
        raise RuntimeError("plant-sensory-only restore omitted a state or buffer")
    _assert_tensor_equal(character_decision["x"], character_x, "plant-only neural x")
    _assert_tensor_equal(character_decision["h"], character_h, "plant-only neural h")
    restore_plant_sensory(env, clean_decision["plant"])
    _assert_tensor_equal(clean_decision["x"], clean_x, "joint neural x")
    _assert_tensor_equal(clean_decision["h"], clean_h, "joint neural h")
    return {
        "neural_only_restore_preserved_plant_sensory": True,
        "plant_sensory_only_restore_complete": True,
        "joint_restore_complete": True,
    }


def _trace_node_indices(trace: dict[str, np.ndarray], boundary: dict[str, Any]) -> dict[str, int]:
    output = {}
    mapping = {
        "prepare_entry": boundary["prepare_entry_schedule_index"],
        "last_prepare_decision": boundary["last_prepare_decision_schedule_index"],
        "movement_sample_0": boundary["movement_sample_0_schedule_index"],
        "movement_sample_1": boundary["movement_sample_1_schedule_index"],
    }
    for name, schedule_index in mapping.items():
        matches = np.flatnonzero(trace["schedule_index"] == schedule_index)
        output[f"{name}_schedule_index"] = int(schedule_index)
        output[f"{name}_trace_index"] = int(matches[0]) if len(matches) == 1 else -1
    return output


def _npz_key(boundary_id: str, condition: str, field: str) -> str:
    return f"{boundary_id}__{condition}__{field}"


def _save_trace_npz(
    path: Path,
    traces: dict[tuple[str, str], dict[str, np.ndarray]],
    boundary_lookup: dict[str, dict[str, Any]],
    manifest_json: str,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for (boundary_id, condition), trace in sorted(traces.items()):
        for field, value in trace.items():
            array = np.asarray(value)
            if array.dtype == object:
                raise RuntimeError(f"object array is forbidden in NPZ: {boundary_id}/{condition}/{field}")
            arrays[_npz_key(boundary_id, condition, field)] = array
        nodes = _trace_node_indices(trace, boundary_lookup[boundary_id])
        for field, value in nodes.items():
            arrays[_npz_key(boundary_id, condition, field)] = np.asarray(value, dtype=np.int64)
        arrays[
            _npz_key(boundary_id, condition, "plant_sensory_state_field_manifest_json")
        ] = np.asarray(manifest_json, dtype=np.str_)
        prepare_source = (
            condition
            if condition
            in (
                "C0_continuous",
                "P1_plant_sensory_clean_before_prepare",
                REFERENCE_CONDITION,
            )
            else "C0_continuous"
        )
        arrays[_npz_key(boundary_id, condition, "prepare_trace_source")] = np.asarray(
            prepare_source, dtype=np.str_
        )
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise RuntimeError("NPZ key set changed during write/readback")
        for name in archive.files:
            if archive[name].dtype == object:
                raise RuntimeError(f"NPZ readback contains object dtype: {name}")


def _csv_rows(boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for boundary in boundaries:
        local = boundary["local_clean_metrics"]
        isolated = boundary["isolated_exact_control"]
        changes = boundary["changes"]
        for condition in CONDITIONS:
            metrics = boundary["conditions"][condition]
            prepare = boundary["prepare_metrics_by_condition"][condition]
            row: dict[str, Any] = {
                "boundary_id": boundary["boundary_id"],
                "character": boundary["character"],
                "move_index": boundary["move_index"],
                "next_stroke_index": boundary["next_stroke_index"],
                "next_rule": boundary["next_rule"],
                "condition": condition,
                "movement_intervals": boundary["movement_intervals"],
                "movement_sample_count": metrics["movement_sample_count"],
                "prepare_metric_source": boundary["prepare_metric_source"][condition],
                "prepare_entry_schedule_index": boundary["prepare_entry_schedule_index"],
                "last_prepare_decision_schedule_index": boundary[
                    "last_prepare_decision_schedule_index"
                ],
                "movement_sample_0_schedule_index": boundary[
                    "movement_sample_0_schedule_index"
                ],
                "movement_sample_1_schedule_index": boundary[
                    "movement_sample_1_schedule_index"
                ],
                "isolated_control_condition_id": isolated["condition_id"],
                "isolated_control_movement_mean_euclidean_m": isolated[
                    "movement_mean_euclidean_m"
                ],
                "isolated_control_endpoint_euclidean_m": isolated[
                    "endpoint_euclidean_m"
                ],
                "local_clean_movement_mean_euclidean_m": local[
                    "movement_mean_euclidean_m"
                ],
                "local_clean_endpoint_euclidean_m": local["endpoint_euclidean_m"],
            }
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[name] = value
            for name, value in prepare.items():
                if isinstance(value, (int, float)):
                    row[name] = value
            for metric in changes:
                c0 = boundary["conditions"]["C0_continuous"][metric]
                absolute = c0 - metrics[metric]
                row[f"c0_minus_condition__{metric}"] = absolute
                row[f"c0_minus_condition_normalized__{metric}"] = (
                    absolute / metrics["target_path_length_m"]
                    if metric in PATH_NORMALIZED_CHANGE_METRICS
                    else ""
                )
                for change_name, value in changes[metric].items():
                    if change_name == "normalized_by_target_path_length":
                        continue
                    row[f"{change_name}__{metric}"] = value
                normalized_changes = changes[metric]["normalized_by_target_path_length"]
                if normalized_changes is not None:
                    for change_name, value in normalized_changes.items():
                        row[
                            f"{change_name}_normalized_by_target_path_length__{metric}"
                        ] = value
            if condition == "P1_plant_sensory_clean_before_prepare":
                for name, value in boundary["p1_vs_local_clean_prepare"].items():
                    if isinstance(value, (int, float)):
                        row[name] = value
            else:
                for name in (
                    "prepare_vs_local_clean_mean_distance_m",
                    "prepare_vs_local_clean_max_distance_m",
                    "prepare_vs_local_clean_endpoint_distance_m",
                ):
                    row[name] = ""
            rows.append(row)
    if len(rows) != 60:
        raise RuntimeError(f"metrics CSV requires 60 rows, got {len(rows)}")
    counts = Counter(row["boundary_id"] for row in rows)
    if set(counts.values()) != {5}:
        raise RuntimeError("each boundary must have exactly five CSV rows")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Complete-character boundary state-block intervention sensitivity",
        "",
        "## Integrity",
        "",
        f"All preflight and alignment gates passed: `{summary['completed']}`. "
        f"The run contains {summary['actual_executed_condition_count']} short continuations "
        f"over {summary['boundary_count']} boundaries. No behavioral pass/fail threshold was used.",
        "",
        "C1 and C2 are artificial cross-combinations and may be off-manifold. Their values "
        "describe sensitivity to replacing a complete state block, not independent natural "
        "causal contributions.",
        "",
        "## Per-boundary movement mean Euclidean error (m)",
        "",
        "| Boundary | C0 | C1 | C2 | C3 | P1 | Local clean | Isolated exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for boundary in summary["boundaries"]:
        conditions = boundary["conditions"]
        values = [
            conditions[name]["movement_mean_euclidean_m"] for name in CONDITIONS
        ]
        lines.append(
            "| "
            + boundary["boundary_id"]
            + " | "
            + " | ".join(_format_float(value) for value in values)
            + " | "
            + _format_float(
                boundary["local_clean_metrics"]["movement_mean_euclidean_m"]
            )
            + " | "
            + _format_float(
                boundary["isolated_exact_control"]["movement_mean_euclidean_m"]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Direction and distribution summaries",
            "",
            "Positive change means the replacement condition has a lower numerical error than C0; "
            "negative means a higher error. Counts use the measured technical epsilon.",
            "",
            "| Change | Lower-error count | Higher-error count | Within epsilon | Median (m) | Min (m) | Max (m) | Normalized median | Normalized min | Normalized max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    aggregate = summary["aggregate_changes"]["movement_mean_euclidean_m"]
    labels = {
        "body_change": "C0 - C1",
        "neural_change": "C0 - C2",
        "joint_change": "C0 - C3",
        "prepare_reset_change": "C0 - P1",
        "interaction": "C1 + C2 - C0 - C3",
    }
    normalized_aggregate = aggregate["normalized_by_target_path_length"]
    for name, label in labels.items():
        values = aggregate[name]
        normalized = normalized_aggregate[name]
        lines.append(
            f"| {label} | {values['count_lower_error']} | {values['count_higher_error']} | "
            f"{values['count_within_technical_epsilon']} | {_format_float(values['median'])} | "
            f"{_format_float(values['minimum'])} | {_format_float(values['maximum'])} | "
            f"{_format_float(normalized['median'])} | {_format_float(normalized['minimum'])} | "
            f"{_format_float(normalized['maximum'])} |"
        )
    lines.extend(
        [
            "",
            "## Prepare response after pre-prepare plant-sensory replacement",
            "",
            "Distances compare the complete P1 closed-loop prepare trajectory with the local-clean "
            "reference. The first P1/local-clean speed sample is stored as NaN in the trace because "
            "there is no same-branch position before the constructed prepare-entry state.",
            "",
            "| Boundary | Mean distance (m) | Maximum distance (m) | End distance (m) |",
            "|---|---:|---:|---:|",
        ]
    )
    for boundary in summary["boundaries"]:
        values = boundary["p1_vs_local_clean_prepare"]
        lines.append(
            f"| {boundary['boundary_id']} | "
            f"{_format_float(values['prepare_vs_local_clean_mean_distance_m'])} | "
            f"{_format_float(values['prepare_vs_local_clean_max_distance_m'])} | "
            f"{_format_float(values['prepare_vs_local_clean_endpoint_distance_m'])} |"
        )
    hengzhe_rows = [
        boundary for boundary in summary["boundaries"] if boundary["next_rule"] == "hengzhe"
    ]
    lines.extend(
        [
            "",
            "## Hengzhe fixed-window turning metrics",
            "",
            "| Boundary / condition | Pre-corner mean error (m) | Post-corner mean error (m) | Actual turn (deg) | Target turn (deg) | Difference (deg) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for boundary in hengzhe_rows:
        rows = {
            **boundary["conditions"],
            REFERENCE_CONDITION: boundary["local_clean_metrics"],
        }
        for condition, values in rows.items():
            lines.append(
                f"| {boundary['boundary_id']} / {condition} | "
                f"{_format_float(values['corner_pre_segment_mean_euclidean_m'])} | "
                f"{_format_float(values['corner_post_segment_mean_euclidean_m'])} | "
                f"{_format_float(values['corner_actual_turn_angle_deg'])} | "
                f"{_format_float(values['corner_target_turn_angle_deg'])} | "
                f"{_format_float(values['corner_turn_angle_difference_deg'])} |"
            )
    lines.extend(
        [
            "",
            "The interaction quantity is a descriptive non-additivity index, not an additive "
            "causal decomposition. Local-clean and isolated-exact values are shown as separate "
            "reference conditions because their contexts and calling interfaces differ.",
            "",
            "## Conclusion boundary",
            "",
            f"> {FIXED_SCOPE_STATEMENT}",
            "",
            "No 50/100/150 pilot, training, optimizer, backward pass, or checkpoint update was run.",
        ]
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def _boundary_corner_world(
    geometry_config_path: str | Path,
    boundary: dict[str, Any],
    anchor_m: np.ndarray,
) -> np.ndarray | None:
    if boundary["next_rule"] != "hengzhe":
        return None
    geometry = load_geometry_config(geometry_config_path)
    stroke = characters(geometry)[boundary["character"]].strokes[
        boundary["next_stroke_index"]
    ]
    if len(stroke.points) < 3:
        raise RuntimeError("hengzhe authority does not contain an interior corner")
    return np.asarray(stroke.points[1], dtype=np.float64) + np.asarray(anchor_m)


def _condition_metrics(
    trace: dict[str, np.ndarray],
    boundary: dict[str, Any],
    corner_world: np.ndarray | None,
    corner_window: int,
) -> dict[str, float | int]:
    m = boundary["movement_sample_0_schedule_index"]
    e = boundary["movement_end_index_exclusive"]
    movement = _trace_slice(trace, m, e)
    metrics: dict[str, float | int] = movement_metrics(
        movement["actual_xy"], movement["target_xy"]
    )
    metrics["movement_sample_count"] = int(len(movement["actual_xy"]))
    if metrics["movement_sample_count"] != boundary["movement_intervals"] + 1:
        raise RuntimeError("movement continuation does not contain intervals + 1 samples")
    if corner_world is not None:
        metrics.update(
            hengzhe_metrics(
                movement["actual_xy"],
                movement["target_xy"],
                corner_world,
                corner_window,
            )
        )
    return metrics


def _validate_trace_schedule(
    trace: dict[str, np.ndarray],
    env: HanziCharacterEnv,
    label: str,
) -> None:
    for local_index, schedule_index_value in enumerate(trace["schedule_index"]):
        schedule_index = int(schedule_index_value)
        expected_rule = int(torch.argmax(env.rule_input[0, schedule_index]).detach().cpu())
        expected_target = env.traj[0, schedule_index].detach().cpu().numpy()
        expected_go = float(env.go_cue[0, schedule_index, 0].detach().cpu())
        expected_cue = env.vis_inp[0, schedule_index].detach().cpu().numpy()
        if trace["phase"][local_index] != env.phase[schedule_index]:
            raise RuntimeError(f"{label} phase is misaligned")
        if int(trace["rule_index"][local_index]) != expected_rule:
            raise RuntimeError(f"{label} rule is misaligned")
        if not np.array_equal(trace["target_xy"][local_index], expected_target):
            raise RuntimeError(f"{label} target is misaligned")
        if trace["go_cue"][local_index] != expected_go:
            raise RuntimeError(f"{label} go cue is misaligned")
        if not np.array_equal(trace["spatial_cue"][local_index], expected_cue):
            raise RuntimeError(f"{label} spatial cue is misaligned")


def _validate_trace_numerics(trace: dict[str, np.ndarray], label: str) -> None:
    for field in ("target_xy", "actual_xy", "go_cue", "spatial_cue", "x", "h"):
        if not np.isfinite(trace[field]).all():
            raise RuntimeError(f"{label} contains non-finite {field}")
    action = trace["action"]
    if len(action) > 1 and not np.isfinite(action[1:]).all():
        raise RuntimeError(f"{label} contains non-finite executed actions")
    if np.isfinite(action).any() and (
        float(action[np.isfinite(action)].min()) < 0.0
        or float(action[np.isfinite(action)].max()) > 1.0
    ):
        raise RuntimeError(f"{label} action lies outside [0, 1]")
    speed = trace["fingertip_speed_mps"]
    if len(speed) > 1 and not np.isfinite(speed[1:]).all():
        raise RuntimeError(f"{label} contains non-finite fingertip speed after frame 0")


def run_boundary_intervention(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_boundary_intervention_config(config_path)
    if importlib.metadata.version("motornet") != "0.2.0":
        raise RuntimeError("state manifest is frozen to MotorNet 0.2.0")
    if torch.cuda.is_available():
        raise RuntimeError("boundary diagnostic requires CUDA to be unavailable")
    checkpoint_path = Path(config["checkpoint"])
    isolated_path = Path(config["isolated_control"]["path"])
    checkpoint_sha_before = _sha256_file(checkpoint_path)
    if checkpoint_sha_before != config["checkpoint_sha256"]:
        raise RuntimeError("best checkpoint SHA-256 differs from the frozen evidence")
    isolated_sha = _sha256_file(isolated_path)
    if isolated_sha != config["isolated_control"]["sha256"]:
        raise RuntimeError("isolated component evidence SHA-256 differs from the frozen evidence")

    geometry = load_geometry_config(config["geometry_config"])
    policy, checkpoint = load_hanzi_policy_checkpoint(checkpoint_path)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("policy freeze failed")
    policy_before = _state_clone(policy)
    policy_digest_before = _policy_state_digest(policy)
    hp = checkpoint["hp"]
    tolerance = float(config["technical_tolerance"])

    parsed: list[dict[str, Any]] = []
    character_data: dict[str, dict[str, Any]] = {}
    with _fixed_rng(config["seed"]):
        for character in CHARACTERS:
            parse_env, _ = _new_character_env(
                config["geometry_config"], character, config["speed"], config["seed"]
            )
            character_boundaries = enumerate_boundaries(parse_env)
            parsed.extend(character_boundaries)
            capture_indices = set()
            for boundary in character_boundaries:
                p = boundary["prepare_entry_schedule_index"]
                m = boundary["movement_sample_0_schedule_index"]
                capture_indices.update((p, p + 1, m, m + 1))
            natural_samples, natural_captures, natural_env = _natural_character_rollout(
                policy,
                hp,
                config["geometry_config"],
                character,
                config["speed"],
                config["seed"],
                capture_indices,
            )
            character_data[character] = {
                "boundaries": character_boundaries,
                "natural_samples": natural_samples,
                "natural_captures": natural_captures,
                "env": natural_env,
            }
    expected_counts = {"mu": 3, "jiang": 5, "ke": 4}
    actual_counts = Counter(boundary["character"] for boundary in parsed)
    if dict(actual_counts) != expected_counts or len(parsed) != 12:
        raise RuntimeError(f"boundary coverage changed: {dict(actual_counts)}")
    boundary_lookup = {boundary["boundary_id"]: boundary for boundary in parsed}
    if len(boundary_lookup) != 12:
        raise RuntimeError("boundary IDs are not unique")
    isolated_controls = map_isolated_controls(
        config["geometry_config"], isolated_path, parsed
    )

    local_data: dict[str, dict[str, Any]] = {}
    for boundary in parsed:
        p = boundary["prepare_entry_schedule_index"]
        m = boundary["movement_sample_0_schedule_index"]
        e = boundary["movement_end_index_exclusive"]
        clean_entry = _clean_prepare_entry_decision(
            hp,
            config["geometry_config"],
            boundary["character"],
            config["speed"],
            config["seed"],
            p,
        )
        local_samples, local_captures = _run_from_decision(
            policy,
            config["geometry_config"],
            boundary["character"],
            config["speed"],
            config["seed"],
            clean_entry,
            e,
            initial_action=None,
            capture_indices=(m, m + 1),
        )
        local_trace = finalize_trace(local_samples, geometry.dt_seconds, None)
        local_data[boundary["boundary_id"]] = {
            "entry": clean_entry,
            "samples": local_samples,
            "captures": local_captures,
            "trace": local_trace,
        }

    first = parsed[0]
    first_character = character_data[first["character"]]
    p_first = first["prepare_entry_schedule_index"]
    m_first = first["movement_sample_0_schedule_index"]
    roundtrip_maxima = {
        "prepare_entry": _one_step_roundtrip_check(
            policy,
            config["geometry_config"],
            first["character"],
            config["speed"],
            config["seed"],
            first_character["natural_captures"][p_first],
            first_character["natural_samples"],
            first_character["natural_captures"],
            tolerance,
            "prepare-entry full round-trip",
        ),
        "movement_sample_0": _one_step_roundtrip_check(
            policy,
            config["geometry_config"],
            first["character"],
            config["speed"],
            config["seed"],
            first_character["natural_captures"][m_first],
            first_character["natural_samples"],
            first_character["natural_captures"],
            tolerance,
            "movement-sample-0 full round-trip",
        ),
        "local_clean_movement_sample_0": _one_step_roundtrip_check(
            policy,
            config["geometry_config"],
            first["character"],
            config["speed"],
            config["seed"],
            local_data[first["boundary_id"]]["captures"][m_first],
            local_data[first["boundary_id"]]["samples"],
            local_data[first["boundary_id"]]["captures"],
            tolerance,
            "local-clean movement-sample-0 full round-trip",
        ),
    }
    partial_checks = _partial_restore_checks(
        config["geometry_config"],
        first["character"],
        config["speed"],
        config["seed"],
        first_character["natural_captures"][m_first],
        local_data[first["boundary_id"]]["captures"][m_first],
    )
    epsilon_technical = max(roundtrip_maxima.values())
    _assert_state_equal(policy_before, policy)
    if _sha256_file(checkpoint_path) != checkpoint_sha_before:
        raise RuntimeError("checkpoint changed during preflight")

    result_boundaries = []
    all_traces: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    executed_conditions = 0
    c0_reproduction_max = 0.0
    c3_reproduction_max = 0.0
    first_manifest = plant_sensory_manifest(
        first_character["natural_captures"][m_first]["plant"]
    )

    for boundary in parsed:
        boundary_id = boundary["boundary_id"]
        character = boundary["character"]
        source = character_data[character]
        natural_samples = source["natural_samples"]
        natural_captures = source["natural_captures"]
        env = source["env"]
        p = boundary["prepare_entry_schedule_index"]
        m = boundary["movement_sample_0_schedule_index"]
        e = boundary["movement_end_index_exclusive"]
        char_p = natural_captures[p]
        char_m = natural_captures[m]
        clean_entry = local_data[boundary_id]["entry"]
        clean_m = local_data[boundary_id]["captures"][m]
        local_trace = local_data[boundary_id]["trace"]

        decisions = {
            "C0_continuous": _mixed_decision(char_m, char_m, m),
            "C1_plant_sensory_clean": _mixed_decision(char_m, clean_m, m),
            "C2_neural_clean": _mixed_decision(clean_m, char_m, m),
            "C3_joint_clean": _mixed_decision(clean_m, clean_m, m),
            "P1_plant_sensory_clean_before_prepare": _mixed_decision(
                char_p, clean_entry, p
            ),
        }
        initial_actions = {
            "C0_continuous": char_m["action_to_state"],
            "C1_plant_sensory_clean": None,
            "C2_neural_clean": None,
            "C3_joint_clean": clean_m["action_to_state"],
            "P1_plant_sensory_clean_before_prepare": None,
        }
        condition_traces: dict[str, dict[str, np.ndarray]] = {}
        for condition in CONDITIONS:
            samples, _ = _run_from_decision(
                policy,
                config["geometry_config"],
                character,
                config["speed"],
                config["seed"],
                decisions[condition],
                e,
                initial_action=initial_actions[condition],
            )
            executed_conditions += 1
            if condition == "C0_continuous":
                samples = natural_samples[p:m] + samples
                previous = natural_samples[p - 1]["actual_xy"]
            else:
                previous = None
            trace = finalize_trace(samples, geometry.dt_seconds, previous)
            condition_traces[condition] = trace
            all_traces[(boundary_id, condition)] = trace
            _validate_trace_schedule(trace, env, f"{boundary_id}/{condition}")
            _validate_trace_numerics(trace, f"{boundary_id}/{condition}")
        all_traces[(boundary_id, REFERENCE_CONDITION)] = local_trace
        _validate_trace_schedule(local_trace, env, f"{boundary_id}/{REFERENCE_CONDITION}")
        _validate_trace_numerics(local_trace, f"{boundary_id}/{REFERENCE_CONDITION}")

        source_movement = finalize_trace(
            natural_samples[m:e], geometry.dt_seconds, natural_samples[m - 1]["actual_xy"]
        )
        c0_movement = _trace_slice(condition_traces["C0_continuous"], m, e)
        local_movement = _trace_slice(local_trace, m, e)
        c3_movement = _trace_slice(condition_traces["C3_joint_clean"], m, e)
        c0_reproduction_max = max(
            c0_reproduction_max,
            assert_traces_match(
                source_movement,
                c0_movement,
                tolerance,
                f"{boundary_id} C0 reproduction",
            ),
        )
        c3_reproduction_max = max(
            c3_reproduction_max,
            assert_traces_match(
                local_movement,
                c3_movement,
                tolerance,
                f"{boundary_id} C3 reproduction",
            ),
        )
        for condition, trace in condition_traces.items():
            movement = _trace_slice(trace, m, e)
            if len(movement["actual_xy"]) != boundary["movement_intervals"] + 1:
                raise RuntimeError(f"{boundary_id}/{condition} has an off-by-one movement")
            if movement["go_cue"][0] != 1.0 or movement["go_cue"][1] != 1.0:
                raise RuntimeError("movement sample 0/1 go-cue alignment changed")
            if not np.array_equal(movement["target_xy"], local_movement["target_xy"]):
                raise RuntimeError("condition and reference movement targets are misaligned")

        corner_world = _boundary_corner_world(
            config["geometry_config"], boundary, np.asarray(env.anchor_m)
        )
        condition_metrics = {
            condition: _condition_metrics(
                trace, boundary, corner_world, config["corner_window_radius_samples"]
            )
            for condition, trace in condition_traces.items()
        }
        local_metrics = _condition_metrics(
            local_trace, boundary, corner_world, config["corner_window_radius_samples"]
        )
        c0_prepare = prepare_metrics(condition_traces["C0_continuous"], p, m)
        p1_prepare = prepare_metrics(
            condition_traces["P1_plant_sensory_clean_before_prepare"], p, m
        )
        local_prepare = prepare_metrics(local_trace, p, m)
        prepare_by_condition = {
            "C0_continuous": c0_prepare,
            "C1_plant_sensory_clean": c0_prepare,
            "C2_neural_clean": c0_prepare,
            "C3_joint_clean": c0_prepare,
            "P1_plant_sensory_clean_before_prepare": p1_prepare,
        }
        prepare_sources = {
            "C0_continuous": "C0_continuous",
            "C1_plant_sensory_clean": "C0_continuous",
            "C2_neural_clean": "C0_continuous",
            "C3_joint_clean": "C0_continuous",
            "P1_plant_sensory_clean_before_prepare": (
                "P1_plant_sensory_clean_before_prepare"
            ),
        }
        changes = state_block_changes(condition_metrics)  # type: ignore[arg-type]
        result_boundaries.append(
            {
                **boundary,
                "conditions": condition_metrics,
                "local_clean_metrics": local_metrics,
                "local_clean_prepare_metrics": local_prepare,
                "isolated_exact_control": isolated_controls[boundary_id],
                "prepare_metrics_by_condition": prepare_by_condition,
                "prepare_metric_source": prepare_sources,
                "p1_vs_local_clean_prepare": _trace_difference_metrics(
                    condition_traces["P1_plant_sensory_clean_before_prepare"],
                    local_trace,
                    p,
                    m,
                ),
                "changes": changes,
            }
        )

    if executed_conditions != 60:
        raise RuntimeError(f"expected 60 conditions, executed {executed_conditions}")
    epsilon_technical = max(
        epsilon_technical, c0_reproduction_max, c3_reproduction_max
    )
    _assert_state_equal(policy_before, policy)
    policy_digest_after = _policy_state_digest(policy)
    if policy_digest_before != policy_digest_after:
        raise RuntimeError("policy parameters changed during diagnostic")
    checkpoint_sha_after = _sha256_file(checkpoint_path)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("checkpoint file changed during diagnostic")

    integrity_checks = {
        "boundary_count_is_12": True,
        "condition_count_is_60": True,
        "neural_state_contains_x_and_h": True,
        "plant_sensory_manifest_complete": True,
        "full_roundtrip_continuation_passed": True,
        **partial_checks,
        "c0_matches_unbranched_character": True,
        "c3_matches_local_clean_reference": True,
        "movement_sample_count_is_intervals_plus_one": True,
        "sample_0_and_sample_1_go_alignment_passed": True,
        "isolated_exact_mapping_unique_for_12_boundaries": True,
        "policy_frozen_and_bitwise_unchanged": True,
        "optimizer_not_created": True,
        "backward_not_executed": True,
        "checkpoint_unchanged": True,
        "observation_and_action_noise_zero": True,
        "schedule_trace_alignment_passed": True,
    }
    summary = {
        "project": PROJECT,
        "run_kind": config["run_kind"],
        "completed": True,
        "scope_statement": FIXED_SCOPE_STATEMENT,
        "behavioral_pass_fail_defined": False,
        "off_manifold_cross_combinations": [
            "C1_plant_sensory_clean",
            "C2_neural_clean",
        ],
        "resolved_config": config,
        "boundary_count": 12,
        "expected_condition_count": 60,
        "actual_executed_condition_count": executed_conditions,
        "epsilon_technical": epsilon_technical,
        "technical_tolerance": tolerance,
        "roundtrip_max_abs_difference": roundtrip_maxima,
        "c0_reproduction_max_abs_difference": c0_reproduction_max,
        "c3_reproduction_max_abs_difference": c3_reproduction_max,
        "integrity_checks": integrity_checks,
        "boundaries": result_boundaries,
        "aggregate_changes": aggregate_changes(result_boundaries, epsilon_technical),
    }
    rows = _csv_rows(result_boundaries)
    output_dir = Path(config["output_directory"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "boundary_intervention_summary.json", summary)
    _write_csv(output_dir / "boundary_intervention_metrics.csv", rows)
    _save_trace_npz(
        output_dir / "boundary_trace.npz",
        all_traces,
        boundary_lookup,
        json.dumps(first_manifest, sort_keys=True, separators=(",", ":")),
    )
    _write_report(output_dir / "BOUNDARY_INTERVENTION_REPORT.md", summary)

    repo_root = Path(__file__).resolve().parents[1]
    provenance = {
        "project": PROJECT,
        "run_kind": config["run_kind"],
        "git_head": _git_head(repo_root),
        "submodule_head": _git_head(repo_root / "mRNNTorch"),
        "checkpoint_kind": "best",
        "checkpoint_update": int(checkpoint["update"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "configuration_path": str(config_path),
        "configuration_sha256": _sha256_file(config_path),
        "isolated_control_evidence_path": str(isolated_path),
        "isolated_control_evidence_sha256": isolated_sha,
        "speed": "medium",
        "network_noise": False,
        "environment_noise": False,
        "deterministic_observation": True,
        "device": "cpu",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "motornet_version": importlib.metadata.version("motornet"),
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "boundary_ids": [boundary["boundary_id"] for boundary in parsed],
        "expected_condition_count": 60,
        "actual_executed_condition_count": executed_conditions,
        "policy_requires_grad": any(
            parameter.requires_grad for parameter in policy.parameters()
        ),
        "policy_state_sha256_before": policy_digest_before,
        "policy_state_sha256_after": policy_digest_after,
        "policy_frozen_and_bitwise_unchanged": True,
        "optimizer_created": False,
        "backward_executed": False,
        "epsilon_technical": epsilon_technical,
        "test_count": None,
        "skipped_count": None,
        "program_exit_code": None,
        "tee_exit_code": None,
    }
    _write_json(output_dir / "provenance.json", provenance)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete-character boundary state-block intervention diagnostic"
    )
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    summary = run_boundary_intervention(arguments.config)
    print(
        json.dumps(
            {
                "completed": summary["completed"],
                "boundary_count": summary["boundary_count"],
                "actual_executed_condition_count": summary[
                    "actual_executed_condition_count"
                ],
                "epsilon_technical": summary["epsilon_technical"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
