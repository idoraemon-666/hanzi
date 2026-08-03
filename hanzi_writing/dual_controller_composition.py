"""Compose three characters from two frozen dual fixed-rule RNNs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.dual_rule_envs import DualFixedRuleEnv
from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    FIXED_SPEED_NAME,
    FixedRuleCondition,
    condition_manifest,
    conditions,
)
from hanzi_writing.dual_rule_training import (
    JOINT_GRADIENT_VARIANT,
    load_config as load_dual_config,
    validate_checkpoint_identity,
)
from hanzi_writing.training import _make_effector, _sha256_file, _state_clone
from train import _build_policy


PROJECT = "hanzi_stroke_temporal_composition"
VARIANT = "frozen_joint_gradient_onset_window_character_composition_full_trial_v2"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "hanzi_stroke_temporal_composition_frozen_dual_controller_composition_full_trial_v2.json"
)
CHARACTERS = ("mu", "jiang", "ke")
EXPECTED_RULE_SEQUENCES = {
    "mu": (
        ("stroke", "medium_heng_mu_0"),
        ("move", "mu_move_0"),
        ("stroke", "long_shu_mu_1"),
        ("move", "mu_move_1"),
        ("stroke", "pie_mu_2"),
        ("move", "mu_move_2"),
        ("stroke", "na_mu_3"),
    ),
    "jiang": (
        ("stroke", "dian_jiang_0"),
        ("move", "jiang_move_0"),
        ("stroke", "dian_jiang_1"),
        ("move", "jiang_move_1"),
        ("stroke", "ti_jiang_2"),
        ("move", "jiang_move_2"),
        ("stroke", "medium_heng_jiang_3"),
        ("move", "jiang_move_3"),
        ("stroke", "medium_shu_jiang_4"),
        ("move", "jiang_move_4"),
        ("stroke", "long_heng_jiang_5"),
    ),
    "ke": (
        ("stroke", "long_heng_ke_0"),
        ("move", "ke_move_0"),
        ("stroke", "short_shu_ke_1"),
        ("move", "ke_move_1"),
        ("stroke", "hengzhe_ke_2"),
        ("move", "ke_move_2"),
        ("stroke", "short_heng_ke_3"),
        ("move", "ke_move_3"),
        ("stroke", "shugou_ke_4"),
    ),
}
METRIC_FIELDS = (
    "character",
    "segment_index",
    "model_kind",
    "rule",
    "condition_id",
    "movement_intervals",
    "trial_timesteps",
    "reset_start_x_m",
    "reset_start_y_m",
    "canonical_start_x_m",
    "canonical_start_y_m",
    "start_offset_euclidean_m",
    "movement_mean_euclidean_m",
    "movement_endpoint_euclidean_m",
    "trial_endpoint_euclidean_m",
    "actual_trial_endpoint_x_m",
    "actual_trial_endpoint_y_m",
    "canonical_endpoint_x_m",
    "canonical_endpoint_y_m",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ")


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
        "frozen composition configuration",
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
        "frozen_dual_controller_character_composition",
        VARIANT,
        True,
        42,
        "cpu",
    ):
        raise ValueError("frozen composition identity differs")
    if config["composition"] != {
        "characters": list(CHARACTERS),
        "controller_order": "stroke_then_alternating_move_stroke",
        "complete_trial_per_task": True,
        "actual_endpoint_as_next_start": True,
        "effector_boundary_reset": "actual_endpoint_zero_velocity_standard_state",
        "recurrent_state_reset_each_trial": True,
        "feedback_buffers": "reinitialize_from_current_sensory_state",
        "translate_canonical_target": False,
        "network_noise": False,
        "deterministic_observation": True,
    }:
        raise ValueError("frozen composition boundary contract differs")
    if config["render"] != {
        "trajectory": "actual_complete_trial_including_reset_state",
        "stroke_linestyle": "solid",
        "move_linestyle": "dashed",
        "equal_aspect": True,
        "postprocessing": False,
    }:
        raise ValueError("frozen composition rendering contract differs")
    if config["output"] != {
        "directory": (
            "runs/hanzi_stroke_temporal_composition/"
            "frozen_dual_controller_character_composition_full_trial/dev42"
        )
    }:
        raise ValueError("frozen composition output differs")
    dual_config, geometry = load_dual_config(config["joint_gradient_config"])
    if dual_config["variant"] != JOINT_GRADIENT_VARIANT:
        raise ValueError("frozen composition requires the joint-gradient config")
    return config, dual_config, geometry


class ActualEndpointDualFixedRuleEnv(DualFixedRuleEnv):
    """Reset MotorNet at an actual prior endpoint without moving the target."""

    def reset(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        supplied = {} if options is None else dict(options)
        if "actual_start_xy_m" not in supplied:
            raise ValueError("composition reset requires actual_start_xy_m")
        actual_start = np.asarray(
            supplied.pop("actual_start_xy_m"), dtype=np.float64
        )
        if actual_start.shape != (2,) or not np.isfinite(actual_start).all():
            raise ValueError("composition actual start must be one finite xy point")
        deterministic = bool(supplied.get("deterministic", False))
        super().reset(
            testing=testing,
            seed=seed,
            options=supplied,
        )
        if len(self.conditions) != 1:
            raise ValueError("composition requires batch size one")
        self._reset_effector(actual_start[None, :])
        self.composition_actual_start_xy_m = actual_start.copy()
        return self._initialize_observation_buffers(1, deterministic=deterministic)


def character_sequences(geometry: Any) -> dict[str, tuple[FixedRuleCondition, ...]]:
    libraries = {
        model_kind: {condition.rule: condition for condition in conditions(model_kind, geometry)}
        for model_kind in ("stroke", "move")
    }
    output = {}
    for character in CHARACTERS:
        sequence = tuple(
            libraries[model_kind][rule]
            for model_kind, rule in EXPECTED_RULE_SEQUENCES[character]
        )
        actual = tuple((condition.model_kind, condition.rule) for condition in sequence)
        if actual != EXPECTED_RULE_SEQUENCES[character]:
            raise RuntimeError(f"frozen character sequence differs: {character}")
        output[character] = sequence
    if sum(len(value) for value in output.values()) != 27:
        raise RuntimeError("frozen character composition must contain 27 trials")
    return output


def _validate_source(
    config: dict[str, Any],
    dual_config: dict[str, Any],
    geometry: Any,
    source_root: Path,
) -> dict[str, dict[str, Any]]:
    source = config["source"]
    if _sha256_file(source_root / "resolved_config.json") != source["resolved_config_sha256"]:
        raise RuntimeError("joint-gradient resolved config hash differs")
    if _load_json(source_root / "resolved_config.json") != dual_config:
        raise RuntimeError("joint-gradient resolved config content differs")
    if _sha256_file(source_root / "provenance.json") != source["provenance_sha256"]:
        raise RuntimeError("joint-gradient provenance hash differs")
    provenance = _load_json(source_root / "provenance.json")
    if (
        provenance.get("completed") is not True
        or provenance.get("variant") != source["variant"]
        or provenance.get("git_head") != source["git_head"]
        or provenance.get("submodule_head") != source["submodule_head"]
        or provenance.get("program_exit_code") != 0
        or provenance.get("tee_exit_code") != 0
        or provenance.get("skipped_test_count") != 0
        or provenance.get("integrity", {}).get("workers_completed") != 6
    ):
        raise RuntimeError("joint-gradient provenance contract differs")
    manifests = {}
    for model_kind in ("stroke", "move"):
        manifest = _load_json(source_root / f"{model_kind}_condition_manifest.json")
        expected = condition_manifest(model_kind, geometry)
        if manifest != expected:
            raise RuntimeError(f"joint-gradient {model_kind} manifest differs")
        if (
            manifest["condition_manifest_sha256"]
            != source["condition_manifest_sha256"][model_kind]
        ):
            raise RuntimeError(f"joint-gradient {model_kind} manifest hash differs")
        manifests[model_kind] = manifest
    return manifests


def _load_frozen_policy(
    config: dict[str, Any],
    dual_config: dict[str, Any],
    source_root: Path,
    model_kind: str,
    manifest: dict[str, Any],
) -> tuple[Any, dict[str, Any], str]:
    selection = config["source"]["checkpoints"][model_kind]
    path = source_root / selection["relative_path"]
    digest = _sha256_file(path)
    if digest != selection["sha256"]:
        raise RuntimeError(f"selected {model_kind} checkpoint hash differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(
        checkpoint,
        model_kind,
        selection["loss_arm"],
        manifest,
        JOINT_GRADIENT_VARIANT,
    )
    if checkpoint.get("update") != selection["update"]:
        raise RuntimeError(f"selected {model_kind} checkpoint update differs")
    hp = checkpoint.get("hp")
    if not isinstance(hp, dict) or hp.get("protocol_config") != dual_config:
        raise RuntimeError(f"selected {model_kind} checkpoint config differs")
    policy = _build_policy(hp, dual_config["model_common"]["output_size"], torch.device("cpu"))
    policy.load_state_dict(checkpoint["agent_state_dict"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, hp, digest


def _make_env(dual_config: dict[str, Any], model_kind: str) -> ActualEndpointDualFixedRuleEnv:
    return ActualEndpointDualFixedRuleEnv(
        effector=_make_effector(),
        model_kind=model_kind,
        geometry_config_path=dual_config["geometry_config"],
        action_frame_stacking=0,
    )


def _rollout_trial(
    policy: Any,
    env: ActualEndpointDualFixedRuleEnv,
    hp: dict[str, Any],
    condition: FixedRuleCondition,
    actual_start_xy_m: np.ndarray,
) -> dict[str, Any]:
    x = torch.zeros((1, hp["hid_size"]), dtype=torch.float32)
    h = torch.zeros_like(x)
    obs, info = env.reset(
        options={
            "conditions": (condition,),
            "speed_name": FIXED_SPEED_NAME,
            "delay_steps": FIXED_DELAY_STEPS,
            "deterministic": True,
            "actual_start_xy_m": actual_start_xy_m,
        }
    )
    reset_xy = (
        info["states"]["fingertip"].detach().cpu().numpy()[0]
        - env.anchor_m
    )
    if not np.allclose(reset_xy, actual_start_xy_m, rtol=0.0, atol=1e-6):
        raise RuntimeError("MotorNet composition reset position differs")
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
        raise RuntimeError("complete-trial trajectory coverage differs")
    actual_full_with_reset = np.concatenate((reset_xy[None, :], actual), axis=0)
    movement_start, movement_end = movement_bounds
    expected_target = condition.points_m
    movement_target = target[movement_start:movement_end]
    if movement_target.shape != expected_target.shape or not np.allclose(
        movement_target, expected_target, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("canonical composition target was translated or changed")
    return {
        "actual": actual,
        "actual_full_with_reset": actual_full_with_reset,
        "target": target,
        "movement_actual": actual[movement_start:movement_end],
        "movement_target": movement_target,
        "reset_start": reset_xy,
        "actual_endpoint": actual[-1].copy(),
        "epoch_bounds": dict(env.epoch_bounds),
        "timesteps": timestep,
    }


def _segment_row(
    character: str,
    segment_index: int,
    condition: FixedRuleCondition,
    result: dict[str, Any],
) -> dict[str, Any]:
    movement_error = np.linalg.norm(
        result["movement_actual"] - result["movement_target"], axis=1
    )
    canonical_start = condition.points_m[0]
    canonical_endpoint = condition.points_m[-1]
    return {
        "character": character,
        "segment_index": segment_index,
        "model_kind": condition.model_kind,
        "rule": condition.rule,
        "condition_id": condition.condition_id,
        "movement_intervals": condition.movement_intervals,
        "trial_timesteps": result["timesteps"],
        "reset_start_x_m": float(result["reset_start"][0]),
        "reset_start_y_m": float(result["reset_start"][1]),
        "canonical_start_x_m": float(canonical_start[0]),
        "canonical_start_y_m": float(canonical_start[1]),
        "start_offset_euclidean_m": float(
            np.linalg.norm(result["reset_start"] - canonical_start)
        ),
        "movement_mean_euclidean_m": float(movement_error.mean()),
        "movement_endpoint_euclidean_m": float(movement_error[-1]),
        "trial_endpoint_euclidean_m": float(
            np.linalg.norm(result["actual_endpoint"] - canonical_endpoint)
        ),
        "actual_trial_endpoint_x_m": float(result["actual_endpoint"][0]),
        "actual_trial_endpoint_y_m": float(result["actual_endpoint"][1]),
        "canonical_endpoint_x_m": float(canonical_endpoint[0]),
        "canonical_endpoint_y_m": float(canonical_endpoint[1]),
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
        points = segment["result"]["actual_full_with_reset"]
        axis.plot(points[:, 0], points[:, 1], style, color=color, linewidth=2.4, label=label)
    axis.set_title(f"{character}: frozen Stroke/Move complete-trial composition")
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
            points = segment["result"]["actual_full_with_reset"]
            axis.plot(points[:, 0], points[:, 1], style, color=color, linewidth=2.2, label=label)
        axis.set_title(character)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Frozen joint-gradient Stroke/Move complete-trial composition")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def run_composition(config_path: str | Path, source_results: str | Path) -> dict[str, Any]:
    config, dual_config, geometry = load_config(config_path)
    source_root = Path(source_results).resolve()
    manifests = _validate_source(config, dual_config, geometry, source_root)
    policies = {}
    hps = {}
    checkpoint_hashes = {}
    initial_states = {}
    environments = {}
    for model_kind in ("stroke", "move"):
        policy, hp, digest = _load_frozen_policy(
            config, dual_config, source_root, model_kind, manifests[model_kind]
        )
        policies[model_kind] = policy
        hps[model_kind] = hp
        checkpoint_hashes[model_kind] = digest
        initial_states[model_kind] = _state_clone(policy)
        environments[model_kind] = _make_env(dual_config, model_kind)
    if not np.array_equal(environments["stroke"].anchor_m, environments["move"].anchor_m):
        raise RuntimeError("Stroke and Move MotorNet anchors differ")

    output = Path(config["output"]["directory"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    plot_directory = output / "plots"
    plot_directory.mkdir()
    sequence_library = character_sequences(geometry)
    all_results: dict[str, list[dict[str, Any]]] = {}
    rows = []
    for character in CHARACTERS:
        character_results = []
        arrays: dict[str, np.ndarray] = {}
        next_start = sequence_library[character][0].points_m[0].copy()
        previous_endpoint = None
        for segment_index, condition in enumerate(sequence_library[character]):
            if previous_endpoint is not None and not np.array_equal(next_start, previous_endpoint):
                raise RuntimeError("actual endpoint was not passed to the next trial")
            result = _rollout_trial(
                policies[condition.model_kind],
                environments[condition.model_kind],
                hps[condition.model_kind],
                condition,
                next_start,
            )
            prefix = f"segment_{segment_index:02d}_{condition.model_kind}_{condition.rule}"
            arrays[f"{prefix}__actual_full"] = result["actual"]
            arrays[f"{prefix}__actual_full_with_reset"] = result[
                "actual_full_with_reset"
            ]
            arrays[f"{prefix}__target_full"] = result["target"]
            arrays[f"{prefix}__actual_movement"] = result["movement_actual"]
            arrays[f"{prefix}__target_movement"] = result["movement_target"]
            rows.append(_segment_row(character, segment_index, condition, result))
            character_results.append({"condition": condition, "result": result})
            previous_endpoint = result["actual_endpoint"].copy()
            next_start = previous_endpoint.copy()
        np.savez_compressed(output / f"{character}_composition_trajectories.npz", **arrays)
        _plot_character(
            plot_directory / f"{character}_actual_full_trial_composition.png",
            character,
            character_results,
        )
        all_results[character] = character_results
    _plot_overview(
        plot_directory / "three_characters_actual_full_trial_composition.png",
        all_results,
    )

    with (output / "character_segment_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    policy_unchanged = {
        model_kind: all(
            torch.equal(value, policies[model_kind].state_dict()[name])
            for name, value in initial_states[model_kind].items()
        )
        for model_kind in ("stroke", "move")
    }
    if not all(policy_unchanged.values()):
        raise RuntimeError("frozen policy state changed during composition")
    numeric_values = [
        float(row[field])
        for row in rows
        for field in METRIC_FIELDS[7:]
    ]
    if not np.isfinite(numeric_values).all():
        raise RuntimeError("composition metrics contain non-finite values")
    summary = {
        "characters": list(CHARACTERS),
        "character_trial_counts": {
            character: len(all_results[character]) for character in CHARACTERS
        },
        "total_trials": len(rows),
        "stroke_trials": sum(row["model_kind"] == "stroke" for row in rows),
        "move_trials": sum(row["model_kind"] == "move" for row in rows),
        "full_trial_render_verified": all(
            segment["result"]["actual_full_with_reset"].shape[0]
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
        "# Frozen dual-controller character composition",
        "",
        "The selected joint-gradient Stroke/onset_window and Move/onset_window",
        "macro-best checkpoints were frozen and alternated over complete trials.",
        "Every next trial reset MotorNet at the preceding actual endpoint.",
        "",
        "Complete Stroke trial trajectories are solid; complete Move trial",
        "trajectories are dashed. Reset, stable, delay, movement, and hold",
        "states are all included.",
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
            "plots": 4,
            "trajectory_archives": 3,
            "metric_rows": len(rows),
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
