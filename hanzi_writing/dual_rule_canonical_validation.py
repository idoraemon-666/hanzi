"""Export complete independent canonical trials for the two frozen rule RNNs."""

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

from hanzi_writing.canonical_overfit import _movement_metrics
from hanzi_writing.dual_controller_composition import (
    _load_frozen_policy,
    _validate_source,
)
from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    FIXED_SPEED_NAME,
    conditions,
    rule_names,
)
from hanzi_writing.dual_rule_training import (
    JOINT_GRADIENT_VARIANT,
    _make_env,
    load_config as load_dual_config,
)
from hanzi_writing.geometry import PROJECT
from hanzi_writing.training import _rollout, _sha256_file, _state_clone


VARIANT = "frozen_joint_gradient_onset_window_canonical_validation_full_trial_v1"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "hanzi_stroke_temporal_composition_dual_rule_canonical_validation_v1.json"
)
MODEL_KINDS = ("stroke", "move")
METRIC_FIELDS = (
    "model_kind",
    "rule",
    "condition_id",
    "checkpoint_update",
    "stable_start",
    "stable_end",
    "delay_start",
    "delay_end",
    "movement_start",
    "movement_end",
    "hold_start",
    "hold_end",
    "trial_timesteps",
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
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    config = _load_json(path)
    if set(config) != {
        "project",
        "run_kind",
        "variant",
        "enabled",
        "seed",
        "device",
        "joint_gradient_config",
        "source",
        "validation",
        "render",
        "output",
    }:
        raise ValueError("canonical validation configuration keys differ")
    if (
        config["project"],
        config["run_kind"],
        config["variant"],
        config["enabled"],
        config["seed"],
        config["device"],
    ) != (
        PROJECT,
        "frozen_dual_rule_canonical_validation",
        VARIANT,
        True,
        42,
        "cpu",
    ):
        raise ValueError("canonical validation identity differs")
    if config["validation"] != {
        "candidate": "best_macro",
        "loss_arm": "onset_window",
        "independent_rollout_per_rule": True,
        "canonical_start_state": True,
        "cross_task_state_carryover": False,
        "effector_state_reset_each_rule": "canonical_standard_state",
        "recurrent_state_reset_each_rule": True,
        "feedback_buffers_reset_each_rule": True,
        "network_noise": False,
        "deterministic_observation": True,
        "batch_size": 1,
        "speed_name": FIXED_SPEED_NAME,
        "delay_steps": FIXED_DELAY_STEPS,
    }:
        raise ValueError("canonical validation rollout contract differs")
    if config["render"] != {
        "movement": "same_rollout_movement_epoch_including_first_and_last_samples",
        "full_trial": "same_rollout_stable_delay_movement_hold",
        "target_linestyle": "dashed_gray",
        "actual_linestyle": "solid_blue",
        "titles_only": True,
        "equal_aspect": True,
        "postprocessing": False,
    }:
        raise ValueError("canonical validation render contract differs")
    if config["output"] != {
        "directory": (
            "runs/hanzi_stroke_temporal_composition/"
            "frozen_dual_rule_canonical_validation/dev42"
        )
    }:
        raise ValueError("canonical validation output differs")
    source = config["source"]
    if set(source) != {
        "variant",
        "git_head",
        "submodule_head",
        "resolved_config_sha256",
        "provenance_sha256",
        "condition_manifest_sha256",
        "candidate_trajectories_sha256",
        "checkpoints",
    }:
        raise ValueError("canonical validation source keys differ")
    if source["variant"] != JOINT_GRADIENT_VARIANT:
        raise ValueError("canonical validation source variant differs")
    if set(source["candidate_trajectories_sha256"]) != set(MODEL_KINDS):
        raise ValueError("canonical validation candidate hashes differ")
    if set(source["checkpoints"]) != set(MODEL_KINDS):
        raise ValueError("canonical validation checkpoint selections differ")
    for model_kind in MODEL_KINDS:
        selection = source["checkpoints"][model_kind]
        if (
            selection["loss_arm"] != "onset_window"
            or selection["relative_path"]
            != f"{model_kind}/onset_window/best_macro_checkpoint.pt"
        ):
            raise ValueError(f"canonical validation {model_kind} selection differs")
    dual_config, geometry = load_dual_config(config["joint_gradient_config"])
    if dual_config["variant"] != JOINT_GRADIENT_VARIANT:
        raise ValueError("canonical validation requires the joint-gradient config")
    return config, dual_config, geometry


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], text=True, encoding="utf-8"
    ).strip()


def _plot(
    path: Path,
    model_kind: str,
    trajectories: list[dict[str, Any]],
    view: str,
) -> None:
    columns = 3
    rows = int(np.ceil(len(trajectories) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, rows * 3.88))
    flat = np.asarray(axes).reshape(-1)
    for axis, trajectory in zip(flat, trajectories):
        target = trajectory[f"target_{view}"]
        actual = trajectory[f"actual_{view}"]
        axis.plot(target[:, 0], target[:, 1], "--", color="#9e9e9e", linewidth=1.6)
        axis.plot(actual[:, 0], actual[:, 1], color="#1f5f91", linewidth=1.8)
        axis.set_title(trajectory["condition"].rule, fontsize=10, pad=7)
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.08)
        axis.axis("off")
    for axis in flat[len(trajectories) :]:
        axis.axis("off")
    label = "Stroke" if model_kind == "stroke" else "Move"
    view_label = "Movement" if view == "movement" else "Full Trial"
    figure.suptitle(
        f"Joint-gradient: {label} / onset_window - {view_label}",
        fontsize=17,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.982), h_pad=2.0, w_pad=2.0)
    figure.savefig(path, dpi=255, facecolor="white")
    plt.close(figure)


def _candidate_arrays(
    config: dict[str, Any], source_root: Path, model_kind: str
) -> Any:
    path = source_root / model_kind / "onset_window" / "candidate_trajectories.npz"
    expected_hash = config["source"]["candidate_trajectories_sha256"][model_kind]
    if _sha256_file(path) != expected_hash:
        raise RuntimeError(f"source {model_kind} candidate trajectory hash differs")
    return np.load(path, allow_pickle=False)


def _canonical_rollout(
    policy: Any,
    env: Any,
    hp: dict[str, Any],
    condition: Any,
    expected: Any,
) -> dict[str, Any]:
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
    actual_full = result["xy"][0].detach().cpu().numpy() - env.anchor_m
    target_full = result["target"][0].detach().cpu().numpy() - env.anchor_m
    movement_start, movement_end = result["epoch_bounds"]["movement"]
    actual_movement = actual_full[movement_start:movement_end]
    target_movement = target_full[movement_start:movement_end]
    expected_actual = expected[f"best_macro__{condition.rule}__actual"]
    expected_target = expected[f"best_macro__{condition.rule}__target"]
    if not np.array_equal(actual_movement, expected_actual):
        raise RuntimeError(f"canonical movement no longer matches source: {condition.rule}")
    if not np.array_equal(target_movement, expected_target):
        raise RuntimeError(f"canonical target no longer matches source: {condition.rule}")
    bounds = result["epoch_bounds"]
    if (
        bounds["stable"] != (0, 25)
        or bounds["delay"] != (25, 75)
        or bounds["movement"] != (75, 75 + len(condition.points_m))
        or bounds["hold"] != (movement_end, movement_end + 25)
        or result["timesteps"] != len(actual_full)
        or len(target_full) != len(actual_full)
    ):
        raise RuntimeError(f"canonical full-trial phase bounds differ: {condition.rule}")
    if not np.isfinite(actual_full).all() or not np.isfinite(target_full).all():
        raise RuntimeError(f"canonical full trial contains non-finite values: {condition.rule}")
    return {
        "condition": condition,
        "actual_full": actual_full,
        "target_full": target_full,
        "actual_movement": actual_movement,
        "target_movement": target_movement,
        "epoch_bounds": bounds,
        "timesteps": result["timesteps"],
    }


def run_validation(config_path: str | Path, source_results: str | Path) -> dict[str, Any]:
    config, dual_config, geometry = load_config(config_path)
    source_root = Path(source_results).resolve()
    manifests = _validate_source(config, dual_config, geometry, source_root)
    output = Path(config["output"]["directory"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    plot_directory = output / "plots"
    plot_directory.mkdir()

    checkpoint_hashes = {}
    policy_unchanged = {}
    all_rows = []
    for model_kind in MODEL_KINDS:
        policy, hp, digest = _load_frozen_policy(
            config, dual_config, source_root, model_kind, manifests[model_kind]
        )
        checkpoint_hashes[model_kind] = digest
        initial_state = _state_clone(policy)
        env = _make_env(dual_config, model_kind)
        expected = _candidate_arrays(config, source_root, model_kind)
        trajectories = [
            _canonical_rollout(policy, env, hp, condition, expected)
            for condition in conditions(model_kind, geometry)
        ]
        if tuple(item["condition"].rule for item in trajectories) != rule_names(model_kind):
            raise RuntimeError(f"canonical {model_kind} rule order differs")
        arrays = {}
        for item in trajectories:
            condition = item["condition"]
            prefix = condition.rule
            for name in ("actual_full", "target_full", "actual_movement", "target_movement"):
                arrays[f"{prefix}__{name}"] = item[name]
            bounds = item["epoch_bounds"]
            metrics = _movement_metrics(
                item["actual_movement"],
                item["target_movement"],
                condition.canonical_rounding,
            )
            all_rows.append(
                {
                    "model_kind": model_kind,
                    "rule": condition.rule,
                    "condition_id": condition.condition_id,
                    "checkpoint_update": config["source"]["checkpoints"][model_kind][
                        "update"
                    ],
                    "stable_start": bounds["stable"][0],
                    "stable_end": bounds["stable"][1],
                    "delay_start": bounds["delay"][0],
                    "delay_end": bounds["delay"][1],
                    "movement_start": bounds["movement"][0],
                    "movement_end": bounds["movement"][1],
                    "hold_start": bounds["hold"][0],
                    "hold_end": bounds["hold"][1],
                    "trial_timesteps": item["timesteps"],
                    "movement_samples": len(item["actual_movement"]),
                    **metrics,
                }
            )
        expected.close()
        np.savez_compressed(output / f"{model_kind}_canonical_trajectories.npz", **arrays)
        _plot(
            plot_directory / f"{model_kind}_movement_trajectories.png",
            model_kind,
            trajectories,
            "movement",
        )
        _plot(
            plot_directory / f"{model_kind}_full_trial_trajectories.png",
            model_kind,
            trajectories,
            "full",
        )
        policy_unchanged[model_kind] = all(
            torch.equal(value, policy.state_dict()[name])
            for name, value in initial_state.items()
        )
        if not policy_unchanged[model_kind]:
            raise RuntimeError(f"frozen {model_kind} policy changed during validation")

    with (output / "canonical_validation_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "total_tasks": len(all_rows),
        "stroke_tasks": sum(row["model_kind"] == "stroke" for row in all_rows),
        "move_tasks": sum(row["model_kind"] == "move" for row in all_rows),
        "source_movement_arrays_matched_exactly": len(all_rows),
        "full_trial_trajectory_archives": len(MODEL_KINDS),
        "plots": 4,
        "all_numeric_values_finite": True,
    }
    _write_json(output / "validation_summary.json", summary)
    (output / "FINAL_REPORT.md").write_text(
        "\n".join(
            [
                "# Frozen dual-rule canonical validation",
                "",
                "The selected joint-gradient Stroke/onset_window and Move/onset_window",
                "macro-best checkpoints were evaluated independently from the canonical",
                "start state for every rule. No state was carried between tasks.",
                "",
                "Each movement trace is the exact movement slice of the corresponding",
                "complete stable-delay-movement-hold trial and exactly matches the",
                "movement trace archived by the completed training run.",
                "No training or trajectory postprocessing was performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
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
        "selected_candidate": "best_macro",
        "validation_contract": config["validation"],
        "render_contract": config["render"],
        "policy_state_bitwise_unchanged": policy_unchanged,
        "source_movement_arrays_matched_exactly": True,
        "training_started": False,
        "optimizer_created": False,
        "backward_executed": False,
        "trajectory_postprocessing_performed": False,
        "behavioral_pass_fail_defined": False,
        "integrity": summary,
        "completed": True,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--source-results", required=True)
    arguments = parser.parse_args()
    result = run_validation(arguments.config, arguments.source_results)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
