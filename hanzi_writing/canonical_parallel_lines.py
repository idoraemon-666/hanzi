"""Parallel canonical scratch-low-LR and selected exact-continuation experiments."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.canonical_overfit import (
    METRIC_NAMES,
    STAGE0_ARTIFACT_NAMES,
    _checkpoint_curves,
    _load_json,
    _task_conditions,
    _train_task,
    _write_json,
    _write_metrics,
    load_canonical_overfit_config,
    require_approved_stage0,
)
from hanzi_writing.canonical_protocol import write_stage0_artifacts
from hanzi_writing.canonical_refinement import (
    REFINEMENT_PREFIX,
    REFINEMENT_TASKS,
    REFINEMENT_VARIANT,
    _trajectory_arrays,
    load_refinement_config,
)
from hanzi_writing.geometry import PROJECT
from hanzi_writing.training import _hp_from_config, _nested_equal, _sha256_file


SCRATCH_VARIANT = "canonical_eight_stroke_scratch_lr1e4_v1"
SCRATCH_PREFIX = "canonical_eight_stroke_scratch_lr1e4"
SCRATCH_TASKS = REFINEMENT_TASKS
FOLLOWUP_VARIANT = "canonical_selected_refinement_followup_v1"
FOLLOWUP_TASKS = ("heng", "pie", "na", "shugou")
EXPECTED_REFINEMENT_HEAD = "d763b59ccc09d6e8673f518f0aead89094fb2226"
MODEL_FILES = {
    "best_checkpoint.pt",
    "continuation_checkpoint.pt",
    "review_checkpoint.pt",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
}


def load_scratch_config(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    config = _load_json(path)
    expected = {
        "project": PROJECT,
        "run_kind": "canonical_eight_stroke_scratch_low_lr",
        "variant": SCRATCH_VARIANT,
        "enabled": True,
        "base_config": (
            "configurations/"
            "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
        ),
        "active_rules": list(SCRATCH_TASKS),
        "optimizer": {
            "learning_rate": 0.0001,
            "initialization": "fresh_seed_42",
        },
        "training": {
            "updates": 10000,
            "validation_interval": 100,
            "log_interval": 100,
            "parallel_processes": 8,
        },
        "diagnostic_contract": {
            "metrics_rows": 16,
            "trajectory_rows": 2616,
            "plots": 9,
            "behavioral_pass_fail": "not_defined",
        },
        "exclusions": {
            "move": True,
            "shared_9task": True,
            "formal_75k": True,
            "complete_character_rollout": True,
        },
        "output": {
            "directory": (
                "runs/hanzi_stroke_temporal_composition/"
                "canonical_stroke_scratch_lr1e4_10000/dev42"
            )
        },
    }
    if config != expected:
        raise ValueError("canonical scratch-low-LR configuration differs")
    base, geometry = load_canonical_overfit_config(config["base_config"])
    runtime = copy.deepcopy(base)
    runtime["run_kind"] = config["run_kind"]
    runtime["variant"] = config["variant"]
    runtime["active_rules"] = list(SCRATCH_TASKS)
    runtime["optimizer"]["learning_rate"] = config["optimizer"]["learning_rate"]
    runtime["training"]["initial_review_updates"] = config["training"]["updates"]
    runtime["training"]["validation_interval"] = config["training"][
        "validation_interval"
    ]
    runtime["training"]["log_interval"] = config["training"]["log_interval"]
    runtime["training"].pop("move_sampler")
    runtime["diagnostic_contract"] = copy.deepcopy(config["diagnostic_contract"])
    runtime["output"] = copy.deepcopy(config["output"])
    return config, runtime, geometry


def load_followup_config(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    config = _load_json(path)
    expected = {
        "project": PROJECT,
        "run_kind": "canonical_selected_refinement_followup",
        "variant": FOLLOWUP_VARIANT,
        "enabled": True,
        "base_config": (
            "configurations/"
            "hanzi_stroke_temporal_composition_canonical_stroke_refinement_v1.json"
        ),
        "active_rules": list(FOLLOWUP_TASKS),
        "source_contract": {
            "variant": REFINEMENT_VARIANT,
            "git_head": EXPECTED_REFINEMENT_HEAD,
            "completed_additional_updates": 2000,
            "learning_rate": 0.0001,
            "continuation_checkpoint": (
                "exact_optimizer_global_rng_environment_rng"
            ),
        },
        "optimizer": {
            "learning_rate": 0.0001,
            "source_optimizer_state": "preserve_exact_continuation_state",
        },
        "training": {
            "additional_updates": 2000,
            "target_total_refinement_updates": 4000,
            "validation_interval": 100,
            "log_interval": 100,
            "parallel_processes": 4,
        },
        "diagnostic_contract": {
            "metrics_rows": 12,
            "trajectory_rows": 1962,
            "plots": 5,
            "behavioral_pass_fail": "not_defined",
        },
        "selection_basis": {
            "heng": "best scheduled validation occurred at refinement update 1900",
            "pie": "terminal review improved beyond the scheduled best",
            "na": "terminal review improved beyond the scheduled best",
            "shugou": "best scheduled validation occurred at refinement update 1900",
        },
        "exclusions": {
            "move": True,
            "shu": True,
            "dian": True,
            "ti": True,
            "hengzhe": True,
            "shared_9task": True,
            "formal_75k": True,
            "complete_character_rollout": True,
        },
        "output": {
            "directory": (
                "runs/hanzi_stroke_temporal_composition/"
                "canonical_stroke_refinement_followup/dev42"
            )
        },
    }
    if config != expected:
        raise ValueError("canonical selected-refinement follow-up configuration differs")
    _, runtime, geometry = load_refinement_config(config["base_config"])
    if (
        runtime["variant"] != REFINEMENT_VARIANT
        or runtime["optimizer"]["learning_rate"]
        != config["optimizer"]["learning_rate"]
        or runtime["training"]["initial_review_updates"]
        != config["source_contract"]["completed_additional_updates"]
        or runtime["training"]["validation_interval"]
        != config["training"]["validation_interval"]
        or runtime["training"]["log_interval"]
        != config["training"]["log_interval"]
    ):
        raise ValueError("canonical follow-up runtime differs from refinement")
    return config, runtime, geometry


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _task_summary(
    directory: Path,
    task: str,
    expected_variant: str,
    checkpoint_prefix: str,
    expected_updates: int,
) -> dict[str, Any]:
    checkpoints = {}
    for identity in ("best", "review", "continuation"):
        checkpoint = torch.load(
            directory / f"{identity}_checkpoint.pt",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if (
            checkpoint.get("project") != PROJECT
            or checkpoint.get("variant") != expected_variant
            or checkpoint.get("checkpoint_kind")
            != f"{checkpoint_prefix}_{identity}"
            or checkpoint.get("task") != task
        ):
            raise ValueError(f"canonical parallel-line checkpoint differs: {task}")
        checkpoints[identity] = checkpoint
    terminal_update = expected_updates - 1
    if (
        checkpoints["review"].get("update") != terminal_update
        or checkpoints["continuation"].get("update") != terminal_update
        or (checkpoints["continuation"].get("continuation_state") or {}).get(
            "next_update"
        )
        != expected_updates
        or int(checkpoints["best"].get("update", -1)) >= expected_updates
    ):
        raise ValueError(f"canonical parallel-line update identity differs: {task}")
    validation_rows = [
        json.loads(line)
        for line in (directory / "validation_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    reviews = [row for row in validation_rows if row["validation_kind"] == "review_read_only"]
    checks = reviews[-1].get("read_only_checks", {}) if reviews else {}
    if (
        not reviews
        or reviews[-1].get("update") != terminal_update
        or not checks
        or not all(checks.values())
    ):
        raise ValueError(f"canonical parallel-line review evidence differs: {task}")
    return {
        "task": task,
        "best_update": int(checkpoints["best"]["update"]),
        "best_validation_loss": float(checkpoints["best"]["validation_loss"]),
        "review_update": int(checkpoints["review"]["update"]),
        "review_validation_loss": float(checkpoints["review"]["validation_loss"]),
        "completed_updates": expected_updates,
        "review_validation_read_only_checks": checks,
    }


def _validate_refinement_task_files(
    directory: Path,
    task: str,
    runtime: dict[str, Any],
    expected_updates: int,
) -> dict[str, str]:
    if {path.name for path in directory.iterdir()} != MODEL_FILES:
        raise ValueError(f"canonical refinement source file set differs: {task}")
    hp = _hp_from_config(runtime)
    checkpoints = {}
    for identity in ("best", "review", "continuation"):
        path = directory / f"{identity}_checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("project") != PROJECT
            or checkpoint.get("variant") != REFINEMENT_VARIANT
            or checkpoint.get("checkpoint_kind") != f"{REFINEMENT_PREFIX}_{identity}"
            or checkpoint.get("task") != task
            or not _nested_equal(checkpoint.get("hp"), hp)
        ):
            raise ValueError(f"canonical refinement source checkpoint differs: {task}")
        checkpoints[identity] = checkpoint
    if (
        checkpoints["review"].get("update") != expected_updates - 1
        or checkpoints["continuation"].get("update") != expected_updates - 1
        or (checkpoints["continuation"].get("continuation_state") or {}).get(
            "next_update"
        )
        != expected_updates
    ):
        raise ValueError(f"canonical refinement source update differs: {task}")
    return {path.name: _sha256_file(path) for path in sorted(directory.iterdir())}


def _source_refinement_provenance(
    source_directory: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = Path(source_directory)
    provenance = _load_json(source / "provenance.json")
    completed = {
        row.get("task"): row.get("completed_additional_updates")
        for row in provenance.get("task_summaries", [])
    }
    contract = config["source_contract"]
    if (
        provenance.get("completed") is not True
        or provenance.get("variant") != contract["variant"]
        or provenance.get("git_head") != contract["git_head"]
        or provenance.get("learning_rate") != contract["learning_rate"]
        or any(
            completed.get(task) != contract["completed_additional_updates"]
            for task in REFINEMENT_TASKS
        )
        or provenance.get("move_started") is not False
        or provenance.get("shared_9task_started") is not False
        or provenance.get("formal_75k_started") is not False
        or provenance.get("complete_character_rollout_started") is not False
    ):
        raise ValueError("canonical selected follow-up source provenance differs")
    return provenance


def prepare_scratch(
    config_path: str | Path,
    approved_stage0_directory: str | Path,
) -> dict[str, Any]:
    config, _, geometry = load_scratch_config(config_path)
    output = Path(config["output"]["directory"])
    stage0 = write_stage0_artifacts(geometry, output)
    approved_hashes = require_approved_stage0(output, approved_stage0_directory)
    (output / "models").mkdir(exist_ok=False)
    return {
        "output_directory": str(output),
        "updates": config["training"]["updates"],
        "parallel_processes": config["training"]["parallel_processes"],
        "stage0": stage0,
        "approved_stage0_sha256": approved_hashes,
    }


def train_scratch_task(config_path: str | Path, task: str) -> dict[str, Any]:
    config, runtime, geometry = load_scratch_config(config_path)
    if task not in SCRATCH_TASKS:
        raise ValueError(f"canonical scratch-low-LR excludes task: {task}")
    output = Path(config["output"]["directory"])
    if any(not (output / name).is_file() for name in STAGE0_ARTIFACT_NAMES):
        raise FileNotFoundError("canonical scratch-low-LR Stage-0 evidence is missing")
    return _train_task(
        task,
        dict(_task_conditions(geometry))[task],
        runtime,
        geometry,
        output / "models" / task,
        target_updates=config["training"]["updates"],
        checkpoint_variant=SCRATCH_VARIANT,
        checkpoint_prefix=SCRATCH_PREFIX,
    )


def prepare_followup(
    config_path: str | Path,
    source_directory: str | Path,
) -> dict[str, Any]:
    config, runtime, _ = load_followup_config(config_path)
    source = Path(source_directory).resolve()
    provenance = _source_refinement_provenance(source, config)
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=False)
    model_root = output / "models"
    snapshot_root = output / "source_checkpoints"
    model_root.mkdir()
    snapshot_root.mkdir()
    tasks = []
    recorded = provenance["checkpoint_sha256"]
    source_updates = config["source_contract"]["completed_additional_updates"]
    for task in FOLLOWUP_TASKS:
        source_task = source / "models" / task
        hashes = _validate_refinement_task_files(
            source_task, task, runtime, source_updates
        )
        for identity in ("best", "review", "continuation"):
            relative = f"models/{task}/{identity}_checkpoint.pt"
            if recorded.get(relative) != hashes[f"{identity}_checkpoint.pt"]:
                raise ValueError(f"canonical refinement provenance hash differs: {task}")
        target_task = model_root / task
        shutil.copytree(source_task, target_task)
        snapshot_task = snapshot_root / task
        snapshot_task.mkdir()
        snapshot = snapshot_task / "review_checkpoint.pt"
        shutil.copy2(source_task / "review_checkpoint.pt", snapshot)
        tasks.append(
            {
                "task": task,
                "source_file_sha256": hashes,
                "source_review_checkpoint": str(snapshot),
                "source_review_sha256": _sha256_file(snapshot),
            }
        )
    manifest = {
        "project": PROJECT,
        "variant": FOLLOWUP_VARIANT,
        "source_directory": str(source),
        "source_git_head": provenance["git_head"],
        "source_provenance_sha256": _sha256_file(source / "provenance.json"),
        "source_completed_additional_updates": source_updates,
        "additional_updates": config["training"]["additional_updates"],
        "target_total_refinement_updates": config["training"][
            "target_total_refinement_updates"
        ],
        "exact_continuation": True,
        "tasks": tasks,
    }
    _write_json(output / "continuation_source.json", manifest)
    return {
        "output_directory": str(output),
        "source_task_count": len(tasks),
        "additional_updates": config["training"]["additional_updates"],
        "target_total_refinement_updates": config["training"][
            "target_total_refinement_updates"
        ],
        "parallel_processes": config["training"]["parallel_processes"],
    }


def _followup_manifest(
    output: Path,
    source_directory: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(output / "continuation_source.json")
    source = Path(source_directory).resolve()
    if (
        Path(manifest["source_directory"]) != source
        or _sha256_file(source / "provenance.json")
        != manifest["source_provenance_sha256"]
        or manifest["target_total_refinement_updates"]
        != config["training"]["target_total_refinement_updates"]
        or {row["task"] for row in manifest["tasks"]} != set(FOLLOWUP_TASKS)
    ):
        raise ValueError("canonical follow-up continuation manifest differs")
    return manifest


def train_followup_task(
    config_path: str | Path,
    source_directory: str | Path,
    task: str,
) -> dict[str, Any]:
    config, runtime, geometry = load_followup_config(config_path)
    if task not in FOLLOWUP_TASKS:
        raise ValueError(f"canonical selected follow-up excludes task: {task}")
    _source_refinement_provenance(source_directory, config)
    output = Path(config["output"]["directory"])
    manifest = _followup_manifest(output, source_directory, config)
    source_row = next(row for row in manifest["tasks"] if row["task"] == task)
    source_task = Path(source_directory).resolve() / "models" / task
    if any(
        _sha256_file(source_task / name) != digest
        for name, digest in source_row["source_file_sha256"].items()
    ):
        raise RuntimeError(f"canonical follow-up source changed: {task}")
    model_output = output / "models" / task
    continuation = torch.load(
        model_output / "continuation_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    next_update = (continuation.get("continuation_state") or {}).get("next_update")
    source_updates = config["source_contract"]["completed_additional_updates"]
    if next_update == source_updates and any(
        _sha256_file(model_output / name) != digest
        for name, digest in source_row["source_file_sha256"].items()
    ):
        raise RuntimeError(f"canonical follow-up copied source differs: {task}")
    return _train_task(
        task,
        dict(_task_conditions(geometry))[task],
        runtime,
        geometry,
        model_output,
        target_updates=config["training"]["target_total_refinement_updates"],
        checkpoint_variant=REFINEMENT_VARIANT,
        checkpoint_prefix=REFINEMENT_PREFIX,
    )


def _write_plots(
    output: Path,
    curves: list[dict[str, Any]],
    tasks: tuple[str, ...],
    colors: dict[str, str],
) -> list[Path]:
    directory = output / "plots"
    directory.mkdir(exist_ok=False)
    for task in tasks:
        selected = [curve for curve in curves if curve["row"]["task"] == task]
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot(
            selected[0]["target"][:, 0],
            selected[0]["target"][:, 1],
            color="0.65",
            label="target",
        )
        for curve in selected:
            checkpoint = curve["row"]["checkpoint"]
            axis.plot(
                curve["actual"][:, 0],
                curve["actual"][:, 1],
                color=colors[checkpoint],
                label=checkpoint,
            )
        axis.set_title(task)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / f"{task}.png", dpi=180)
        plt.close(figure)
    return sorted(directory.glob("*.png"))


def _validation_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_validation_plot(
    output: Path,
    tasks: tuple[str, ...],
    *,
    continuation_boundary: int | None = None,
    source_losses: dict[str, float] | None = None,
) -> Path:
    rows_count = (len(tasks) + 1) // 2
    figure, axes = plt.subplots(rows_count, 2, figsize=(12, 4 * rows_count))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis, task in zip(axes_array, tasks):
        rows = _validation_rows(output / "models" / task / "validation_metrics.jsonl")
        scheduled = [row for row in rows if row["validation_kind"] == "scheduled"]
        axis.plot(
            [row["update"] for row in scheduled],
            [row["aggregate"]["phase_normalized_position_l1"] for row in scheduled],
            marker="o",
            markersize=2,
            linewidth=1,
            label="validation",
        )
        if continuation_boundary is not None:
            axis.axvline(
                continuation_boundary,
                color="0.35",
                linestyle=":",
                label="follow-up start",
            )
        if source_losses is not None:
            axis.axhline(
                source_losses[task],
                color="#4c78a8",
                linestyle="--",
                label="source review",
            )
        axis.set_title(task)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes_array[len(tasks) :]:
        axis.set_visible(False)
    figure.tight_layout()
    path = output / "plots" / "validation_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_report(
    path: Path,
    title: str,
    description: str,
    rows: list[dict[str, Any]],
    tasks: tuple[str, ...],
) -> None:
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| task | checkpoint | update | validation loss | mean error (mm) | endpoint error (mm) | path ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for row in [value for value in rows if value["task"] == task]:
            lines.append(
                f"| {task} | {row['checkpoint']} | {row['checkpoint_update']} | "
                f"{float(row['checkpoint_validation_loss']):.9g} | "
                f"{float(row['movement_mean_euclidean_m']) * 1000:.6g} | "
                f"{float(row['endpoint_euclidean_m']) * 1000:.6g} | "
                f"{float(row['path_length_ratio']):.6g} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _finite_metrics(rows: list[dict[str, Any]], expected_rows: int) -> None:
    numeric = [
        float(value)
        for row in rows
        for name, value in row.items()
        if name in METRIC_NAMES and value is not None
    ]
    if len(rows) != expected_rows or not np.isfinite(numeric).all():
        raise RuntimeError("canonical parallel-line metrics differ or are non-finite")


def finalize_scratch(
    config_path: str | Path,
    approved_stage0_directory: str | Path,
) -> dict[str, Any]:
    config, runtime, geometry = load_scratch_config(config_path)
    output = Path(config["output"]["directory"])
    approved_hashes = require_approved_stage0(output, approved_stage0_directory)
    model_root = output / "models"
    summaries = []
    conditions = dict(_task_conditions(geometry))
    curves = []
    for task in SCRATCH_TASKS:
        task_output = model_root / task
        if {path.name for path in task_output.iterdir()} != MODEL_FILES:
            raise RuntimeError(f"canonical scratch task output differs: {task}")
        summaries.append(
            _task_summary(
                task_output,
                task,
                SCRATCH_VARIANT,
                SCRATCH_PREFIX,
                config["training"]["updates"],
            )
        )
        for identity in ("best", "review"):
            curves.extend(
                _checkpoint_curves(
                    task,
                    conditions[task],
                    identity,
                    task_output / f"{identity}_checkpoint.pt",
                    runtime,
                    geometry,
                    expected_variant=SCRATCH_VARIANT,
                    checkpoint_prefix=SCRATCH_PREFIX,
                )
            )
    rows = [curve["row"] for curve in curves]
    expected = config["diagnostic_contract"]
    _finite_metrics(rows, expected["metrics_rows"])
    _write_metrics(output / "scratch_low_lr_metrics.csv", rows)
    arrays = _trajectory_arrays(curves)
    if len(arrays["sample_index"]) != expected["trajectory_rows"]:
        raise RuntimeError("canonical scratch trajectory rows differ")
    np.savez_compressed(output / "scratch_low_lr_trajectories.npz", **arrays)
    plots = _write_plots(
        output, curves, SCRATCH_TASKS, {"best": "#e45756", "review": "#f58518"}
    )
    plots.append(_write_validation_plot(output, SCRATCH_TASKS))
    if len(plots) != expected["plots"]:
        raise RuntimeError("canonical scratch plot count differs")
    _write_report(
        output / "SCRATCH_LOW_LR_REPORT.md",
        "Canonical eight-stroke scratch low-LR report",
        (
            "Eight independent stroke tasks started from the same fresh seed-42 "
            "initialization and trained for 10000 updates at learning rate 0.0001. "
            "Move was excluded and no behavioral pass/fail threshold is defined."
        ),
        rows,
        SCRATCH_TASKS,
    )
    checkpoint_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(model_root.glob("*/*_checkpoint.pt"))
    }
    if len(checkpoint_hashes) != 24:
        raise RuntimeError("canonical scratch checkpoint count differs")
    provenance = {
        "project": PROJECT,
        "variant": SCRATCH_VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "approved_stage0_directory": str(Path(approved_stage0_directory)),
        "approved_stage0_sha256": approved_hashes,
        "seed": runtime["seed"],
        "learning_rate": config["optimizer"]["learning_rate"],
        "from_scratch": True,
        "updates": config["training"]["updates"],
        "validation_interval": config["training"]["validation_interval"],
        "execution_mode": "eight_independent_parallel_processes",
        "task_summaries": summaries,
        "checkpoint_sha256": checkpoint_hashes,
        "integrity": {
            "stroke_tasks_completed": len(summaries),
            "metrics_rows": len(rows),
            "trajectory_rows": len(arrays["sample_index"]),
            "plots": len(plots),
            "all_numeric_values_finite": True,
        },
        "move_started": False,
        "shared_9task_started": False,
        "formal_75k_started": False,
        "complete_character_rollout_started": False,
        "behavioral_pass_fail_defined": False,
        "completed": True,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def finalize_followup(
    config_path: str | Path,
    source_directory: str | Path,
) -> dict[str, Any]:
    config, runtime, geometry = load_followup_config(config_path)
    source_provenance = _source_refinement_provenance(source_directory, config)
    output = Path(config["output"]["directory"])
    manifest = _followup_manifest(output, source_directory, config)
    source_rows = {row["task"]: row for row in manifest["tasks"]}
    conditions = dict(_task_conditions(geometry))
    summaries = []
    curves = []
    target_updates = config["training"]["target_total_refinement_updates"]
    for task in FOLLOWUP_TASKS:
        task_output = output / "models" / task
        if {path.name for path in task_output.iterdir()} != MODEL_FILES:
            raise RuntimeError(f"canonical follow-up task output differs: {task}")
        summaries.append(
            _task_summary(
                task_output,
                task,
                REFINEMENT_VARIANT,
                REFINEMENT_PREFIX,
                target_updates,
            )
        )
        source_checkpoint = Path(source_rows[task]["source_review_checkpoint"])
        if _sha256_file(source_checkpoint) != source_rows[task]["source_review_sha256"]:
            raise RuntimeError(f"canonical follow-up source snapshot changed: {task}")
        curves.extend(
            _checkpoint_curves(
                task,
                conditions[task],
                "source",
                source_checkpoint,
                runtime,
                geometry,
                checkpoint_identity_name="review",
                expected_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
        )
        for identity in ("best", "review"):
            curves.extend(
                _checkpoint_curves(
                    task,
                    conditions[task],
                    identity,
                    task_output / f"{identity}_checkpoint.pt",
                    runtime,
                    geometry,
                    expected_variant=REFINEMENT_VARIANT,
                    checkpoint_prefix=REFINEMENT_PREFIX,
                )
            )
    rows = [curve["row"] for curve in curves]
    expected = config["diagnostic_contract"]
    _finite_metrics(rows, expected["metrics_rows"])
    _write_metrics(output / "selected_followup_metrics.csv", rows)
    arrays = _trajectory_arrays(curves)
    if len(arrays["sample_index"]) != expected["trajectory_rows"]:
        raise RuntimeError("canonical follow-up trajectory rows differ")
    np.savez_compressed(output / "selected_followup_trajectories.npz", **arrays)
    plots = _write_plots(
        output,
        curves,
        FOLLOWUP_TASKS,
        {"source": "#4c78a8", "best": "#e45756", "review": "#f58518"},
    )
    source_losses = {
        task: float(
            torch.load(
                Path(source_rows[task]["source_review_checkpoint"]),
                map_location="cpu",
                weights_only=False,
            )["validation_loss"]
        )
        for task in FOLLOWUP_TASKS
    }
    plots.append(
        _write_validation_plot(
            output,
            FOLLOWUP_TASKS,
            continuation_boundary=config["source_contract"][
                "completed_additional_updates"
            ],
            source_losses=source_losses,
        )
    )
    if len(plots) != expected["plots"]:
        raise RuntimeError("canonical follow-up plot count differs")
    _write_report(
        output / "SELECTED_REFINEMENT_FOLLOWUP_REPORT.md",
        "Canonical selected refinement follow-up report",
        (
            "Heng, pie, na, and shugou continued exactly from their 2000-update "
            "refinement continuation states for 2000 more updates at learning rate "
            "0.0001. No behavioral pass/fail threshold is defined."
        ),
        rows,
        FOLLOWUP_TASKS,
    )
    checkpoint_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(output.glob("models/*/*_checkpoint.pt"))
    }
    source_checkpoint_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(output.glob("source_checkpoints/*/*.pt"))
    }
    if len(checkpoint_hashes) != 12 or len(source_checkpoint_hashes) != 4:
        raise RuntimeError("canonical follow-up checkpoint count differs")
    provenance = {
        "project": PROJECT,
        "variant": FOLLOWUP_VARIANT,
        "checkpoint_variant": REFINEMENT_VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "source_directory": str(Path(source_directory).resolve()),
        "source_git_head": source_provenance["git_head"],
        "source_provenance_sha256": manifest["source_provenance_sha256"],
        "selected_tasks": list(FOLLOWUP_TASKS),
        "selection_basis": config["selection_basis"],
        "learning_rate": config["optimizer"]["learning_rate"],
        "source_completed_additional_updates": config["source_contract"][
            "completed_additional_updates"
        ],
        "additional_updates": config["training"]["additional_updates"],
        "target_total_refinement_updates": target_updates,
        "exact_continuation": True,
        "execution_mode": "four_independent_parallel_processes",
        "task_summaries": summaries,
        "checkpoint_sha256": checkpoint_hashes,
        "source_checkpoint_sha256": source_checkpoint_hashes,
        "integrity": {
            "stroke_tasks_completed": len(summaries),
            "metrics_rows": len(rows),
            "trajectory_rows": len(arrays["sample_index"]),
            "plots": len(plots),
            "all_numeric_values_finite": True,
        },
        "move_started": False,
        "excluded_strokes_started": False,
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
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("scratch-prepare", "scratch-finalize"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--approved-stage0", required=True)
    scratch_task = subparsers.add_parser("scratch-task")
    scratch_task.add_argument("--config", required=True)
    scratch_task.add_argument("--task", required=True, choices=SCRATCH_TASKS)
    for mode in ("followup-prepare", "followup-finalize"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--source", required=True)
    followup_task = subparsers.add_parser("followup-task")
    followup_task.add_argument("--config", required=True)
    followup_task.add_argument("--source", required=True)
    followup_task.add_argument("--task", required=True, choices=FOLLOWUP_TASKS)
    arguments = parser.parse_args()
    if arguments.mode == "scratch-prepare":
        result = prepare_scratch(arguments.config, arguments.approved_stage0)
    elif arguments.mode == "scratch-task":
        result = train_scratch_task(arguments.config, arguments.task)
    elif arguments.mode == "scratch-finalize":
        result = finalize_scratch(arguments.config, arguments.approved_stage0)
    elif arguments.mode == "followup-prepare":
        result = prepare_followup(arguments.config, arguments.source)
    elif arguments.mode == "followup-task":
        result = train_followup_task(arguments.config, arguments.source, arguments.task)
    else:
        result = finalize_followup(arguments.config, arguments.source)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
