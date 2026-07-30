"""Eight-stroke low-learning-rate refinement after the canonical Stage-1 review."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.canonical_overfit import (
    CANONICAL_VARIANT,
    METRIC_FIELDS,
    METRIC_NAMES,
    _checkpoint_curves,
    _load_json,
    _task_conditions,
    _train_task,
    _write_json,
    _write_metrics,
    load_canonical_overfit_config,
)
from hanzi_writing.geometry import ACTIVE_RULES, PROJECT
from hanzi_writing.training import _sha256_file


REFINEMENT_VARIANT = "canonical_stroke_refinement_v1"
REFINEMENT_PREFIX = "canonical_stroke_refinement"
REFINEMENT_TASKS = ACTIVE_RULES[:-1]
EXPECTED_SOURCE_HEAD = "6b2007b5ddd2ef914f98336007ec9d75463d97d5"
EXPECTED_METRIC_ROWS = 24
EXPECTED_PLOTS = 9
EXPECTED_TRAJECTORY_ROWS = 3924


def load_refinement_config(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    config = _load_json(path)
    expected = {
        "project": PROJECT,
        "run_kind": "canonical_stroke_refinement",
        "variant": REFINEMENT_VARIANT,
        "enabled": True,
        "base_config": (
            "configurations/"
            "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
        ),
        "active_rules": list(REFINEMENT_TASKS),
        "source_contract": {
            "variant": CANONICAL_VARIANT,
            "git_head": EXPECTED_SOURCE_HEAD,
            "completed_updates_per_task": 6000,
            "selection_candidates": [
                "best_checkpoint.pt",
                "review_checkpoint.pt",
            ],
            "selection_metric": "validation_loss",
            "selection_rule": "minimum_with_best_on_exact_tie",
        },
        "optimizer": {
            "learning_rate": 0.0001,
            "source_optimizer_state": (
                "preserve_adam_state_then_override_learning_rate"
            ),
        },
        "training": {
            "additional_updates": 2000,
            "validation_interval": 100,
            "log_interval": 100,
            "parallel_processes": 8,
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
                "canonical_stroke_refinement/dev42"
            )
        },
    }
    if config != expected:
        raise ValueError("canonical stroke-refinement configuration differs")
    base, geometry = load_canonical_overfit_config(config["base_config"])
    runtime = copy.deepcopy(base)
    runtime["run_kind"] = config["run_kind"]
    runtime["variant"] = config["variant"]
    runtime["active_rules"] = list(REFINEMENT_TASKS)
    runtime["optimizer"]["learning_rate"] = config["optimizer"]["learning_rate"]
    runtime["training"]["initial_review_updates"] = config["training"][
        "additional_updates"
    ]
    runtime["training"]["validation_interval"] = config["training"][
        "validation_interval"
    ]
    runtime["training"]["log_interval"] = config["training"]["log_interval"]
    runtime["training"].pop("move_sampler")
    runtime["diagnostic_contract"] = {
        "metrics_rows": EXPECTED_METRIC_ROWS,
        "plots": EXPECTED_PLOTS,
        "behavioral_pass_fail": "not_defined",
        "complete_character_rollout": False,
    }
    runtime["output"] = copy.deepcopy(config["output"])
    return config, runtime, geometry


def _source_provenance(
    source_directory: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = Path(source_directory)
    provenance = _load_json(source / "provenance.json")
    contract = config["source_contract"]
    summaries = provenance.get("task_summaries", [])
    completed = {
        summary.get("task"): summary.get("completed_updates")
        for summary in summaries
    }
    if (
        provenance.get("completed") is not True
        or provenance.get("variant") != contract["variant"]
        or provenance.get("git_head") != contract["git_head"]
        or provenance.get("execution_mode")
        != "nine_independent_parallel_processes"
        or any(
            completed.get(task) != contract["completed_updates_per_task"]
            for task in REFINEMENT_TASKS
        )
        or provenance.get("shared_9task_started") is not False
        or provenance.get("formal_75k_started") is not False
        or provenance.get("complete_character_rollout_started") is not False
    ):
        raise ValueError("canonical refinement source provenance differs")
    return provenance


def select_source_checkpoints(
    config_path: str | Path,
    source_directory: str | Path,
) -> dict[str, Any]:
    config, _, _ = load_refinement_config(config_path)
    source = Path(source_directory).resolve()
    provenance = _source_provenance(source, config)
    recorded_hashes = provenance.get("checkpoint_sha256", {})
    selections = []
    for task in REFINEMENT_TASKS:
        candidates = []
        for name, identity in (
            ("best_checkpoint.pt", "best"),
            ("review_checkpoint.pt", "review"),
        ):
            path = source / "models" / task / name
            checkpoint = torch.load(
                path, map_location=torch.device("cpu"), weights_only=False
            )
            expected_kind = f"canonical_single_task_{identity}"
            relative = str(path.relative_to(source)).replace("\\", "/")
            digest = _sha256_file(path)
            value = float(checkpoint.get("validation_loss", np.nan))
            if (
                checkpoint.get("project") != PROJECT
                or checkpoint.get("variant") != CANONICAL_VARIANT
                or checkpoint.get("checkpoint_kind") != expected_kind
                or checkpoint.get("task") != task
                or not np.isfinite(value)
                or recorded_hashes.get(relative) != digest
            ):
                raise ValueError(f"canonical refinement source differs: {task}/{name}")
            candidates.append(
                {
                    "checkpoint": identity,
                    "path": str(path),
                    "relative_path": relative,
                    "sha256": digest,
                    "update": int(checkpoint["update"]),
                    "validation_loss": value,
                }
            )
        selected = min(
            candidates,
            key=lambda value: (
                value["validation_loss"],
                0 if value["checkpoint"] == "best" else 1,
            ),
        )
        selections.append(
            {
                "task": task,
                "candidates": candidates,
                "selected_checkpoint": selected["checkpoint"],
                "selected_path": selected["path"],
                "selected_relative_path": selected["relative_path"],
                "selected_sha256": selected["sha256"],
                "selected_update": selected["update"],
                "selected_validation_loss": selected["validation_loss"],
            }
        )
    return {
        "project": PROJECT,
        "variant": REFINEMENT_VARIANT,
        "source_directory": str(source),
        "source_git_head": provenance["git_head"],
        "source_provenance_sha256": _sha256_file(source / "provenance.json"),
        "selection_metric": "validation_loss",
        "selection_rule": "minimum_with_best_on_exact_tie",
        "move_included": False,
        "selections": selections,
    }


def prepare_refinement(
    config_path: str | Path,
    source_directory: str | Path,
) -> dict[str, Any]:
    config, _, _ = load_refinement_config(config_path)
    manifest = select_source_checkpoints(config_path, source_directory)
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    _write_json(output / "source_selection.json", manifest)
    return {
        "output_directory": str(output),
        "source_task_count": len(manifest["selections"]),
        "additional_updates": config["training"]["additional_updates"],
        "parallel_processes": config["training"]["parallel_processes"],
    }


def _selection_for_task(
    output: Path,
    task: str,
    source_directory: str | Path,
) -> dict[str, Any]:
    manifest = _load_json(output / "source_selection.json")
    if Path(manifest["source_directory"]) != Path(source_directory).resolve():
        raise ValueError("canonical refinement worker source directory changed")
    matches = [value for value in manifest["selections"] if value["task"] == task]
    if len(matches) != 1:
        raise ValueError(f"canonical refinement source selection differs: {task}")
    selected = matches[0]
    if _sha256_file(selected["selected_path"]) != selected["selected_sha256"]:
        raise RuntimeError(f"canonical refinement source changed: {task}")
    return selected


def train_refinement_task(
    config_path: str | Path,
    source_directory: str | Path,
    task: str,
) -> dict[str, Any]:
    config, runtime, geometry = load_refinement_config(config_path)
    if task not in REFINEMENT_TASKS:
        raise ValueError(f"canonical refinement excludes task: {task}")
    output = Path(config["output"]["directory"])
    _source_provenance(source_directory, config)
    selection = _selection_for_task(output, task, source_directory)
    conditions = dict(_task_conditions(geometry))[task]
    summary = _train_task(
        task,
        conditions,
        runtime,
        geometry,
        output / "models" / task,
        target_updates=config["training"]["additional_updates"],
        initial_checkpoint_path=Path(selection["selected_path"]),
        checkpoint_variant=REFINEMENT_VARIANT,
        checkpoint_prefix=REFINEMENT_PREFIX,
    )
    return {
        **summary,
        "source_checkpoint": selection["selected_checkpoint"],
        "source_update": selection["selected_update"],
        "source_validation_loss": selection["selected_validation_loss"],
    }


def _refinement_summary(output: Path, task: str) -> dict[str, Any]:
    checkpoints = {}
    for identity in ("best", "review", "continuation"):
        checkpoint = torch.load(
            output / f"{identity}_checkpoint.pt",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if (
            checkpoint.get("project") != PROJECT
            or checkpoint.get("variant") != REFINEMENT_VARIANT
            or checkpoint.get("checkpoint_kind")
            != f"{REFINEMENT_PREFIX}_{identity}"
            or checkpoint.get("task") != task
        ):
            raise ValueError(f"canonical refinement checkpoint differs: {task}")
        checkpoints[identity] = checkpoint
    if checkpoints["review"]["update"] != checkpoints["continuation"]["update"]:
        raise ValueError(f"canonical refinement terminal checkpoints differ: {task}")
    rows = [
        json.loads(line)
        for line in (output / "validation_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    review_rows = [row for row in rows if row["validation_kind"] == "review_read_only"]
    checks = review_rows[-1].get("read_only_checks", {}) if review_rows else {}
    if (
        not review_rows
        or review_rows[-1]["update"] != checkpoints["review"]["update"]
        or not checks
        or not all(checks.values())
    ):
        raise ValueError(f"canonical refinement review evidence differs: {task}")
    return {
        "task": task,
        "best_update": int(checkpoints["best"]["update"]),
        "best_validation_loss": float(checkpoints["best"]["validation_loss"]),
        "review_update": int(checkpoints["review"]["update"]),
        "review_validation_loss": float(checkpoints["review"]["validation_loss"]),
        "completed_additional_updates": int(checkpoints["review"]["update"]) + 1,
        "review_validation_read_only_checks": checks,
    }


def _trajectory_arrays(curves: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "task": [],
        "checkpoint": [],
        "sample_index": [],
        "target_xy_m": [],
        "actual_xy_m": [],
        "euclidean_error_m": [],
        "subphase": [],
    }
    for curve in curves:
        samples = len(curve["target"])
        values["task"].append(np.full(samples, curve["row"]["task"], dtype="U12"))
        values["checkpoint"].append(
            np.full(samples, curve["row"]["checkpoint"], dtype="U12")
        )
        values["sample_index"].append(np.arange(samples, dtype=np.int64))
        values["target_xy_m"].append(curve["target"])
        values["actual_xy_m"].append(curve["actual"])
        values["euclidean_error_m"].append(
            np.linalg.norm(curve["actual"] - curve["target"], axis=1)
        )
        values["subphase"].append(curve["subphase"])
    return {name: np.concatenate(parts) for name, parts in values.items()}


def _write_trajectory_plots(output: Path, curves: list[dict[str, Any]]) -> list[Path]:
    directory = output / "plots"
    directory.mkdir(exist_ok=False)
    colors = {"source": "#4c78a8", "best": "#e45756", "review": "#f58518"}
    for task in REFINEMENT_TASKS:
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


def _write_validation_plot(output: Path, manifest: dict[str, Any]) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(12, 16))
    selections = {value["task"]: value for value in manifest["selections"]}
    for axis, task in zip(axes.flat, REFINEMENT_TASKS):
        rows = [
            json.loads(line)
            for line in (output / "models" / task / "validation_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        scheduled = [row for row in rows if row["validation_kind"] == "scheduled"]
        axis.plot(
            [row["update"] for row in scheduled],
            [row["aggregate"]["phase_normalized_position_l1"] for row in scheduled],
            marker="o",
            markersize=2,
            linewidth=1,
            label="refinement validation",
        )
        axis.axhline(
            selections[task]["selected_validation_loss"],
            color="#4c78a8",
            linestyle="--",
            label="source",
        )
        axis.set_title(task)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "plots" / "validation_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Canonical eight-stroke refinement report",
        "",
        (
            "Eight strokes were refined independently for 2000 additional updates "
            "at learning rate 0.0001. Move was excluded. No behavioral pass/fail "
            "threshold or shared-model claim is defined."
        ),
        "",
        "| task | checkpoint | validation loss | mean error (mm) | endpoint error (mm) | path ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in REFINEMENT_TASKS:
        for row in [value for value in rows if value["task"] == task]:
            lines.append(
                f"| {task} | {row['checkpoint']} | "
                f"{float(row['checkpoint_validation_loss']):.9g} | "
                f"{float(row['movement_mean_euclidean_m']) * 1000:.6g} | "
                f"{float(row['endpoint_euclidean_m']) * 1000:.6g} | "
                f"{float(row['path_length_ratio']):.6g} |"
            )
    lines.extend(["", "## Source selections", ""])
    for selection in manifest["selections"]:
        lines.append(
            f"- {selection['task']}: {selection['selected_checkpoint']} at source "
            f"update {selection['selected_update']} with validation loss "
            f"{selection['selected_validation_loss']:.9g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _git_value(*arguments: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def finalize_refinement(
    config_path: str | Path,
    source_directory: str | Path,
) -> dict[str, Any]:
    config, runtime, geometry = load_refinement_config(config_path)
    output = Path(config["output"]["directory"])
    manifest = _load_json(output / "source_selection.json")
    if Path(manifest["source_directory"]) != Path(source_directory).resolve():
        raise ValueError("canonical refinement source directory changed")
    _source_provenance(source_directory, config)
    if _sha256_file(Path(source_directory) / "provenance.json") != manifest[
        "source_provenance_sha256"
    ]:
        raise RuntimeError("canonical refinement source provenance changed")
    for selection in manifest["selections"]:
        if _sha256_file(selection["selected_path"]) != selection["selected_sha256"]:
            raise RuntimeError(
                f"canonical refinement source checkpoint changed: {selection['task']}"
            )
    expected_files = {
        "best_checkpoint.pt",
        "continuation_checkpoint.pt",
        "review_checkpoint.pt",
        "training_metrics.jsonl",
        "validation_metrics.jsonl",
    }
    task_summaries = []
    for task in REFINEMENT_TASKS:
        task_output = output / "models" / task
        if {path.name for path in task_output.iterdir()} != expected_files:
            raise RuntimeError(f"canonical refinement task output differs: {task}")
        task_summaries.append(_refinement_summary(task_output, task))
    task_conditions = dict(_task_conditions(geometry))
    selections = {value["task"]: value for value in manifest["selections"]}
    curves = []
    for task in REFINEMENT_TASKS:
        selection = selections[task]
        curves.extend(
            _checkpoint_curves(
                task,
                task_conditions[task],
                "source",
                Path(selection["selected_path"]),
                runtime,
                geometry,
                checkpoint_identity_name=selection["selected_checkpoint"],
            )
        )
        for identity in ("best", "review"):
            curves.extend(
                _checkpoint_curves(
                    task,
                    task_conditions[task],
                    identity,
                    output / "models" / task / f"{identity}_checkpoint.pt",
                    runtime,
                    geometry,
                    expected_variant=REFINEMENT_VARIANT,
                    checkpoint_prefix=REFINEMENT_PREFIX,
                )
            )
    rows = [curve["row"] for curve in curves]
    if (
        len(rows) != EXPECTED_METRIC_ROWS
        or len({(row["task"], row["checkpoint"]) for row in rows})
        != EXPECTED_METRIC_ROWS
    ):
        raise RuntimeError("canonical refinement metric rows differ")
    numeric = [
        float(value)
        for row in rows
        for name, value in row.items()
        if name in METRIC_NAMES and value is not None
    ]
    if not np.isfinite(numeric).all():
        raise RuntimeError("canonical refinement metrics contain NaN or Inf")
    _write_metrics(output / "stroke_refinement_metrics.csv", rows)
    arrays = _trajectory_arrays(curves)
    if len(arrays["sample_index"]) != EXPECTED_TRAJECTORY_ROWS:
        raise RuntimeError("canonical refinement trajectory row count differs")
    np.savez_compressed(output / "stroke_refinement_trajectories.npz", **arrays)
    trajectory_plots = _write_trajectory_plots(output, curves)
    validation_plot = _write_validation_plot(output, manifest)
    plots = [*trajectory_plots, validation_plot]
    if len(plots) != EXPECTED_PLOTS:
        raise RuntimeError("canonical refinement plot count differs")
    _write_report(output / "STROKE_REFINEMENT_REPORT.md", rows, manifest)
    checkpoint_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in sorted((output / "models").glob("*/*_checkpoint.pt"))
    }
    if len(checkpoint_hashes) != 24:
        raise RuntimeError("canonical refinement checkpoint count differs")
    provenance = {
        "project": PROJECT,
        "variant": REFINEMENT_VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "source_directory": str(Path(source_directory).resolve()),
        "source_git_head": manifest["source_git_head"],
        "source_provenance_sha256": manifest["source_provenance_sha256"],
        "source_selections": manifest["selections"],
        "learning_rate": config["optimizer"]["learning_rate"],
        "additional_updates": config["training"]["additional_updates"],
        "validation_interval": config["training"]["validation_interval"],
        "execution_mode": "eight_independent_parallel_processes",
        "task_summaries": task_summaries,
        "checkpoint_sha256": checkpoint_hashes,
        "integrity": {
            "stroke_tasks_completed": len(task_summaries),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--task", choices=REFINEMENT_TASKS)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--source", required=True)
    arguments = parser.parse_args()
    if arguments.prepare:
        result = prepare_refinement(arguments.config, arguments.source)
    elif arguments.task is not None:
        result = train_refinement_task(
            arguments.config, arguments.source, arguments.task
        )
    else:
        result = finalize_refinement(arguments.config, arguments.source)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
