"""Matched two-loss, eight-stroke experiment for canonical start drift."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hanzi_writing.canonical_checkpoint_review import (
    COMPOUND_COMPARISON_METRICS,
    GENERAL_COMPARISON_METRICS,
    TASKS,
    _load_source,
    _validate_candidates,
    load_config as load_review_config,
)
from hanzi_writing.canonical_overfit import (
    METRIC_FIELDS,
    _checkpoint_curves,
    _load_json,
    _task_conditions,
    _train_task,
    _write_json,
    load_canonical_overfit_config,
)
from hanzi_writing.canonical_shugou_checkpoint_experiment import (
    _verified_checkpoint,
    _write_candidate_metrics,
    _write_rankings,
    rank_task_candidates,
)
from hanzi_writing.geometry import PROJECT
from hanzi_writing.training import _sha256_file
from losses import (
    ONSET_DELAY_LAST_STEPS,
    ONSET_DELAY_WEIGHTS,
    ONSET_MOVEMENT_FIRST_STEPS,
    ONSET_MOVEMENT_WEIGHTS,
    PHASE_WEIGHTS,
)


VARIANT = "canonical_start_loss_comparison_v1"
ARM_NAMES = ("full_trial", "onset_window")
PHASE_NAMES = ("phase1", "phase2")
CONFIG_PATH = (
    "configurations/"
    "hanzi_stroke_temporal_composition_canonical_start_loss_comparison_v1.json"
)


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ")


def load_config(path: str | Path) -> dict[str, Any]:
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
            "base_config",
            "checkpoint_review_config",
            "tasks",
            "baseline",
            "arms",
            "optimizer",
            "training",
            "onset_window",
            "selection",
            "integrity_contract",
            "exclusions",
            "output",
        },
        "start-loss comparison configuration",
    )
    if (
        config["project"],
        config["run_kind"],
        config["variant"],
        config["enabled"],
        config["seed"],
        config["validation_seed"],
    ) != (
        PROJECT,
        "canonical_start_loss_comparison",
        VARIANT,
        True,
        42,
        1042,
    ):
        raise ValueError("start-loss comparison identity differs")
    if config["base_config"] != (
        "configurations/"
        "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
    ):
        raise ValueError("start-loss comparison base config differs")
    if config["checkpoint_review_config"] != (
        "configurations/"
        "hanzi_stroke_temporal_composition_canonical_checkpoint_metric_review_v1.json"
    ):
        raise ValueError("start-loss comparison review config differs")
    if tuple(config["tasks"]) != TASKS:
        raise ValueError("start-loss comparison tasks differ")
    if config["baseline"] != {
        "training_position_objective": "phase_normalized_l1",
        "reuse_existing_checkpoints": True,
        "rerun": False,
        "shugou_control_extension": {
            "directory": (
                "runs/hanzi_stroke_temporal_composition/"
                "canonical_shugou_checkpoint_experiment/dev42"
            ),
            "metrics_file": "shugou_candidate_metrics.csv",
            "variant": "canonical_shugou_checkpoint_experiment_v1",
            "git_head": "de9f4b070396c4e8c80be00089dee40a458c1a78",
            "provenance_sha256": (
                "0a85cd592163dfb9108b6f90b00feb6c7924b468f98ac9f47931ded5d36cf492"
            ),
            "included_arm": "control",
            "checkpoint_variant": "canonical_shugou_checkpoint_control_v1",
            "checkpoint_prefix": "canonical_shugou_checkpoint_control",
            "expected_control_rows": 21,
        },
    }:
        raise ValueError("start-loss comparison baseline contract differs")
    if config["arms"] != {
        "full_trial": {
            "training_position_objective": "full_trial_l1",
            "phase1_variant": "canonical_full_trial_loss_phase1_v1",
            "phase1_checkpoint_prefix": "canonical_full_trial_loss_phase1",
            "phase2_variant": "canonical_full_trial_loss_phase2_v1",
            "phase2_checkpoint_prefix": "canonical_full_trial_loss_phase2",
        },
        "onset_window": {
            "training_position_objective": "onset_window_phase_normalized_l1",
            "phase1_variant": "canonical_onset_window_loss_phase1_v1",
            "phase1_checkpoint_prefix": "canonical_onset_window_loss_phase1",
            "phase2_variant": "canonical_onset_window_loss_phase2_v1",
            "phase2_checkpoint_prefix": "canonical_onset_window_loss_phase2",
        },
    }:
        raise ValueError("start-loss comparison arms differ")
    if config["optimizer"] != {
        "phase1_learning_rate": 0.001,
        "phase2_learning_rate": 0.0001,
        "phase2_state": (
            "preserve_phase1_policy_adam_and_rng_then_override_learning_rate"
        ),
    }:
        raise ValueError("start-loss comparison optimizer differs")
    if config["training"] != {
        "batch_size": 1,
        "phase1_updates": 6000,
        "phase2_updates": 2000,
        "total_updates_per_worker": 8000,
        "phase1_validation_interval": 250,
        "phase2_validation_interval": 100,
        "log_interval": 100,
        "parallel_processes": 16,
        "initialization": "fresh_seed_42_matched_between_arms",
        "delay_schedule": "matched_seeded_random_choice_25_50_75",
    }:
        raise ValueError("start-loss comparison training schedule differs")
    if config["onset_window"] != {
        "outer_phase_weights": {
            "stable": 0.1,
            "delay": 0.1,
            "movement": 0.6,
            "hold": 0.2,
        },
        "delay_internal_weights": {
            "whole_delay": 0.5,
            "last_10_steps": 0.5,
        },
        "movement_internal_weights": {
            "whole_movement": 0.9,
            "first_5_steps": 0.1,
        },
        "changed_factor": "delay_and_movement_internal_aggregation_only",
    }:
        raise ValueError("start-loss onset-window contract differs")
    if (
        PHASE_WEIGHTS
        != config["onset_window"]["outer_phase_weights"]
        or ONSET_DELAY_LAST_STEPS != 10
        or ONSET_DELAY_WEIGHTS != (0.5, 0.5)
        or ONSET_MOVEMENT_FIRST_STEPS != 5
        or ONSET_MOVEMENT_WEIGHTS != (0.9, 0.1)
    ):
        raise ValueError("start-loss implementation differs from its config")
    if config["selection"] != {
        "all_task_metrics": list(GENERAL_COMPARISON_METRICS),
        "compound_only_metrics": list(COMPOUND_COMPARISON_METRICS),
        "metric_objective": "minimum",
        "scale_normalization": "competition_rank_fraction",
        "pareto_frontier_required": True,
        "aggregate_rule": (
            "minimize_worst_rank_then_mean_rank_then_validation_loss_then_candidate_id"
        ),
        "behavioral_pass_fail": "not_defined",
    }:
        raise ValueError("start-loss checkpoint selection differs")
    if config["integrity_contract"] != {
        "baseline_candidate_rows": 56,
        "baseline_selected_tasks": 8,
        "shugou_control_candidate_rows": 21,
        "shugou_baseline_extension_rows": 22,
        "parallel_workers": 16,
        "phase1_scheduled_checkpoints_per_worker": 24,
        "phase2_scheduled_checkpoints_per_worker": 20,
        "candidate_checkpoints_per_worker": 46,
        "new_candidate_rows": 736,
        "champion_rows": 24,
        "plots": 8,
    }:
        raise ValueError("start-loss comparison integrity contract differs")
    if config["exclusions"] != {
        "move": True,
        "shared_8task": True,
        "shared_9task": True,
        "formal_75k": True,
        "complete_character_rollout": True,
    }:
        raise ValueError("start-loss comparison exclusions differ")
    if config["output"] != {
        "directory": (
            "runs/hanzi_stroke_temporal_composition/"
            "canonical_start_loss_comparison/dev42"
        )
    }:
        raise ValueError("start-loss comparison output differs")
    load_canonical_overfit_config(config["base_config"])
    load_review_config(config["checkpoint_review_config"])
    return config


def _metric_values(row: dict[str, Any]) -> dict[str, float]:
    metrics = list(GENERAL_COMPARISON_METRICS)
    if row["task"] in {"hengzhe", "shugou"}:
        metrics.extend(COMPOUND_COMPARISON_METRICS)
    values = {}
    for metric in metrics:
        if metric == "path_length_ratio_abs_error":
            value = abs(float(row["path_length_ratio"]) - 1.0)
        else:
            value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite start-loss metric: {row['task']}/{metric}")
        values[metric] = value
    return values


def _typed_metric_row(raw: dict[str, str]) -> dict[str, Any]:
    integer_fields = {
        "source_component_index",
        "checkpoint_update",
        "movement_intervals",
        "movement_samples",
    }
    text_fields = {
        "task",
        "category",
        "rule",
        "condition_id",
        "source_character",
        "checkpoint",
    }
    row = {}
    for field in METRIC_FIELDS:
        value = raw[field]
        if field in text_fields:
            row[field] = value
        elif field in integer_fields:
            row[field] = int(value)
        else:
            row[field] = None if value == "" else float(value)
    return row


def _load_shugou_control_candidates(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = config["baseline"]["shugou_control_extension"]
    directory = Path(source["directory"]).resolve()
    provenance_path = directory / "provenance.json"
    provenance = _load_json(provenance_path)
    if (
        _sha256_file(provenance_path) != source["provenance_sha256"]
        or provenance.get("variant") != source["variant"]
        or provenance.get("git_head") != source["git_head"]
        or provenance.get("completed") is not True
        or provenance.get("recommended_checkpoint_sha256") is None
    ):
        raise ValueError("start-loss shugou control provenance differs")
    hashes = provenance.get("checkpoint_sha256", {})
    rows = []
    with (directory / source["metrics_file"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for raw in csv.DictReader(handle):
            if raw["arm"] != source["included_arm"]:
                continue
            path = Path(raw["checkpoint_path"])
            if not path.is_absolute():
                path = path.resolve()
            relative = str(path.relative_to(directory)).replace("\\", "/")
            digest = _sha256_file(path)
            if hashes.get(relative) != digest:
                raise ValueError(
                    f"start-loss shugou control checkpoint hash differs: {relative}"
                )
            checkpoint = torch.load(
                path, map_location=torch.device("cpu"), weights_only=False
            )
            kind = checkpoint.get("checkpoint_kind")
            expected_prefix = source["checkpoint_prefix"]
            if kind == f"{expected_prefix}_candidate":
                identity = "candidate"
            elif kind == f"{expected_prefix}_review":
                identity = "review"
            else:
                raise ValueError(
                    f"start-loss shugou control checkpoint kind differs: {relative}"
                )
            row = {
                "candidate_id": raw["candidate_id"],
                "arm": "baseline",
                "stage": "existing_shugou_control",
                "completed_updates": None,
                "checkpoint_path": str(path),
                "checkpoint_identity": identity,
                "expected_variant": source["checkpoint_variant"],
                "checkpoint_prefix": expected_prefix,
                **_typed_metric_row(raw),
            }
            row["_values"] = _metric_values(row)
            if (
                checkpoint.get("project") != PROJECT
                or checkpoint.get("variant") != source["checkpoint_variant"]
                or checkpoint.get("task") != "shugou"
                or checkpoint.get("training_position_objective")
                != config["baseline"]["training_position_objective"]
                or int(checkpoint.get("update", -1)) != row["checkpoint_update"]
                or not math.isclose(
                    float(checkpoint.get("validation_loss", math.nan)),
                    row["checkpoint_validation_loss"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"start-loss shugou control checkpoint identity differs: {relative}"
                )
            rows.append(row)
    if len(rows) != source["expected_control_rows"]:
        raise ValueError("start-loss shugou control candidate count differs")
    return rows, {
        "directory": str(directory),
        "variant": provenance["variant"],
        "git_head": provenance["git_head"],
        "provenance_sha256": source["provenance_sha256"],
        "control_candidate_rows": len(rows),
    }


def prepare_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    review = load_review_config(config["checkpoint_review_config"])
    rows: list[dict[str, Any]] = []
    manifests = []
    sources = {source["id"]: source for source in review["sources"]}
    for source in review["sources"]:
        source_rows, manifest = _load_source(source)
        rows.extend(source_rows)
        manifests.append(manifest)
    _validate_candidates(rows, review)
    if len(rows) != config["integrity_contract"]["baseline_candidate_rows"]:
        raise ValueError("start-loss baseline candidate count differs")

    rankings = []
    selections = []
    selected_ids = set()
    by_id = {row["candidate_id"]: row for row in rows}
    for task in TASKS:
        task_rankings, selected_id = rank_task_candidates(rows, task)
        rankings.extend(task_rankings)
        selected_ids.add(selected_id)
        selected = by_id[selected_id]
        selections.append(
            {
                "task": task,
                "candidate_id": selected_id,
                "source_id": selected["source_id"],
                "checkpoint": selected["checkpoint"],
                "checkpoint_identity": selected["checkpoint"],
                "checkpoint_update": int(selected["checkpoint_update"]),
                "metrics": _metric_values(selected),
                "metric_row": {
                    field: selected.get(field)
                    for field in METRIC_FIELDS
                    if field in selected
                },
                **_verified_checkpoint(selected, sources[selected["source_id"]]),
            }
        )
    if len(selections) != config["integrity_contract"]["baseline_selected_tasks"]:
        raise ValueError("start-loss baseline selection count differs")

    shugou_source = next(
        selection for selection in selections if selection["task"] == "shugou"
    )
    shugou_source_row = {
        "candidate_id": shugou_source["candidate_id"],
        "arm": "baseline",
        "stage": "existing_source",
        "completed_updates": None,
        "checkpoint_path": shugou_source["selected_path"],
        "checkpoint_identity": shugou_source["checkpoint_identity"],
        "expected_variant": shugou_source["checkpoint_variant"],
        "checkpoint_prefix": shugou_source["checkpoint_prefix"],
        **shugou_source["metric_row"],
        "_values": shugou_source["metrics"],
    }
    control_rows, control_manifest = _load_shugou_control_candidates(config)
    shugou_extension_rows = [shugou_source_row, *control_rows]
    if (
        len(shugou_extension_rows)
        != config["integrity_contract"]["shugou_baseline_extension_rows"]
    ):
        raise ValueError("start-loss shugou baseline extension count differs")
    shugou_rankings, shugou_selected_id = rank_task_candidates(
        shugou_extension_rows, "shugou"
    )
    if shugou_selected_id != shugou_source["candidate_id"]:
        selected = next(
            row
            for row in control_rows
            if row["candidate_id"] == shugou_selected_id
        )
        selections = [
            selection for selection in selections if selection["task"] != "shugou"
        ]
        selections.append(
            {
                "task": "shugou",
                "candidate_id": selected["candidate_id"],
                "source_id": "shugou_control_extension",
                "checkpoint": selected["checkpoint"],
                "checkpoint_identity": selected["checkpoint_identity"],
                "checkpoint_update": selected["checkpoint_update"],
                "metrics": selected["_values"],
                "metric_row": {
                    field: selected.get(field)
                    for field in METRIC_FIELDS
                    if field in selected
                },
                "selected_path": selected["checkpoint_path"],
                "selected_relative_path": str(
                    Path(selected["checkpoint_path"]).relative_to(
                        Path(control_manifest["directory"])
                    )
                ).replace("\\", "/"),
                "selected_sha256": _sha256_file(selected["checkpoint_path"]),
                "checkpoint_variant": selected["expected_variant"],
                "checkpoint_prefix": selected["checkpoint_prefix"],
            }
        )
        selections.sort(key=lambda value: TASKS.index(value["task"]))

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "arms").mkdir()
    for arm in ARM_NAMES:
        (output / "arms" / arm).mkdir()
    manifest = {
        "project": PROJECT,
        "variant": VARIANT,
        "source_manifests": manifests,
        "shugou_control_extension": {
            **control_manifest,
            "selection_candidate_rows": len(shugou_extension_rows),
            "selected_candidate": shugou_selected_id,
        },
        "selection_rule": config["selection"],
        "candidate_rows": len(rows),
        "selections": selections,
    }
    _write_json(output / "baseline_selection.json", manifest)
    _write_rankings(
        output / "baseline_checkpoint_rankings.csv", rankings, selected_ids
    )
    _write_rankings(
        output / "baseline_shugou_control_rankings.csv",
        shugou_rankings,
        {shugou_selected_id},
    )
    return {
        "output_directory": str(output),
        "baseline_candidate_rows": len(rows),
        "baseline_selected_tasks": len(selections),
        "parallel_workers": config["training"]["parallel_processes"],
    }


def _position_loss_spec(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == "full_trial":
        return {"type": "full_trial_l1"}
    return {
        "type": "onset_window_phase_normalized_l1",
        **copy.deepcopy(config["onset_window"]),
    }


def _runtime(
    config: dict[str, Any], arm: str, phase: str, task: str
) -> tuple[dict[str, Any], Any]:
    base, geometry = load_canonical_overfit_config(config["base_config"])
    runtime = copy.deepcopy(base)
    arm_config = config["arms"][arm]
    training = config["training"]
    runtime["run_kind"] = config["run_kind"]
    runtime["variant"] = arm_config[f"{phase}_variant"]
    runtime["active_rules"] = [task]
    runtime["optimizer"]["learning_rate"] = config["optimizer"][
        f"{phase}_learning_rate"
    ]
    runtime["training"]["initial_review_updates"] = training[f"{phase}_updates"]
    runtime["training"]["validation_interval"] = training[
        f"{phase}_validation_interval"
    ]
    runtime["training"]["log_interval"] = training["log_interval"]
    runtime["training"].pop("move_sampler")
    runtime["position_loss"] = _position_loss_spec(config, arm)
    runtime["output"] = {
        "directory": str(
            Path(config["output"]["directory"]) / "arms" / arm / task / phase
        )
    }
    return runtime, geometry


def _stage_candidates(
    config: dict[str, Any],
    arm: str,
    task: str,
    phase: str,
    runtime: dict[str, Any],
    geometry,
) -> list[dict[str, Any]]:
    output = Path(config["output"]["directory"])
    stage_output = output / "arms" / arm / task / phase
    arm_config = config["arms"][arm]
    phase_number = int(phase[-1])
    updates = config["training"][f"{phase}_updates"]
    interval = config["training"][f"{phase}_validation_interval"]
    expected_scheduled = list(range(0, updates, interval))
    scheduled = sorted((stage_output / "scheduled_checkpoints").glob("*.pt"))
    if len(scheduled) != len(expected_scheduled):
        raise ValueError(f"scheduled checkpoint count differs: {arm}/{task}/{phase}")
    descriptors = [
        (path, "candidate", expected_update)
        for path, expected_update in zip(scheduled, expected_scheduled)
    ]
    descriptors.append((stage_output / "review_checkpoint.pt", "review", updates - 1))
    conditions = dict(_task_conditions(geometry))[task]
    rows = []
    for checkpoint_path, identity, expected_update in descriptors:
        completed_updates = expected_update + 1
        if phase_number == 2:
            completed_updates += config["training"]["phase1_updates"]
        candidate_id = (
            f"{arm}:{task}:{phase}:{identity}:completed{completed_updates}"
        )
        curve = _checkpoint_curves(
            task,
            conditions,
            candidate_id,
            checkpoint_path,
            runtime,
            geometry,
            checkpoint_identity_name=identity,
            expected_variant=arm_config[f"{phase}_variant"],
            checkpoint_prefix=arm_config[f"{phase}_checkpoint_prefix"],
        )[0]
        if curve["row"]["checkpoint_update"] != expected_update:
            raise ValueError(f"checkpoint update differs: {candidate_id}")
        row = {
            "candidate_id": candidate_id,
            "arm": arm,
            "stage": phase,
            "completed_updates": completed_updates,
            "checkpoint_path": str(checkpoint_path.relative_to(output)).replace(
                "\\", "/"
            ),
            "checkpoint_identity": identity,
            "expected_variant": arm_config[f"{phase}_variant"],
            "checkpoint_prefix": arm_config[f"{phase}_checkpoint_prefix"],
            **curve["row"],
        }
        row["_values"] = _metric_values(row)
        rows.append(row)
    return rows


def train_worker(config_path: str | Path, arm: str, task: str) -> dict[str, Any]:
    config = load_config(config_path)
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown start-loss arm: {arm}")
    if task not in TASKS:
        raise ValueError(f"unknown start-loss task: {task}")
    output = Path(config["output"]["directory"])
    task_output = output / "arms" / arm / task
    task_output.mkdir(exist_ok=True)
    objective = config["arms"][arm]["training_position_objective"]

    phase1_runtime, geometry = _runtime(config, arm, "phase1", task)
    conditions = dict(_task_conditions(geometry))[task]
    phase1_summary = _train_task(
        task,
        conditions,
        phase1_runtime,
        geometry,
        task_output / "phase1",
        target_updates=config["training"]["phase1_updates"],
        checkpoint_variant=config["arms"][arm]["phase1_variant"],
        checkpoint_prefix=config["arms"][arm]["phase1_checkpoint_prefix"],
        training_position_objective=objective,
        save_scheduled_checkpoints=True,
    )

    phase2_runtime, _ = _runtime(config, arm, "phase2", task)
    phase2_summary = _train_task(
        task,
        conditions,
        phase2_runtime,
        geometry,
        task_output / "phase2",
        target_updates=config["training"]["phase2_updates"],
        initial_checkpoint_path=task_output / "phase1" / "review_checkpoint.pt",
        checkpoint_variant=config["arms"][arm]["phase2_variant"],
        checkpoint_prefix=config["arms"][arm]["phase2_checkpoint_prefix"],
        initial_checkpoint_variant=config["arms"][arm]["phase1_variant"],
        initial_checkpoint_prefix=config["arms"][arm]["phase1_checkpoint_prefix"],
        training_position_objective=objective,
        save_scheduled_checkpoints=True,
    )

    rows = []
    for phase, runtime in (
        ("phase1", phase1_runtime),
        ("phase2", phase2_runtime),
    ):
        rows.extend(
            _stage_candidates(config, arm, task, phase, runtime, geometry)
        )
    expected = config["integrity_contract"]["candidate_checkpoints_per_worker"]
    if len(rows) != expected:
        raise ValueError(f"candidate checkpoint count differs: {arm}/{task}")
    serializable_rows = [
        {key: value for key, value in row.items() if key != "_values"}
        for row in rows
    ]
    _write_json(task_output / "candidate_metrics.json", serializable_rows)
    summary = {
        "project": PROJECT,
        "variant": VARIANT,
        "arm": arm,
        "task": task,
        "training_position_objective": objective,
        "phase1": phase1_summary,
        "phase2": phase2_summary,
        "phase1_source": "fresh_seed_42",
        "phase2_source": "phase1/review_checkpoint.pt",
        "phase2_source_sha256": _sha256_file(
            task_output / "phase1" / "review_checkpoint.pt"
        ),
        "candidate_checkpoints": len(rows),
        "completed_updates": config["training"]["total_updates_per_worker"],
        "completed": True,
    }
    _write_json(task_output / "worker_summary.json", summary)
    return summary


def _baseline_rows(output: Path) -> list[dict[str, Any]]:
    manifest = _load_json(output / "baseline_selection.json")
    rows = []
    for selection in manifest["selections"]:
        path = Path(selection["selected_path"])
        if _sha256_file(path) != selection["selected_sha256"]:
            raise RuntimeError(
                f"baseline checkpoint changed: {selection['candidate_id']}"
            )
        row = {
            "candidate_id": selection["candidate_id"],
            "arm": "baseline",
            "stage": "existing",
            "completed_updates": None,
            "checkpoint_path": str(path),
            "checkpoint_identity": selection["checkpoint_identity"],
            "expected_variant": selection["checkpoint_variant"],
            "checkpoint_prefix": selection["checkpoint_prefix"],
            **selection["metric_row"],
        }
        row["_values"] = {
            name: float(value) for name, value in selection["metrics"].items()
        }
        rows.append(row)
    return rows


def _load_new_candidate_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = Path(config["output"]["directory"])
    rows = []
    for arm in ARM_NAMES:
        for task in TASKS:
            path = output / "arms" / arm / task / "candidate_metrics.json"
            values = _load_json_list(path)
            expected = config["integrity_contract"]["candidate_checkpoints_per_worker"]
            if len(values) != expected:
                raise ValueError(f"worker candidate count differs: {arm}/{task}")
            for row in values:
                if row.get("arm") != arm or row.get("task") != task:
                    raise ValueError(f"worker candidate identity differs: {arm}/{task}")
                row["_values"] = _metric_values(row)
                rows.append(row)
    if len(rows) != config["integrity_contract"]["new_candidate_rows"]:
        raise ValueError("new start-loss candidate row count differs")
    return rows


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise ValueError(f"expected a JSON row list: {path}")
    return value


def _curve_for_row(
    row: dict[str, Any],
    output: Path,
    base_runtime: dict[str, Any],
    geometry,
) -> dict[str, Any]:
    path = Path(row["checkpoint_path"])
    if not path.is_absolute():
        path = output / path
    conditions = dict(_task_conditions(geometry))[row["task"]]
    return _checkpoint_curves(
        row["task"],
        conditions,
        row["candidate_id"],
        path,
        base_runtime,
        geometry,
        checkpoint_identity_name=row["checkpoint_identity"],
        expected_variant=row["expected_variant"],
        checkpoint_prefix=row["checkpoint_prefix"],
    )[0]


def _write_comparison_plots(
    output: Path,
    champions: list[dict[str, Any]],
    recommended: dict[str, str],
    base_runtime: dict[str, Any],
    geometry,
) -> list[Path]:
    directory = output / "plots"
    directory.mkdir(exist_ok=False)
    colors = {
        "baseline": "#4c78a8",
        "full_trial": "#f58518",
        "onset_window": "#54a24b",
    }
    paths = []
    for task in TASKS:
        task_rows = [row for row in champions if row["task"] == task]
        curves = [
            (row, _curve_for_row(row, output, base_runtime, geometry))
            for row in task_rows
        ]
        figure, axis = plt.subplots(figsize=(7, 7))
        axis.plot(
            curves[0][1]["target"][:, 0],
            curves[0][1]["target"][:, 1],
            color="0.65",
            linewidth=2,
            label="target",
        )
        for row, curve in curves:
            selected = row["candidate_id"] == recommended[task]
            axis.plot(
                curve["actual"][:, 0],
                curve["actual"][:, 1],
                color=colors[row["arm"]],
                linewidth=3 if selected else 1.5,
                label=f"{row['arm']}{' (selected)' if selected else ''}",
            )
        axis.set_title(f"{task}: start-loss comparison")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = directory / f"{task}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def _overall_group_ranking(
    champion_rankings: list[dict[str, Any]],
    champions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for arm in ("baseline", *ARM_NAMES):
        selected = [
            ranking
            for ranking in champion_rankings
            if champions_by_id[ranking["candidate_id"]]["arm"] == arm
        ]
        if len(selected) != len(TASKS):
            raise ValueError(f"overall group ranking task count differs: {arm}")
        rows.append(
            {
                "arm": arm,
                "max_task_worst_rank_fraction": max(
                    row["worst_rank_fraction"] for row in selected
                ),
                "mean_task_worst_rank_fraction": float(
                    np.mean([row["worst_rank_fraction"] for row in selected])
                ),
                "mean_task_mean_rank_fraction": float(
                    np.mean([row["mean_rank_fraction"] for row in selected])
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["max_task_worst_rank_fraction"],
            row["mean_task_worst_rank_fraction"],
            row["mean_task_mean_rank_fraction"],
            row["arm"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _write_report(
    path: Path,
    champions: list[dict[str, Any]],
    recommended: dict[str, str],
    overall: list[dict[str, Any]],
) -> None:
    lines = [
        "# Canonical start-loss comparison",
        "",
        (
            "The existing phase-normalized baseline was reused without retraining. "
            "Full-trial L1 and onset-window phase-normalized L1 were each trained "
            "from the same seed for 6000 updates at 1e-3 followed by an exact "
            "2000-update continuation at 1e-4."
        ),
        "",
        "## Overall group ranking",
        "",
        "| rank | group | max task worst-rank | mean task worst-rank | mean task mean-rank |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['rank']} | {row['arm']} | "
            f"{row['max_task_worst_rank_fraction']:.6g} | "
            f"{row['mean_task_worst_rank_fraction']:.6g} | "
            f"{row['mean_task_mean_rank_fraction']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Per-task champions",
            "",
            "| task | group | start (mm) | movement mean (mm) | endpoint (mm) | max (mm) | selected |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for task in TASKS:
        for arm in ("baseline", *ARM_NAMES):
            row = next(
                value
                for value in champions
                if value["task"] == task and value["arm"] == arm
            )
            values = row["_values"]
            lines.append(
                f"| {task} | {arm} | "
                f"{values['start_error_euclidean_m'] * 1000:.6g} | "
                f"{values['movement_mean_euclidean_m'] * 1000:.6g} | "
                f"{values['endpoint_euclidean_m'] * 1000:.6g} | "
                f"{values['max_euclidean_m'] * 1000:.6g} | "
                f"{'yes' if row['candidate_id'] == recommended[task] else ''} |"
            )
    lines.extend(
        [
            "",
            "## Compound-stroke detail",
            "",
            "| task | group | transition mean/max (mm) | post-straight mean (mm) | exit error (deg) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for task in ("hengzhe", "shugou"):
        for arm in ("baseline", *ARM_NAMES):
            row = next(
                value
                for value in champions
                if value["task"] == task and value["arm"] == arm
            )
            values = row["_values"]
            lines.append(
                f"| {task} | {arm} | "
                f"{values['transition_mean_euclidean_m'] * 1000:.6g} / "
                f"{values['transition_max_euclidean_m'] * 1000:.6g} | "
                f"{values['post_straight_mean_euclidean_m'] * 1000:.6g} | "
                f"{values['exit_direction_error_deg']:.6g} |"
            )
    lines.extend(
        [
            "",
            f"- Overall ranked group: `{overall[0]['arm']}`.",
            "- No behavioral pass/fail threshold is defined.",
            "- Move, shared-model training, 75k training, and character rollout were not run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def finalize_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(config["output"]["directory"])
    base_runtime, geometry = load_canonical_overfit_config(config["base_config"])
    baseline = _baseline_rows(output)
    candidates = _load_new_candidate_rows(config)
    _write_candidate_metrics(output / "new_candidate_metrics.csv", candidates)

    arm_rankings = []
    arm_champions = []
    arm_selected_ids = set()
    by_id = {row["candidate_id"]: row for row in candidates}
    for arm in ARM_NAMES:
        for task in TASKS:
            arm_rows = [
                row
                for row in candidates
                if row["arm"] == arm and row["task"] == task
            ]
            rankings, selected_id = rank_task_candidates(arm_rows, task)
            arm_rankings.extend(rankings)
            arm_selected_ids.add(selected_id)
            arm_champions.append(by_id[selected_id])
    _write_rankings(
        output / "arm_checkpoint_rankings.csv",
        arm_rankings,
        arm_selected_ids,
    )

    champions = [*baseline, *arm_champions]
    if len(champions) != config["integrity_contract"]["champion_rows"]:
        raise ValueError("start-loss champion row count differs")
    _write_candidate_metrics(output / "champion_metrics.csv", champions)
    champions_by_id = {row["candidate_id"]: row for row in champions}
    champion_rankings = []
    recommended = {}
    for task in TASKS:
        rankings, selected_id = rank_task_candidates(champions, task)
        champion_rankings.extend(rankings)
        recommended[task] = selected_id
    _write_rankings(
        output / "champion_comparison_rankings.csv",
        champion_rankings,
        set(recommended.values()),
    )
    overall = _overall_group_ranking(champion_rankings, champions_by_id)
    with (output / "overall_group_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall[0]))
        writer.writeheader()
        writer.writerows(overall)

    plots = _write_comparison_plots(
        output, champions, recommended, base_runtime, geometry
    )
    _write_report(
        output / "START_LOSS_COMPARISON_REPORT.md",
        champions,
        recommended,
        overall,
    )
    champion_records = {
        f"{row['arm']}:{row['task']}": {
            "candidate_id": row["candidate_id"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": _sha256_file(
                Path(row["checkpoint_path"])
                if Path(row["checkpoint_path"]).is_absolute()
                else output / row["checkpoint_path"]
            ),
            "metrics": row["_values"],
        }
        for row in champions
    }
    provenance = {
        "project": PROJECT,
        "variant": VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "baseline_retrained": False,
        "training_position_objectives": {
            arm: config["arms"][arm]["training_position_objective"]
            for arm in ARM_NAMES
        },
        "learning_rate_schedule": {
            "phase1": {
                "updates": config["training"]["phase1_updates"],
                "learning_rate": config["optimizer"]["phase1_learning_rate"],
            },
            "phase2": {
                "updates": config["training"]["phase2_updates"],
                "learning_rate": config["optimizer"]["phase2_learning_rate"],
            },
        },
        "parallel_workers": config["training"]["parallel_processes"],
        "champions": champion_records,
        "per_task_recommended_candidates": recommended,
        "per_task_recommended_groups": {
            task: champions_by_id[candidate_id]["arm"]
            for task, candidate_id in recommended.items()
        },
        "overall_group_ranking": overall,
        "overall_recommended_group": overall[0]["arm"],
        "selection_rule": config["selection"],
        "integrity": {
            "baseline_selected_tasks": len(baseline),
            "parallel_workers_completed": len(ARM_NAMES) * len(TASKS),
            "new_candidate_rows": len(candidates),
            "champion_rows": len(champions),
            "plots": len(plots),
            "all_numeric_values_finite": True,
        },
        "automatic_checkpoint_selection_performed": True,
        "behavioral_pass_fail_defined": False,
        "training_started": True,
        "move_started": False,
        "shared_8task_started": False,
        "shared_9task_started": False,
        "formal_75k_started": False,
        "complete_character_rollout_started": False,
        "completed": True,
    }
    _write_json(output / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    train = commands.add_parser("train-worker")
    train.add_argument("--arm", choices=ARM_NAMES, required=True)
    train.add_argument("--task", choices=TASKS, required=True)
    commands.add_parser("finalize")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare_experiment(arguments.config)
    elif arguments.command == "train-worker":
        result = train_worker(arguments.config, arguments.arm, arguments.task)
    else:
        result = finalize_experiment(arguments.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
