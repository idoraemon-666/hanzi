"""Scale-free checkpoint selection and a matched shugou loss intervention."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
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
from hanzi_writing.geometry import PROJECT
from hanzi_writing.training import _sha256_file


VARIANT = "canonical_shugou_checkpoint_experiment_v1"
TASK = "shugou"
ARM_NAMES = ("control", "subphase_equal")
CONFIG_PATH = (
    "configurations/"
    "hanzi_stroke_temporal_composition_canonical_shugou_checkpoint_experiment_v1.json"
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
            "task",
            "base_config",
            "checkpoint_review_config",
            "selection",
            "optimizer",
            "training",
            "arms",
            "intervention",
            "integrity_contract",
            "exclusions",
            "output",
        },
        "shugou checkpoint experiment config",
    )
    if (
        config["project"] != PROJECT
        or config["run_kind"] != "canonical_shugou_checkpoint_experiment"
        or config["variant"] != VARIANT
        or config["enabled"] is not True
        or config["seed"] != 42
        or config["validation_seed"] != 1042
        or config["task"] != TASK
        or config["base_config"]
        != (
            "configurations/"
            "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
        )
        or config["checkpoint_review_config"]
        != (
            "configurations/"
            "hanzi_stroke_temporal_composition_canonical_checkpoint_metric_review_v1.json"
        )
    ):
        raise ValueError("shugou checkpoint experiment identity differs")
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
        raise ValueError("shugou checkpoint selection rule differs")
    if config["optimizer"] != {
        "learning_rate": 0.0001,
        "source_optimizer_state": "preserve_then_set_learning_rate",
    }:
        raise ValueError("shugou checkpoint experiment optimizer differs")
    if config["training"] != {
        "additional_updates_per_arm": 2000,
        "batch_size": 1,
        "validation_interval": 100,
        "log_interval": 100,
        "parallel_processes": 2,
        "candidate_checkpoint_interval": 100,
    }:
        raise ValueError("shugou checkpoint experiment schedule differs")
    if config["arms"] != {
        "control": {
            "variant": "canonical_shugou_checkpoint_control_v1",
            "checkpoint_prefix": "canonical_shugou_checkpoint_control",
            "training_position_objective": "phase_normalized_l1",
        },
        "subphase_equal": {
            "variant": "canonical_shugou_subphase_equal_v1",
            "checkpoint_prefix": "canonical_shugou_subphase_equal",
            "training_position_objective": "compound_subphase_equal_l1",
        },
    }:
        raise ValueError("shugou checkpoint experiment arms differ")
    if config["intervention"] != {
        "outer_phase_weights": {
            "stable": 0.1,
            "delay": 0.1,
            "movement": 0.6,
            "hold": 0.2,
        },
        "movement_subphases": [
            "pre_straight",
            "transition",
            "post_straight",
        ],
        "movement_subphase_weights": [1.0 / 3.0] * 3,
        "changed_factor": "movement_internal_aggregation_only",
    }:
        raise ValueError("shugou checkpoint intervention differs")
    if config["integrity_contract"] != {
        "initial_candidate_rows": 56,
        "initial_selected_tasks": 8,
        "scheduled_checkpoints_per_arm": 20,
        "final_shugou_candidate_rows": 43,
    }:
        raise ValueError("shugou checkpoint experiment integrity contract differs")
    if config["exclusions"] != {
        "move": True,
        "shared_9task": True,
        "formal_75k": True,
        "complete_character_rollout": True,
    }:
        raise ValueError("shugou checkpoint experiment exclusions differ")
    if config["output"] != {
        "directory": (
            "runs/hanzi_stroke_temporal_composition/"
            "canonical_shugou_checkpoint_experiment/dev42"
        )
    }:
        raise ValueError("shugou checkpoint experiment output differs")
    load_review_config(config["checkpoint_review_config"])
    load_canonical_overfit_config(config["base_config"])
    return config


def _metrics_for_task(task: str) -> tuple[str, ...]:
    metrics = list(GENERAL_COMPARISON_METRICS)
    if task in {"hengzhe", "shugou"}:
        metrics.extend(COMPOUND_COMPARISON_METRICS)
    return tuple(metrics)


def _dominates(
    left: dict[str, Any], right: dict[str, Any], metrics: tuple[str, ...]
) -> bool:
    return all(left["_values"][name] <= right["_values"][name] for name in metrics) and any(
        left["_values"][name] < right["_values"][name] for name in metrics
    )


def rank_task_candidates(
    rows: list[dict[str, Any]], task: str
) -> tuple[list[dict[str, Any]], str]:
    task_rows = [row for row in rows if row["task"] == task]
    if not task_rows:
        raise ValueError(f"checkpoint selection has no candidates: {task}")
    candidate_ids = [str(row["candidate_id"]) for row in task_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"checkpoint selection candidate IDs differ: {task}")
    metrics = _metrics_for_task(task)
    denominator = max(len(task_rows) - 1, 1)
    rank_fractions: dict[str, dict[str, float]] = {
        candidate_id: {} for candidate_id in candidate_ids
    }
    for metric in metrics:
        ordered = sorted(row["_values"][metric] for row in task_rows)
        first_rank = {
            value: ordered.index(value) + 1 for value in set(ordered)
        }
        for row in task_rows:
            rank_fractions[row["candidate_id"]][metric] = (
                first_rank[row["_values"][metric]] - 1
            ) / denominator
    frontier_ids = {
        row["candidate_id"]
        for row in task_rows
        if not any(
            other["candidate_id"] != row["candidate_id"]
            and _dominates(other, row, metrics)
            for other in task_rows
        )
    }
    rankings = []
    for row in task_rows:
        fractions = rank_fractions[row["candidate_id"]]
        rankings.append(
            {
                "task": task,
                "candidate_id": row["candidate_id"],
                "is_pareto": row["candidate_id"] in frontier_ids,
                "worst_rank_fraction": max(fractions.values()),
                "mean_rank_fraction": float(np.mean(tuple(fractions.values()))),
                "validation_loss": row["_values"]["checkpoint_validation_loss"],
                "metric_rank_fractions": fractions,
            }
        )
    selected = min(
        (row for row in rankings if row["is_pareto"]),
        key=lambda row: (
            row["worst_rank_fraction"],
            row["mean_rank_fraction"],
            row["validation_loss"],
            row["candidate_id"],
        ),
    )
    return sorted(rankings, key=lambda row: row["candidate_id"]), selected[
        "candidate_id"
    ]


def _write_rankings(
    path: Path,
    rankings: list[dict[str, Any]],
    selected_ids: set[str],
) -> None:
    metric_columns = tuple(
        f"{metric}_rank_fraction"
        for metric in (*GENERAL_COMPARISON_METRICS, *COMPOUND_COMPARISON_METRICS)
    )
    fields = (
        "task",
        "candidate_id",
        "is_pareto",
        "worst_rank_fraction",
        "mean_rank_fraction",
        "validation_loss",
        "selected",
        *metric_columns,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ranking in rankings:
            fractions = ranking["metric_rank_fractions"]
            writer.writerow(
                {
                    **{key: ranking[key] for key in fields if key in ranking},
                    "selected": ranking["candidate_id"] in selected_ids,
                    **{
                        f"{metric}_rank_fraction": fractions.get(metric, "")
                        for metric in (
                            *GENERAL_COMPARISON_METRICS,
                            *COMPOUND_COMPARISON_METRICS,
                        )
                    },
                }
            )


def _verified_checkpoint(
    row: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    directory = Path(source["directory"]).resolve()
    relative = (
        Path("models")
        / row["task"]
        / f"{row['checkpoint']}_checkpoint.pt"
    )
    path = directory / relative
    provenance = _load_json(directory / "provenance.json")
    digest = _sha256_file(path)
    relative_text = str(relative).replace("\\", "/")
    if provenance.get("checkpoint_sha256", {}).get(relative_text) != digest:
        raise ValueError(f"selected checkpoint SHA256 differs: {row['candidate_id']}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    identity = str(row["checkpoint"])
    kind = checkpoint.get("checkpoint_kind")
    if (
        checkpoint.get("project") != PROJECT
        or checkpoint.get("task") != row["task"]
        or not isinstance(kind, str)
        or not kind.endswith(f"_{identity}")
        or int(checkpoint.get("update", -1)) != int(row["checkpoint_update"])
        or not math.isclose(
            float(checkpoint.get("validation_loss", math.nan)),
            float(row["_values"]["checkpoint_validation_loss"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"selected checkpoint identity differs: {row['candidate_id']}")
    return {
        "selected_path": str(path),
        "selected_relative_path": relative_text,
        "selected_sha256": digest,
        "checkpoint_variant": checkpoint["variant"],
        "checkpoint_prefix": kind[: -len(f"_{identity}")],
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
    if len(rows) != config["integrity_contract"]["initial_candidate_rows"]:
        raise ValueError("initial checkpoint candidate count differs")
    rankings: list[dict[str, Any]] = []
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
                "checkpoint_update": int(selected["checkpoint_update"]),
                "metrics": {
                    metric: selected["_values"][metric]
                    for metric in _metrics_for_task(task)
                },
                **_verified_checkpoint(selected, sources[selected["source_id"]]),
            }
        )
    if len(selections) != config["integrity_contract"]["initial_selected_tasks"]:
        raise ValueError("initial checkpoint selection task count differs")
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "arms").mkdir()
    manifest = {
        "project": PROJECT,
        "variant": VARIANT,
        "source_manifests": manifests,
        "selection_rule": config["selection"],
        "candidate_rows": len(rows),
        "selections": selections,
    }
    _write_json(output / "initial_checkpoint_selection.json", manifest)
    _write_rankings(
        output / "initial_checkpoint_rankings.csv", rankings, selected_ids
    )
    return {
        "output_directory": str(output),
        "candidate_rows": len(rows),
        "selected_tasks": len(selections),
        "shugou_source_candidate": next(
            value["candidate_id"] for value in selections if value["task"] == TASK
        ),
        "parallel_processes": config["training"]["parallel_processes"],
    }


def _runtime(config: dict[str, Any], arm_name: str) -> tuple[dict[str, Any], Any]:
    base, geometry = load_canonical_overfit_config(config["base_config"])
    arm = config["arms"][arm_name]
    runtime = copy.deepcopy(base)
    runtime["run_kind"] = config["run_kind"]
    runtime["variant"] = arm["variant"]
    runtime["active_rules"] = [TASK]
    runtime["optimizer"]["learning_rate"] = config["optimizer"]["learning_rate"]
    runtime["training"]["initial_review_updates"] = config["training"][
        "additional_updates_per_arm"
    ]
    runtime["training"]["validation_interval"] = config["training"][
        "validation_interval"
    ]
    runtime["training"]["log_interval"] = config["training"]["log_interval"]
    runtime["training"].pop("move_sampler")
    runtime["output"] = {
        "directory": str(Path(config["output"]["directory"]) / "arms" / arm_name)
    }
    return runtime, geometry


def _selected_source(output: Path) -> dict[str, Any]:
    manifest = _load_json(output / "initial_checkpoint_selection.json")
    matches = [value for value in manifest["selections"] if value["task"] == TASK]
    if len(matches) != 1:
        raise ValueError("shugou source checkpoint selection differs")
    selected = matches[0]
    if _sha256_file(selected["selected_path"]) != selected["selected_sha256"]:
        raise RuntimeError("selected shugou source checkpoint changed")
    return selected


def train_arm(config_path: str | Path, arm_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    if arm_name not in ARM_NAMES:
        raise ValueError(f"unknown shugou experiment arm: {arm_name}")
    output = Path(config["output"]["directory"])
    source = _selected_source(output)
    runtime, geometry = _runtime(config, arm_name)
    arm = config["arms"][arm_name]
    return _train_task(
        TASK,
        dict(_task_conditions(geometry))[TASK],
        runtime,
        geometry,
        output / "arms" / arm_name,
        target_updates=config["training"]["additional_updates_per_arm"],
        initial_checkpoint_path=Path(source["selected_path"]),
        checkpoint_variant=arm["variant"],
        checkpoint_prefix=arm["checkpoint_prefix"],
        initial_checkpoint_variant=source["checkpoint_variant"],
        initial_checkpoint_prefix=source["checkpoint_prefix"],
        training_position_objective=arm["training_position_objective"],
        save_scheduled_checkpoints=True,
    )


def _values(row: dict[str, Any]) -> dict[str, float]:
    values = {
        metric: float(row[metric])
        for metric in GENERAL_COMPARISON_METRICS
        if metric != "path_length_ratio_abs_error"
    }
    values["path_length_ratio_abs_error"] = abs(
        float(row["path_length_ratio"]) - 1.0
    )
    values.update(
        {metric: float(row[metric]) for metric in COMPOUND_COMPARISON_METRICS}
    )
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("shugou candidate metrics are not finite")
    return values


def _curve_record(
    curve: dict[str, Any],
    candidate_id: str,
    arm: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    row = {
        "candidate_id": candidate_id,
        "arm": arm,
        "checkpoint_path": str(checkpoint_path),
        **curve["row"],
    }
    row["_values"] = _values(row)
    curve.update(
        {
            "candidate_id": candidate_id,
            "arm": arm,
            "checkpoint_path": str(checkpoint_path),
        }
    )
    return row


def _experiment_candidates(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = Path(config["output"]["directory"])
    source = _selected_source(output)
    runtime, geometry = _runtime(config, "control")
    conditions = dict(_task_conditions(geometry))[TASK]
    rows = []
    curves = []
    source_path = Path(source["selected_path"])
    source_curve = _checkpoint_curves(
        TASK,
        conditions,
        "source",
        source_path,
        runtime,
        geometry,
        checkpoint_identity_name=source["checkpoint"],
        expected_variant=source["checkpoint_variant"],
        checkpoint_prefix=source["checkpoint_prefix"],
    )[0]
    rows.append(
        _curve_record(
            source_curve, source["candidate_id"], "source", source_path
        )
    )
    curves.append(source_curve)
    for arm_name in ARM_NAMES:
        arm = config["arms"][arm_name]
        arm_output = output / "arms" / arm_name
        scheduled = sorted((arm_output / "scheduled_checkpoints").glob("*.pt"))
        if (
            len(scheduled)
            != config["integrity_contract"]["scheduled_checkpoints_per_arm"]
        ):
            raise ValueError(f"scheduled checkpoint count differs: {arm_name}")
        expected_updates = list(
            range(
                0,
                config["training"]["additional_updates_per_arm"],
                config["training"]["candidate_checkpoint_interval"],
            )
        )
        scheduled_updates = []
        for path in scheduled:
            curve = _checkpoint_curves(
                TASK,
                conditions,
                f"{arm_name}_{path.stem}",
                path,
                runtime,
                geometry,
                checkpoint_identity_name="candidate",
                expected_variant=arm["variant"],
                checkpoint_prefix=arm["checkpoint_prefix"],
            )[0]
            candidate_id = f"{arm_name}:scheduled:u{curve['row']['checkpoint_update']}"
            scheduled_updates.append(curve["row"]["checkpoint_update"])
            rows.append(_curve_record(curve, candidate_id, arm_name, path))
            curves.append(curve)
        if scheduled_updates != expected_updates:
            raise ValueError(f"scheduled checkpoint updates differ: {arm_name}")
        review_path = arm_output / "review_checkpoint.pt"
        curve = _checkpoint_curves(
            TASK,
            conditions,
            f"{arm_name}_review",
            review_path,
            runtime,
            geometry,
            checkpoint_identity_name="review",
            expected_variant=arm["variant"],
            checkpoint_prefix=arm["checkpoint_prefix"],
        )[0]
        if (
            curve["row"]["checkpoint_update"]
            != config["training"]["additional_updates_per_arm"] - 1
        ):
            raise ValueError(f"review checkpoint update differs: {arm_name}")
        candidate_id = f"{arm_name}:review:u{curve['row']['checkpoint_update']}"
        rows.append(_curve_record(curve, candidate_id, arm_name, review_path))
        curves.append(curve)
    if len(rows) != config["integrity_contract"]["final_shugou_candidate_rows"]:
        raise ValueError("final shugou candidate count differs")
    return rows, curves


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["worst_rank_fraction"],
        row["mean_rank_fraction"],
        row["validation_loss"],
        row["candidate_id"],
    )


def _write_candidate_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "candidate_id",
        "arm",
        "checkpoint_path",
        *METRIC_FIELDS,
        "path_length_ratio_abs_error",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "path_length_ratio_abs_error": row["_values"][
                        "path_length_ratio_abs_error"
                    ],
                }
            )


def _write_plot(
    path: Path,
    curves_by_id: dict[str, dict[str, Any]],
    selected_ids: list[tuple[str, str]],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    first = next(iter(curves_by_id.values()))
    axis.plot(
        first["target"][:, 0],
        first["target"][:, 1],
        color="0.65",
        linewidth=2,
        label="target",
    )
    colors = {
        "source": "#4c78a8",
        "control": "#f58518",
        "subphase_equal": "#54a24b",
        "recommended": "#e45756",
    }
    plotted = set()
    for label, candidate_id in selected_ids:
        if candidate_id in plotted:
            continue
        plotted.add(candidate_id)
        curve = curves_by_id[candidate_id]
        axis.plot(
            curve["actual"][:, 0],
            curve["actual"][:, 1],
            color=colors[label],
            linewidth=2 if label == "recommended" else 1.5,
            label=f"{label}: {candidate_id}",
        )
    axis.set_title("shugou checkpoint experiment")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def finalize_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(config["output"]["directory"])
    source_manifest = _load_json(output / "initial_checkpoint_selection.json")
    source = _selected_source(output)
    rows, curves = _experiment_candidates(config)
    rankings, selected_id = rank_task_candidates(rows, TASK)
    ranking_by_id = {row["candidate_id"]: row for row in rankings}
    row_by_id = {row["candidate_id"]: row for row in rows}
    curve_by_id = {curve["candidate_id"]: curve for curve in curves}
    champions = {
        arm: min(
            (
                ranking
                for ranking in rankings
                if row_by_id[ranking["candidate_id"]]["arm"] == arm
            ),
            key=_ranking_key,
        )["candidate_id"]
        for arm in ARM_NAMES
    }
    metrics = _metrics_for_task(TASK)
    control = row_by_id[champions["control"]]
    intervention = row_by_id[champions["subphase_equal"]]
    selected_row = row_by_id[selected_id]
    intervention_dominates_control = _dominates(intervention, control, metrics)
    control_dominates_intervention = _dominates(control, intervention, metrics)
    if (
        selected_row["arm"] == "subphase_equal"
        and intervention_dominates_control
    ):
        outcome = "subphase_equal_supported_by_pareto_dominance"
    elif selected_row["arm"] == "subphase_equal":
        outcome = "subphase_equal_selected_with_metric_tradeoffs"
    elif selected_row["arm"] == "control":
        outcome = "standard_control_selected"
    else:
        outcome = "source_checkpoint_retained"
    selected_checkpoint = output / "selected_checkpoint.pt"
    shutil.copy2(selected_row["checkpoint_path"], selected_checkpoint)
    _write_candidate_metrics(output / "shugou_candidate_metrics.csv", rows)
    _write_rankings(
        output / "shugou_checkpoint_rankings.csv", rankings, {selected_id}
    )
    plot_path = output / "shugou_checkpoint_comparison.png"
    _write_plot(
        plot_path,
        curve_by_id,
        [
            ("source", source["candidate_id"]),
            ("control", champions["control"]),
            ("subphase_equal", champions["subphase_equal"]),
            ("recommended", selected_id),
        ],
    )
    np.savez_compressed(
        output / "shugou_selected_trajectories.npz",
        candidate_id=np.asarray(
            [
                source["candidate_id"],
                champions["control"],
                champions["subphase_equal"],
                selected_id,
            ],
            dtype="U96",
        ),
        target_xy_m=np.stack(
            [
                curve_by_id[candidate_id]["target"]
                for candidate_id in (
                    source["candidate_id"],
                    champions["control"],
                    champions["subphase_equal"],
                    selected_id,
                )
            ]
        ),
        actual_xy_m=np.stack(
            [
                curve_by_id[candidate_id]["actual"]
                for candidate_id in (
                    source["candidate_id"],
                    champions["control"],
                    champions["subphase_equal"],
                    selected_id,
                )
            ]
        ),
    )
    report_lines = [
        "# Shugou checkpoint-selection experiment",
        "",
        (
            "Checkpoint choice is scale-free and threshold-free: candidates are "
            "restricted to the Pareto frontier, then ranked by minimum worst "
            "competition-rank fraction, mean rank fraction, validation loss, "
            "and candidate ID."
        ),
        "",
        (
            "Both training arms start from the same selected checkpoint and preserve "
            "its model, Adam state, and global RNG state. The only intervention is "
            "equal averaging of pre-straight, transition, and post-straight errors "
            "inside the unchanged 0.6 movement loss."
        ),
        "",
        "| role | candidate | validation loss | mean (mm) | endpoint (mm) | "
        "transition mean/max (mm) | exit error (deg) | worst rank | mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report_ids = (
        ("source", source["candidate_id"]),
        ("control champion", champions["control"]),
        ("subphase-equal champion", champions["subphase_equal"]),
        ("recommended", selected_id),
    )
    for role, candidate_id in report_ids:
        row = row_by_id[candidate_id]
        ranking = ranking_by_id[candidate_id]
        report_lines.append(
            f"| {role} | {candidate_id} | "
            f"{row['_values']['checkpoint_validation_loss']:.9g} | "
            f"{row['_values']['movement_mean_euclidean_m'] * 1000:.6g} | "
            f"{row['_values']['endpoint_euclidean_m'] * 1000:.6g} | "
            f"{row['_values']['transition_mean_euclidean_m'] * 1000:.6g} / "
            f"{row['_values']['transition_max_euclidean_m'] * 1000:.6g} | "
            f"{row['_values']['exit_direction_error_deg']:.6g} | "
            f"{ranking['worst_rank_fraction']:.6g} | "
            f"{ranking['mean_rank_fraction']:.6g} |"
        )
    report_lines.extend(
        [
            "",
            f"- Experimental outcome: `{outcome}`.",
            (
                "- Subphase-equal champion Pareto-dominates control champion: "
                f"`{str(intervention_dominates_control).lower()}`."
            ),
            (
                "- Control champion Pareto-dominates subphase-equal champion: "
                f"`{str(control_dominates_intervention).lower()}`."
            ),
            "- No behavioral pass/fail threshold is defined.",
            "- Move, shared-model training, 75k training, and character rollout were not run.",
            "",
        ]
    )
    (output / "SHUGOU_CHECKPOINT_EXPERIMENT_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8", newline="\n"
    )
    checkpoint_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in sorted((output / "arms").glob("*/*_checkpoint.pt"))
    }
    checkpoint_hashes.update(
        {
            str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
            for path in sorted(
                (output / "arms").glob("*/scheduled_checkpoints/*.pt")
            )
        }
    )
    provenance = {
        "project": PROJECT,
        "variant": VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "initial_selection_sha256": _sha256_file(
            output / "initial_checkpoint_selection.json"
        ),
        "source_candidate": source["candidate_id"],
        "source_checkpoint_sha256": source["selected_sha256"],
        "arm_champions": champions,
        "recommended_candidate": selected_id,
        "recommended_arm": selected_row["arm"],
        "recommended_checkpoint_sha256": _sha256_file(selected_checkpoint),
        "selection_rule": config["selection"],
        "experimental_outcome": outcome,
        "intervention_champion_pareto_dominates_control_champion": (
            intervention_dominates_control
        ),
        "control_champion_pareto_dominates_intervention_champion": (
            control_dominates_intervention
        ),
        "checkpoint_sha256": checkpoint_hashes,
        "integrity": {
            "initial_candidate_rows": source_manifest["candidate_rows"],
            "initial_selected_tasks": len(source_manifest["selections"]),
            "final_shugou_candidate_rows": len(rows),
            "scheduled_checkpoints_per_arm": config["integrity_contract"][
                "scheduled_checkpoints_per_arm"
            ],
            "all_numeric_values_finite": True,
            "selected_checkpoint_count": 1,
            "plots": 1,
        },
        "automatic_checkpoint_selection_performed": True,
        "behavioral_pass_fail_defined": False,
        "training_started": True,
        "move_started": False,
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
    train = commands.add_parser("train-arm")
    train.add_argument("--arm", choices=ARM_NAMES, required=True)
    commands.add_parser("finalize")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare_experiment(arguments.config)
    elif arguments.command == "train-arm":
        result = train_arm(arguments.config, arguments.arm)
    else:
        result = finalize_experiment(arguments.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
