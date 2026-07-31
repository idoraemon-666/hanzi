from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


PROJECT = "hanzi_stroke_temporal_composition"
VARIANT = "canonical_checkpoint_metric_review_v1"
TASKS = ("heng", "shu", "pie", "na", "dian", "ti", "hengzhe", "shugou")
COMPOUND_TASKS = {"hengzhe", "shugou"}

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
GENERAL_COMPARISON_METRICS = (
    "checkpoint_validation_loss",
    "start_error_euclidean_m",
    "movement_mean_euclidean_m",
    "endpoint_euclidean_m",
    "path_length_ratio_abs_error",
    "max_euclidean_m",
)
COMPOUND_COMPARISON_METRICS = (
    "pre_straight_mean_euclidean_m",
    "transition_mean_euclidean_m",
    "post_straight_mean_euclidean_m",
    "transition_max_euclidean_m",
    "exit_direction_error_deg",
)
OUTPUT_METRIC_FIELDS = (
    "candidate_id",
    "source_id",
    "source_variant",
    "source_git_head",
    *METRIC_FIELDS,
    "path_length_ratio_abs_error",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], description: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{description} keys differ")


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load_json(path)
    _require_exact_keys(
        config,
        {
            "project",
            "run_kind",
            "variant",
            "enabled",
            "tasks",
            "sources",
            "comparison",
            "integrity_contract",
            "exclusions",
            "output",
        },
        "checkpoint review config",
    )
    if (
        config["project"] != PROJECT
        or config["run_kind"] != "canonical_checkpoint_metric_review"
        or config["variant"] != VARIANT
        or config["enabled"] is not True
        or tuple(config["tasks"]) != TASKS
    ):
        raise ValueError("checkpoint review identity differs")
    comparison = config["comparison"]
    _require_exact_keys(
        comparison,
        {
            "all_task_metrics",
            "compound_only_metrics",
            "metric_objective",
            "behavioral_pass_fail",
            "automatic_checkpoint_selection",
            "manual_review_required",
        },
        "checkpoint review comparison",
    )
    if (
        tuple(comparison["all_task_metrics"]) != GENERAL_COMPARISON_METRICS
        or tuple(comparison["compound_only_metrics"])
        != COMPOUND_COMPARISON_METRICS
        or comparison["metric_objective"] != "minimum"
        or comparison["behavioral_pass_fail"] != "not_defined"
        or comparison["automatic_checkpoint_selection"] != "none"
        or comparison["manual_review_required"] is not True
    ):
        raise ValueError("checkpoint review comparison contract differs")
    exclusions = config["exclusions"]
    if set(exclusions) != {
        "training",
        "move",
        "shared_9task",
        "formal_75k",
        "complete_character_rollout",
    } or not all(exclusions.values()):
        raise ValueError("checkpoint review exclusions differ")
    if set(config["output"]) != {"directory"}:
        raise ValueError("checkpoint review output contract differs")
    source_ids: set[str] = set()
    for source in config["sources"]:
        _require_exact_keys(
            source,
            {
                "id",
                "directory",
                "metrics_file",
                "variant",
                "git_head",
                "provenance_sha256",
                "expected_metrics_rows",
                "included_checkpoints",
                "expected_filtered_rows",
            },
            "checkpoint review source",
        )
        if (
            not source["id"]
            or source["id"] in source_ids
            or not source["included_checkpoints"]
        ):
            raise ValueError("checkpoint review source identity differs")
        source_ids.add(source["id"])
    integrity = config["integrity_contract"]
    _require_exact_keys(
        integrity,
        {
            "source_count",
            "candidate_rows",
            "leader_rows",
            "candidates_per_task",
        },
        "checkpoint review integrity",
    )
    if integrity["source_count"] != len(config["sources"]):
        raise ValueError("checkpoint review source count differs")
    if set(integrity["candidates_per_task"]) != set(TASKS):
        raise ValueError("checkpoint review task count contract differs")
    return config


def _verify_sha256sums(directory: Path) -> int:
    sums_path = directory / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(f"missing source SHA256SUMS: {sums_path}")
    count = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid source SHA256SUMS line: {line}")
        relative_text = parts[1].lstrip("*")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source SHA256SUMS path: {relative_text}")
        target = directory / relative
        if not target.is_file() or _sha256_file(target) != parts[0].lower():
            raise ValueError(f"source SHA256 mismatch: {target}")
        count += 1
    if count == 0:
        raise ValueError("source SHA256SUMS is empty")
    return count


def _required_float(row: dict[str, str], field: str) -> float:
    value = row[field]
    if value == "":
        raise ValueError(f"required metric is empty: {field}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"metric is not finite: {field}")
    return parsed


def _load_source(
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(source["directory"])
    provenance_path = directory / "provenance.json"
    metrics_path = directory / source["metrics_file"]
    if not directory.is_dir() or not provenance_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"checkpoint review source is incomplete: {directory}")
    provenance_sha256 = _sha256_file(provenance_path)
    if provenance_sha256 != source["provenance_sha256"]:
        raise ValueError(f"source provenance SHA256 differs: {source['id']}")
    verified_sha256_entries = _verify_sha256sums(directory)
    provenance = _load_json(provenance_path)
    if (
        provenance.get("project") != PROJECT
        or provenance.get("variant") != source["variant"]
        or provenance.get("git_head") != source["git_head"]
        or provenance.get("completed") is not True
        or provenance.get("behavioral_pass_fail_defined") is not False
        or provenance.get("integrity", {}).get("metrics_rows")
        != source["expected_metrics_rows"]
    ):
        raise ValueError(f"source provenance contract differs: {source['id']}")
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ValueError(f"source metric columns differ: {source['id']}")
        all_rows = list(reader)
    if len(all_rows) != source["expected_metrics_rows"]:
        raise ValueError(f"source metric row count differs: {source['id']}")
    included = set(source["included_checkpoints"])
    rows: list[dict[str, Any]] = []
    for raw in all_rows:
        if (
            raw["category"] != "stroke"
            or raw["task"] not in TASKS
            or raw["checkpoint"] not in included
        ):
            continue
        if raw["rule"] != raw["task"]:
            raise ValueError(f"source task/rule mismatch: {source['id']}")
        update = int(raw["checkpoint_update"])
        if update < 0:
            raise ValueError(f"negative checkpoint update: {source['id']}")
        values = {
            metric: _required_float(raw, metric)
            for metric in GENERAL_COMPARISON_METRICS
            if metric != "path_length_ratio_abs_error"
        }
        ratio = _required_float(raw, "path_length_ratio")
        values["path_length_ratio_abs_error"] = abs(ratio - 1.0)
        if raw["task"] in COMPOUND_TASKS:
            values.update(
                {
                    metric: _required_float(raw, metric)
                    for metric in COMPOUND_COMPARISON_METRICS
                }
            )
        candidate_id = (
            f"{source['id']}:{raw['task']}:{raw['checkpoint']}:"
            f"u{raw['checkpoint_update']}"
        )
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "source_id": source["id"],
            "source_variant": source["variant"],
            "source_git_head": source["git_head"],
            **raw,
            "path_length_ratio_abs_error": values[
                "path_length_ratio_abs_error"
            ],
            "_values": values,
        }
        rows.append(row)
    if len(rows) != source["expected_filtered_rows"]:
        raise ValueError(f"source filtered metric row count differs: {source['id']}")
    manifest = {
        "id": source["id"],
        "directory": str(directory.resolve()),
        "metrics_file": source["metrics_file"],
        "variant": source["variant"],
        "git_head": source["git_head"],
        "provenance_sha256": provenance_sha256,
        "verified_sha256_entries": verified_sha256_entries,
        "included_checkpoints": list(source["included_checkpoints"]),
        "candidate_rows": len(rows),
    }
    return rows, manifest


def _validate_candidates(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("checkpoint review candidate IDs are not unique")
    if len(rows) != config["integrity_contract"]["candidate_rows"]:
        raise ValueError("checkpoint review candidate row count differs")
    by_task = {task: [] for task in TASKS}
    for row in rows:
        by_task[row["task"]].append(row)
    for task, task_rows in by_task.items():
        expected = config["integrity_contract"]["candidates_per_task"][task]
        if len(task_rows) != expected:
            raise ValueError(f"checkpoint review candidate count differs: {task}")
        identities = {
            (
                row["condition_id"],
                row["source_character"],
                row["source_component_index"],
                row["movement_intervals"],
                row["movement_samples"],
                row["target_path_length_m"],
            )
            for row in task_rows
        }
        if len(identities) != 1:
            raise ValueError(f"checkpoint review target identity differs: {task}")


def _metric_leaders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaders: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        metrics = list(GENERAL_COMPARISON_METRICS)
        if task in COMPOUND_TASKS:
            metrics.extend(COMPOUND_COMPARISON_METRICS)
        for metric in metrics:
            minimum = min(row["_values"][metric] for row in task_rows)
            candidate_ids = sorted(
                row["candidate_id"]
                for row in task_rows
                if row["_values"][metric] == minimum
            )
            leaders.append(
                {
                    "task": task,
                    "metric": metric,
                    "objective": "minimum",
                    "leader_value": minimum,
                    "leader_candidate_ids": ";".join(candidate_ids),
                    "exact_tie_count": len(candidate_ids),
                }
            )
    return leaders


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _leader_id(
    leaders: list[dict[str, Any]], task: str, metric: str
) -> str:
    return next(
        row["leader_candidate_ids"]
        for row in leaders
        if row["task"] == task and row["metric"] == metric
    )


def _write_report(
    path: Path,
    manifests: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
) -> None:
    lines = [
        "# Canonical checkpoint metric review",
        "",
        "This is a threshold-free descriptive comparison. It defines no behavioral "
        "pass/fail threshold, performs no automatic checkpoint selection, and does "
        "not authorize any later training stage.",
        "",
        "## Verified formal sources",
        "",
        "| Source | Variant | Candidate rows | Verified SHA256 entries |",
        "|---|---|---:|---:|",
    ]
    for source in manifests:
        lines.append(
            f"| {source['id']} | {source['variant']} | "
            f"{source['candidate_rows']} | {source['verified_sha256_entries']} |"
        )
    lines.extend(
        [
            "",
            "## Per-metric leaders",
            "",
            "A row may name different candidates across metrics. That difference is "
            "the review result; it is not resolved by implicit weighting.",
            "",
            "| Task | Validation loss | Movement mean | Endpoint | "
            "Path-ratio absolute error | Transition mean | Exit direction |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for task in TASKS:
        transition = (
            _leader_id(leaders, task, "transition_mean_euclidean_m")
            if task in COMPOUND_TASKS
            else ""
        )
        exit_direction = (
            _leader_id(leaders, task, "exit_direction_error_deg")
            if task in COMPOUND_TASKS
            else ""
        )
        lines.append(
            f"| {task} | "
            f"{_leader_id(leaders, task, 'checkpoint_validation_loss')} | "
            f"{_leader_id(leaders, task, 'movement_mean_euclidean_m')} | "
            f"{_leader_id(leaders, task, 'endpoint_euclidean_m')} | "
            f"{_leader_id(leaders, task, 'path_length_ratio_abs_error')} | "
            f"{transition} | {exit_direction} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "- `automatic_checkpoint_selection_performed = false`",
            "- `behavioral_pass_fail_defined = false`",
            "- `manual_review_required = true`",
            "- No move, shared model, 75k training, or complete-character rollout was run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_review(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    output = Path(config["output"]["directory"])
    if output.exists():
        raise FileExistsError(f"checkpoint review output already exists: {output}")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for source in config["sources"]:
        source_rows, manifest = _load_source(source)
        rows.extend(source_rows)
        manifests.append(manifest)
    _validate_candidates(rows, config)
    leaders = _metric_leaders(rows)
    if len(leaders) != config["integrity_contract"]["leader_rows"]:
        raise ValueError("checkpoint review leader row count differs")
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output / "checkpoint_candidate_metrics.csv",
        OUTPUT_METRIC_FIELDS,
        rows,
    )
    _write_csv(
        output / "checkpoint_metric_leaders.csv",
        (
            "task",
            "metric",
            "objective",
            "leader_value",
            "leader_candidate_ids",
            "exact_tie_count",
        ),
        leaders,
    )
    _write_json(output / "source_manifest.json", manifests)
    _write_report(output / "CHECKPOINT_METRIC_REVIEW_REPORT.md", manifests, leaders)
    provenance = {
        "project": PROJECT,
        "variant": VARIANT,
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "submodule_head": _git_value("-C", "mRNNTorch", "rev-parse", "HEAD"),
        "config_sha256": _sha256_file(config_path),
        "sources": manifests,
        "comparison_metrics": {
            "all_tasks": list(GENERAL_COMPARISON_METRICS),
            "compound_only": list(COMPOUND_COMPARISON_METRICS),
            "objective": "minimum",
        },
        "integrity": {
            "source_count": len(manifests),
            "candidate_rows": len(rows),
            "leader_rows": len(leaders),
            "task_count": len(TASKS),
            "all_numeric_values_finite": True,
        },
        "behavioral_pass_fail_defined": False,
        "automatic_checkpoint_selection_performed": False,
        "manual_review_required": True,
        "training_started": False,
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
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_review(args.config)


if __name__ == "__main__":
    main()
