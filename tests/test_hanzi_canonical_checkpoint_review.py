from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hanzi_writing.canonical_checkpoint_review import (
    COMPOUND_COMPARISON_METRICS,
    GENERAL_COMPARISON_METRICS,
    METRIC_FIELDS,
    TASKS,
    run_review,
)


SOURCE_DEFINITIONS = (
    (
        "stage1_lr1e3_6000",
        "canonical_single_duration_overfit_v1",
        "1" * 40,
        "single_task_overfit_metrics.csv",
        ("best", "review"),
        40,
        16,
    ),
    (
        "refinement_lr1e4_2000",
        "canonical_stroke_refinement_v1",
        "2" * 40,
        "stroke_refinement_metrics.csv",
        ("best", "review"),
        24,
        16,
    ),
    (
        "scratch_lr1e4_10000",
        "canonical_eight_stroke_scratch_lr1e4_v1",
        "3" * 40,
        "scratch_low_lr_metrics.csv",
        ("best", "review"),
        16,
        16,
    ),
    (
        "followup_lr1e4_total4000",
        "canonical_selected_refinement_followup_v1",
        "4" * 40,
        "selected_followup_metrics.csv",
        ("best", "review"),
        12,
        8,
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _stroke_row(
    task: str,
    checkpoint: str,
    update: int,
    value_offset: float,
) -> dict[str, object]:
    compound = task in {"hengzhe", "shugou"}
    row: dict[str, object] = {
        "task": task,
        "category": "stroke",
        "rule": task,
        "condition_id": f"canonical_{task}",
        "source_character": "mu",
        "source_component_index": TASKS.index(task),
        "checkpoint": checkpoint,
        "checkpoint_update": update,
        "checkpoint_validation_loss": 0.01 + value_offset,
        "movement_intervals": 200 if compound else 150,
        "movement_samples": 201 if compound else 151,
        "start_error_euclidean_m": 0.02 + value_offset,
        "movement_mean_euclidean_m": 0.03 + value_offset,
        "endpoint_euclidean_m": 0.04 + value_offset,
        "target_path_length_m": 0.2,
        "actual_path_length_m": 0.18 + value_offset,
        "path_length_ratio": 0.9 + value_offset,
        "max_euclidean_m": 0.05 + value_offset,
        "pre_straight_mean_euclidean_m": 0.02 + value_offset if compound else "",
        "transition_mean_euclidean_m": 0.03 + value_offset if compound else "",
        "post_straight_mean_euclidean_m": 0.04 + value_offset if compound else "",
        "transition_max_euclidean_m": 0.05 + value_offset if compound else "",
        "exit_direction_error_deg": 1.0 + value_offset if compound else "",
    }
    return row


def _move_row(index: int, checkpoint: str) -> dict[str, object]:
    row = _stroke_row("heng", checkpoint, index, index * 0.0001)
    row.update(
        {
            "task": "move",
            "category": "move",
            "rule": "move",
            "condition_id": f"move_{index:02d}",
            "source_character": "mu",
            "source_component_index": index,
        }
    )
    return row


def _source_rows(source_index: int, expected_rows: int) -> list[dict[str, object]]:
    if expected_rows == 40:
        rows = [
            _stroke_row(task, checkpoint, 100 + source_index, source_index * 0.001)
            for task in TASKS
            for checkpoint in ("best", "review")
        ]
        rows.extend(
            _move_row(index, checkpoint)
            for index in range(12)
            for checkpoint in ("best", "review")
        )
        return rows
    checkpoints = ("source", "best", "review") if expected_rows in (24, 12) else (
        "best",
        "review",
    )
    tasks = TASKS if expected_rows != 12 else ("heng", "pie", "na", "shugou")
    return [
        _stroke_row(
            task,
            checkpoint,
            100 + source_index,
            source_index * 0.001 + checkpoint_index * 0.0001,
        )
        for task in tasks
        for checkpoint_index, checkpoint in enumerate(checkpoints)
    ]


def _seal_source(directory: Path, metrics_file: str) -> None:
    lines = []
    for name in (metrics_file, "provenance.json"):
        lines.append(f"{_sha256(directory / name)}  ./{name}")
    (directory / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_config(root: Path) -> tuple[Path, list[dict[str, object]]]:
    sources = []
    for source_index, definition in enumerate(SOURCE_DEFINITIONS):
        (
            source_id,
            variant,
            git_head,
            metrics_file,
            checkpoints,
            expected_rows,
            expected_filtered_rows,
        ) = definition
        directory = root / source_id
        directory.mkdir()
        rows = _source_rows(source_index, expected_rows)
        with (directory / metrics_file).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        _write_json(
            directory / "provenance.json",
            {
                "project": "hanzi_stroke_temporal_composition",
                "variant": variant,
                "git_head": git_head,
                "completed": True,
                "behavioral_pass_fail_defined": False,
                "integrity": {"metrics_rows": expected_rows},
            },
        )
        _seal_source(directory, metrics_file)
        sources.append(
            {
                "id": source_id,
                "directory": str(directory),
                "metrics_file": metrics_file,
                "variant": variant,
                "git_head": git_head,
                "provenance_sha256": _sha256(directory / "provenance.json"),
                "expected_metrics_rows": expected_rows,
                "included_checkpoints": list(checkpoints),
                "expected_filtered_rows": expected_filtered_rows,
            }
        )
    config = {
        "project": "hanzi_stroke_temporal_composition",
        "run_kind": "canonical_checkpoint_metric_review",
        "variant": "canonical_checkpoint_metric_review_v1",
        "enabled": True,
        "tasks": list(TASKS),
        "sources": sources,
        "comparison": {
            "all_task_metrics": list(GENERAL_COMPARISON_METRICS),
            "compound_only_metrics": list(COMPOUND_COMPARISON_METRICS),
            "metric_objective": "minimum",
            "behavioral_pass_fail": "not_defined",
            "automatic_checkpoint_selection": "none",
            "manual_review_required": True,
        },
        "integrity_contract": {
            "source_count": 4,
            "candidate_rows": 56,
            "leader_rows": 58,
            "candidates_per_task": {
                "heng": 8,
                "shu": 6,
                "pie": 8,
                "na": 8,
                "dian": 6,
                "ti": 6,
                "hengzhe": 6,
                "shugou": 8,
            },
        },
        "exclusions": {
            "training": True,
            "move": True,
            "shared_9task": True,
            "formal_75k": True,
            "complete_character_rollout": True,
        },
        "output": {"directory": str(root / "output")},
    }
    config_path = root / "config.json"
    _write_json(config_path, config)
    return config_path, sources


class CanonicalCheckpointReviewTest(unittest.TestCase):
    def test_review_writes_threshold_free_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = _build_config(root)
            provenance = run_review(config_path)
            output = root / "output"
            with (output / "checkpoint_candidate_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                candidate_rows = list(csv.DictReader(handle))
            with (output / "checkpoint_metric_leaders.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                leader_rows = list(csv.DictReader(handle))
            self.assertEqual(len(candidate_rows), 56)
            self.assertEqual(len(leader_rows), 58)
            self.assertFalse(provenance["behavioral_pass_fail_defined"])
            self.assertFalse(
                provenance["automatic_checkpoint_selection_performed"]
            )
            self.assertTrue(provenance["manual_review_required"])

    def test_review_rejects_tampered_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, sources = _build_config(root)
            metrics = (
                Path(sources[0]["directory"]) / str(sources[0]["metrics_file"])
            )
            metrics.write_text(
                metrics.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                run_review(config_path)

    def test_review_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = _build_config(root)
            (root / "output").mkdir()
            with self.assertRaises(FileExistsError):
                run_review(config_path)

    def test_review_rejects_source_provenance_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = _build_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["sources"][0]["provenance_sha256"] = "0" * 64
            _write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "provenance SHA256"):
                run_review(config_path)

    def test_review_requires_compound_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, sources = _build_config(root)
            source = sources[2]
            directory = Path(source["directory"])
            metrics_path = directory / str(source["metrics_file"])
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[-2]["transition_mean_euclidean_m"] = ""
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            _seal_source(directory, str(source["metrics_file"]))
            with self.assertRaisesRegex(ValueError, "required metric is empty"):
                run_review(config_path)


if __name__ == "__main__":
    unittest.main()
