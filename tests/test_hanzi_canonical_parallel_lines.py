from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from hanzi_writing.canonical_overfit import _task_conditions, _train_task
from hanzi_writing.canonical_parallel_lines import (
    FOLLOWUP_TASKS,
    SCRATCH_PREFIX,
    SCRATCH_TASKS,
    SCRATCH_VARIANT,
    _validate_refinement_task_files,
    finalize_followup,
    finalize_scratch,
    load_followup_config,
    load_scratch_config,
    prepare_followup,
    prepare_scratch,
    train_followup_task,
)
from hanzi_writing.canonical_protocol import write_stage0_artifacts
from hanzi_writing.canonical_refinement import (
    REFINEMENT_PREFIX,
    REFINEMENT_VARIANT,
    load_refinement_config,
)
from hanzi_writing.training import _sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRATCH_CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_scratch_lr1e4_v1.json"
)
FOLLOWUP_CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_selected_refinement_followup_v1.json"
)
REFINEMENT_CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_stroke_refinement_v1.json"
)
SERVER_SCRIPT = ROOT / "server" / "run_hanzi_canonical_parallel_lr_lines.sh"


class CanonicalParallelLinesTests(unittest.TestCase):
    def test_scratch_config_freezes_eight_fresh_low_lr_tasks(self) -> None:
        config, runtime, _ = load_scratch_config(SCRATCH_CONFIG)
        self.assertEqual(tuple(config["active_rules"]), SCRATCH_TASKS)
        self.assertNotIn("move", config["active_rules"])
        self.assertEqual(config["optimizer"]["learning_rate"], 0.0001)
        self.assertEqual(config["optimizer"]["initialization"], "fresh_seed_42")
        self.assertEqual(config["training"]["updates"], 10000)
        self.assertEqual(config["training"]["parallel_processes"], 8)
        self.assertEqual(runtime["optimizer"]["learning_rate"], 0.0001)
        self.assertEqual(runtime["training"]["initial_review_updates"], 10000)
        self.assertNotIn("move_sampler", runtime["training"])

    def test_followup_config_freezes_four_exact_continuations(self) -> None:
        config, runtime, _ = load_followup_config(FOLLOWUP_CONFIG)
        _, refinement_runtime, _ = load_refinement_config(REFINEMENT_CONFIG)
        self.assertEqual(tuple(config["active_rules"]), FOLLOWUP_TASKS)
        self.assertEqual(FOLLOWUP_TASKS, ("heng", "pie", "na", "shugou"))
        self.assertEqual(config["training"]["additional_updates"], 2000)
        self.assertEqual(config["training"]["target_total_refinement_updates"], 4000)
        self.assertEqual(config["training"]["parallel_processes"], 4)
        self.assertEqual(runtime, refinement_runtime)

    def test_scratch_checkpoint_identity_is_isolated(self) -> None:
        _, runtime, geometry = load_scratch_config(SCRATCH_CONFIG)
        runtime = copy.deepcopy(runtime)
        runtime["training"].update(
            {"initial_review_updates": 1, "validation_interval": 1, "log_interval": 1}
        )
        conditions = dict(_task_conditions(geometry))["heng"]
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "scratch"
            _train_task(
                "heng",
                conditions,
                runtime,
                geometry,
                output,
                target_updates=1,
                checkpoint_variant=SCRATCH_VARIANT,
                checkpoint_prefix=SCRATCH_PREFIX,
            )
            checkpoint = torch.load(
                output / "review_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["variant"], SCRATCH_VARIANT)
            self.assertEqual(
                checkpoint["checkpoint_kind"], f"{SCRATCH_PREFIX}_review"
            )
            self.assertEqual(checkpoint["update"], 0)

    def test_copied_refinement_continuation_matches_uninterrupted(self) -> None:
        _, runtime, geometry = load_refinement_config(REFINEMENT_CONFIG)
        runtime = copy.deepcopy(runtime)
        runtime["training"].update(
            {"initial_review_updates": 2, "validation_interval": 1, "log_interval": 1}
        )
        conditions = dict(_task_conditions(geometry))["heng"]
        with tempfile.TemporaryDirectory() as parent:
            source = Path(parent) / "source"
            resumed = Path(parent) / "resumed"
            direct = Path(parent) / "direct"
            _train_task(
                "heng",
                conditions,
                runtime,
                geometry,
                source,
                target_updates=1,
                checkpoint_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
            hashes = _validate_refinement_task_files(
                source, "heng", runtime, expected_updates=1
            )
            self.assertEqual(set(hashes), {path.name for path in source.iterdir()})
            shutil.copytree(source, resumed)
            _train_task(
                "heng",
                conditions,
                runtime,
                geometry,
                resumed,
                target_updates=2,
                checkpoint_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
            _train_task(
                "heng",
                conditions,
                runtime,
                geometry,
                direct,
                target_updates=2,
                checkpoint_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
            continued = torch.load(
                resumed / "continuation_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            uninterrupted = torch.load(
                direct / "continuation_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(continued["update"], 1)
            for name, value in continued["agent_state_dict"].items():
                self.assertTrue(torch.equal(value, uninterrupted["agent_state_dict"][name]))
            self.assertEqual(
                continued["validation_loss"], uninterrupted["validation_loss"]
            )

    def test_server_runner_launches_twelve_workers_without_move(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "canonical-parallel-scratch1e4-10000-selected-followup2000", script
        )
        self.assertIn("SCRATCH_TASKS=(heng shu pie na dian ti hengzhe shugou)", script)
        self.assertIn("FOLLOWUP_TASKS=(heng pie na shugou)", script)
        self.assertIn("CANONICAL_TOTAL_PARALLEL_WORKER_COUNT=12", script)
        self.assertIn("CANONICAL_SCRATCH_LOW_LR_STARTED=1", script)
        self.assertIn("CANONICAL_SELECTED_FOLLOWUP_STARTED=1", script)
        self.assertNotIn("TASKS=(heng shu pie na dian ti hengzhe shugou move)", script)
        self.assertNotIn("unittest discover", script)
        self.assertNotIn("test_digit", script)

    def test_miniature_two_line_finalization_contracts(self) -> None:
        scratch_config, scratch_runtime, geometry = load_scratch_config(
            SCRATCH_CONFIG
        )
        followup_config, refinement_runtime, _ = load_followup_config(
            FOLLOWUP_CONFIG
        )

        def clone_task(source: Path, target: Path, task: str) -> None:
            shutil.copytree(source, target)
            for identity in ("best", "review", "continuation"):
                path = target / f"{identity}_checkpoint.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                checkpoint["task"] = task
                torch.save(checkpoint, path)

        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            approved = root / "approved"
            write_stage0_artifacts(geometry, approved)

            miniature_scratch = copy.deepcopy(scratch_config)
            miniature_scratch["training"]["updates"] = 1
            miniature_scratch["output"]["directory"] = str(root / "scratch")
            miniature_scratch_runtime = copy.deepcopy(scratch_runtime)
            miniature_scratch_runtime["training"].update(
                {
                    "initial_review_updates": 1,
                    "validation_interval": 1,
                    "log_interval": 1,
                }
            )
            miniature_scratch_runtime["output"] = copy.deepcopy(
                miniature_scratch["output"]
            )
            with patch(
                "hanzi_writing.canonical_parallel_lines.load_scratch_config",
                return_value=(
                    miniature_scratch,
                    miniature_scratch_runtime,
                    geometry,
                ),
            ):
                prepare_scratch(SCRATCH_CONFIG, approved)
                scratch_models = root / "scratch" / "models"
                _train_task(
                    "heng",
                    dict(_task_conditions(geometry))["heng"],
                    miniature_scratch_runtime,
                    geometry,
                    scratch_models / "heng",
                    target_updates=1,
                    checkpoint_variant=SCRATCH_VARIANT,
                    checkpoint_prefix=SCRATCH_PREFIX,
                )
                for task in SCRATCH_TASKS[1:]:
                    clone_task(scratch_models / "heng", scratch_models / task, task)
                scratch_provenance = finalize_scratch(SCRATCH_CONFIG, approved)
            self.assertEqual(
                scratch_provenance["integrity"],
                {
                    "stroke_tasks_completed": 8,
                    "metrics_rows": 16,
                    "trajectory_rows": 2616,
                    "plots": 9,
                    "all_numeric_values_finite": True,
                },
            )

            refinement_seed = root / "refinement_seed"
            _train_task(
                "heng",
                dict(_task_conditions(geometry))["heng"],
                refinement_runtime,
                geometry,
                refinement_seed,
                target_updates=1,
                checkpoint_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
            source = root / "source_refinement"
            source_models = source / "models"
            source_models.mkdir(parents=True)
            checkpoint_hashes = {}
            for task in FOLLOWUP_TASKS:
                target = source_models / task
                clone_task(refinement_seed, target, task)
                for identity in ("best", "review", "continuation"):
                    path = target / f"{identity}_checkpoint.pt"
                    checkpoint_hashes[
                        str(path.relative_to(source)).replace("\\", "/")
                    ] = _sha256_file(path)
            source_provenance = {
                "completed": True,
                "variant": REFINEMENT_VARIANT,
                "git_head": "d763b59ccc09d6e8673f518f0aead89094fb2226",
                "learning_rate": 0.0001,
                "task_summaries": [
                    {"task": task, "completed_additional_updates": 1}
                    for task in SCRATCH_TASKS
                ],
                "checkpoint_sha256": checkpoint_hashes,
                "move_started": False,
                "shared_9task_started": False,
                "formal_75k_started": False,
                "complete_character_rollout_started": False,
            }
            (source / "provenance.json").write_text(
                json.dumps(source_provenance), encoding="utf-8"
            )

            miniature_followup = copy.deepcopy(followup_config)
            miniature_followup["source_contract"][
                "completed_additional_updates"
            ] = 1
            miniature_followup["training"].update(
                {"additional_updates": 1, "target_total_refinement_updates": 2}
            )
            miniature_followup["output"]["directory"] = str(root / "followup")
            with patch(
                "hanzi_writing.canonical_parallel_lines.load_followup_config",
                return_value=(miniature_followup, refinement_runtime, geometry),
            ):
                prepare_followup(FOLLOWUP_CONFIG, source)
                for task in FOLLOWUP_TASKS:
                    train_followup_task(FOLLOWUP_CONFIG, source, task)
                followup_provenance = finalize_followup(FOLLOWUP_CONFIG, source)
            self.assertEqual(
                followup_provenance["integrity"],
                {
                    "stroke_tasks_completed": 4,
                    "metrics_rows": 12,
                    "trajectory_rows": 1962,
                    "plots": 5,
                    "all_numeric_values_finite": True,
                },
            )
            self.assertTrue(followup_provenance["exact_continuation"])


if __name__ == "__main__":
    unittest.main()
