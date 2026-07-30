from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hanzi_writing.canonical_overfit import _train_task, load_canonical_overfit_config
from hanzi_writing.canonical_protocol import canonical_stroke_conditions
from hanzi_writing.canonical_refinement import (
    EXPECTED_SOURCE_HEAD,
    REFINEMENT_PREFIX,
    REFINEMENT_TASKS,
    REFINEMENT_VARIANT,
    load_refinement_config,
    select_source_checkpoints,
)
from hanzi_writing.training import _sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_stroke_refinement_v1.json"
)
SERVER_PATH = ROOT / "server" / "run_hanzi_canonical_stroke_refinement.sh"


class CanonicalRefinementTests(unittest.TestCase):
    def test_config_freezes_eight_strokes_and_excludes_move(self) -> None:
        config, runtime, _ = load_refinement_config(CONFIG_PATH)
        self.assertEqual(tuple(config["active_rules"]), REFINEMENT_TASKS)
        self.assertNotIn("move", config["active_rules"])
        self.assertTrue(config["exclusions"]["move"])
        self.assertEqual(config["optimizer"]["learning_rate"], 0.0001)
        self.assertEqual(config["training"]["additional_updates"], 2000)
        self.assertEqual(config["training"]["validation_interval"], 100)
        self.assertEqual(config["training"]["parallel_processes"], 8)
        self.assertEqual(runtime["optimizer"]["learning_rate"], 0.0001)
        self.assertNotIn("move_sampler", runtime["training"])

    def test_source_selection_uses_lower_loss_and_best_on_exact_tie(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            source = Path(parent) / "source"
            hashes = {}
            summaries = []
            for index, task in enumerate(REFINEMENT_TASKS):
                directory = source / "models" / task
                directory.mkdir(parents=True)
                best_loss = 0.1 + index
                review_loss = best_loss if index == 0 else best_loss - 0.01
                for identity, loss, update in (
                    ("best", best_loss, 1000 + index),
                    ("review", review_loss, 5999),
                ):
                    path = directory / f"{identity}_checkpoint.pt"
                    torch.save(
                        {
                            "project": "hanzi_stroke_temporal_composition",
                            "variant": "canonical_single_duration_overfit_v1",
                            "checkpoint_kind": f"canonical_single_task_{identity}",
                            "task": task,
                            "update": update,
                            "validation_loss": loss,
                        },
                        path,
                    )
                    hashes[str(path.relative_to(source)).replace("\\", "/")] = (
                        _sha256_file(path)
                    )
                summaries.append({"task": task, "completed_updates": 6000})
            provenance = {
                "completed": True,
                "variant": "canonical_single_duration_overfit_v1",
                "git_head": EXPECTED_SOURCE_HEAD,
                "execution_mode": "nine_independent_parallel_processes",
                "task_summaries": summaries,
                "checkpoint_sha256": hashes,
                "shared_9task_started": False,
                "formal_75k_started": False,
                "complete_character_rollout_started": False,
            }
            (source / "provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            manifest = select_source_checkpoints(CONFIG_PATH, source)
            selected = {
                value["task"]: value["selected_checkpoint"]
                for value in manifest["selections"]
            }
            self.assertEqual(selected[REFINEMENT_TASKS[0]], "best")
            self.assertTrue(
                all(selected[task] == "review" for task in REFINEMENT_TASKS[1:])
            )
            self.assertFalse(manifest["move_included"])

    def test_refinement_initialization_overrides_only_optimizer_learning_rate(self) -> None:
        base, geometry = load_canonical_overfit_config(
            ROOT
            / "configurations"
            / "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
        )
        source_config = copy.deepcopy(base)
        source_config["training"].update(
            {
                "initial_review_updates": 1,
                "validation_interval": 1,
                "log_interval": 1,
            }
        )
        _, runtime, _ = load_refinement_config(CONFIG_PATH)
        runtime["training"].update(
            {
                "initial_review_updates": 1,
                "validation_interval": 1,
                "log_interval": 1,
            }
        )
        condition = (canonical_stroke_conditions(geometry)[0],)
        with tempfile.TemporaryDirectory() as parent:
            source_output = Path(parent) / "source"
            refinement_output = Path(parent) / "refinement"
            _train_task(
                "heng", condition, source_config, geometry, source_output
            )
            summary = _train_task(
                "heng",
                condition,
                runtime,
                geometry,
                refinement_output,
                target_updates=1,
                initial_checkpoint_path=source_output / "best_checkpoint.pt",
                checkpoint_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
            checkpoint = torch.load(
                refinement_output / "review_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(summary["review_update"], 0)
            self.assertEqual(checkpoint["variant"], REFINEMENT_VARIANT)
            self.assertEqual(
                checkpoint["checkpoint_kind"], f"{REFINEMENT_PREFIX}_review"
            )
            self.assertEqual(
                {group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]},
                {0.0001},
            )
            resumed = _train_task(
                "heng",
                condition,
                runtime,
                geometry,
                refinement_output,
                target_updates=2,
                initial_checkpoint_path=source_output / "best_checkpoint.pt",
                checkpoint_variant=REFINEMENT_VARIANT,
                checkpoint_prefix=REFINEMENT_PREFIX,
            )
            self.assertEqual(resumed["resumed_from_update"], 1)
            self.assertEqual(resumed["review_update"], 1)

    def test_server_runner_launches_exactly_eight_strokes_in_parallel(self) -> None:
        script = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "canonical-eight-stroke-refinement-lr1e4-2000", script
        )
        self.assertIn(
            "TASKS=(heng shu pie na dian ti hengzhe shugou)", script
        )
        self.assertIn('> "$WORKER_LOG_DIR/$task.log" 2>&1 &', script)
        self.assertIn(
            'REFINEMENT_PARALLEL_WORKER_COUNT=${#WORKER_PIDS[@]}', script
        )
        self.assertIn("MOVE_REFINEMENT_STARTED=0", script)
        self.assertNotIn("TASKS=(heng shu pie na dian ti hengzhe shugou move)", script)
        self.assertNotIn("unittest discover", script)
        self.assertNotIn("test_digit", script)


if __name__ == "__main__":
    unittest.main()
