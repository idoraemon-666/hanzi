from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hanzi_writing.canonical_overfit import (
    _task_conditions,
    _train_task,
    load_canonical_overfit_config,
)
from hanzi_writing.canonical_shugou_checkpoint_experiment import (
    ARM_NAMES,
    _metrics_for_task,
    load_config,
    rank_task_candidates,
)
from losses import compound_subphase_equal_position_l1, position_l1_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_shugou_checkpoint_experiment_v1.json"
)
BASE_CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
)
SERVER_SCRIPT = (
    ROOT / "server" / "run_hanzi_canonical_shugou_checkpoint_experiment.sh"
)


def _candidate(candidate_id: str, values: list[float]) -> dict[str, object]:
    metrics = _metrics_for_task("shugou")
    return {
        "task": "shugou",
        "candidate_id": candidate_id,
        "_values": dict(zip(metrics, values)),
    }


class CanonicalShugouCheckpointExperimentTests(unittest.TestCase):
    def test_config_freezes_selection_and_matched_two_arm_design(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(tuple(config["arms"]), ARM_NAMES)
        self.assertEqual(config["training"]["batch_size"], 1)
        self.assertEqual(config["training"]["additional_updates_per_arm"], 2000)
        self.assertEqual(config["training"]["validation_interval"], 100)
        self.assertEqual(
            config["intervention"]["changed_factor"],
            "movement_internal_aggregation_only",
        )
        self.assertNotIn("move", config["arms"])

    def test_rank_selection_is_pareto_restricted_and_scale_free(self) -> None:
        metric_count = len(_metrics_for_task("shugou"))
        rows = [
            _candidate("a", [1.0] * (metric_count - 1) + [4.0]),
            _candidate("b", [2.0] * metric_count),
            _candidate("c", [3.0] * metric_count),
        ]
        rankings, selected = rank_task_candidates(rows, "shugou")
        self.assertEqual(selected, "b")
        self.assertFalse(
            next(row for row in rankings if row["candidate_id"] == "c")[
                "is_pareto"
            ]
        )
        scaled = [
            _candidate(
                row["candidate_id"],
                [
                    value * (index + 2)
                    for index, value in enumerate(row["_values"].values())
                ],
            )
            for row in rows
        ]
        scaled_rankings, scaled_selected = rank_task_candidates(scaled, "shugou")
        self.assertEqual(scaled_selected, selected)
        self.assertEqual(
            [
                (
                    row["candidate_id"],
                    row["worst_rank_fraction"],
                    row["mean_rank_fraction"],
                )
                for row in scaled_rankings
            ],
            [
                (
                    row["candidate_id"],
                    row["worst_rank_fraction"],
                    row["mean_rank_fraction"],
                )
                for row in rankings
            ],
        )

    def test_subphase_equal_loss_changes_only_movement_internal_average(self) -> None:
        prediction = torch.zeros((1, 11, 2), dtype=torch.float32)
        target = torch.zeros_like(prediction)
        prediction[0, 2:8, 0] = 1.0
        prediction[0, 8:10, 0] = 9.0
        bounds = {
            "stable": (0, 1),
            "delay": (1, 2),
            "movement": (2, 10),
            "hold": (10, 11),
        }
        labels = (
            *(("pre_straight",) * 6),
            "transition",
            "post_straight",
        )
        baseline = position_l1_metrics(prediction, target, bounds)
        intervention = compound_subphase_equal_position_l1(
            prediction, target, bounds, labels
        )
        self.assertAlmostEqual(float(baseline["movement_mean_l1"]), 3.0)
        self.assertAlmostEqual(
            float(intervention["movement_subphase_equal_mean_l1"]),
            19.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(intervention["objective"]), 0.6 * 19.0 / 3.0, places=6
        )
        self.assertAlmostEqual(float(intervention["pre_straight_mean_l1"]), 1.0)
        self.assertAlmostEqual(float(intervention["transition_mean_l1"]), 9.0)
        self.assertAlmostEqual(float(intervention["post_straight_mean_l1"]), 9.0)

    def test_training_saves_auditable_intervention_candidates(self) -> None:
        base, geometry = load_canonical_overfit_config(BASE_CONFIG)
        source_runtime = copy.deepcopy(base)
        source_runtime["variant"] = "test_shugou_source"
        source_runtime["training"].update(
            {"initial_review_updates": 1, "validation_interval": 1, "log_interval": 1}
        )
        conditions = dict(_task_conditions(geometry))["shugou"]
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            source = root / "source"
            _train_task(
                "shugou",
                conditions,
                source_runtime,
                geometry,
                source,
                target_updates=1,
                checkpoint_variant="test_shugou_source",
                checkpoint_prefix="test_shugou_source",
            )
            runtime = copy.deepcopy(source_runtime)
            runtime["variant"] = "test_shugou_subphase_equal"
            experimental = root / "experimental"
            result = _train_task(
                "shugou",
                conditions,
                runtime,
                geometry,
                experimental,
                target_updates=1,
                initial_checkpoint_path=source / "review_checkpoint.pt",
                checkpoint_variant="test_shugou_subphase_equal",
                checkpoint_prefix="test_shugou_subphase_equal",
                initial_checkpoint_variant="test_shugou_source",
                initial_checkpoint_prefix="test_shugou_source",
                training_position_objective="compound_subphase_equal_l1",
                save_scheduled_checkpoints=True,
            )
            self.assertEqual(result["review_update"], 0)
            candidate = torch.load(
                experimental / "scheduled_checkpoints" / "update_000000.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                candidate["checkpoint_kind"],
                "test_shugou_subphase_equal_candidate",
            )
            self.assertEqual(
                candidate["training_position_objective"],
                "compound_subphase_equal_l1",
            )
            training_row = json.loads(
                (experimental / "training_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("movement_subphase_equal_mean_l1", training_row)
            self.assertIn("transition_mean_l1", training_row)

    def test_server_runner_is_single_batch_and_excludes_shared_training(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("canonical-shugou-checkpoint-experiment-v1", script)
        self.assertIn("CANONICAL_CHECKPOINT_SELECTION_COMPLETE=1", script)
        self.assertIn("for arm in control subphase_equal", script)
        self.assertIn("CANONICAL_SHUGOU_CHECKPOINT_EXPERIMENT_COMPLETE=1", script)
        self.assertIn("SHARED_9TASK_STARTED=0", script)
        self.assertNotIn("canonical_shared", script)
        self.assertNotIn("TASKS=(heng shu pie na dian ti hengzhe shugou move)", script)


if __name__ == "__main__":
    unittest.main()
