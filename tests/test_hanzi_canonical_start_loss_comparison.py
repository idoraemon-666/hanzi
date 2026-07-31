from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hanzi_writing.canonical_overfit import _task_conditions, _train_task
from hanzi_writing.canonical_start_loss_comparison import (
    ARM_NAMES,
    TASKS,
    _runtime,
    load_config,
)
from losses import onset_window_position_l1, position_l1_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_start_loss_comparison_v1.json"
)
SERVER_SCRIPT = (
    ROOT / "server" / "run_hanzi_canonical_start_loss_comparison.sh"
)


class CanonicalStartLossComparisonTests(unittest.TestCase):
    def test_config_freezes_matched_sixteen_worker_design(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(tuple(config["arms"]), ARM_NAMES)
        self.assertEqual(tuple(config["tasks"]), TASKS)
        self.assertEqual(config["training"]["parallel_processes"], 16)
        self.assertEqual(config["training"]["phase1_updates"], 6000)
        self.assertEqual(config["training"]["phase2_updates"], 2000)
        self.assertEqual(config["optimizer"]["phase1_learning_rate"], 0.001)
        self.assertEqual(config["optimizer"]["phase2_learning_rate"], 0.0001)
        self.assertFalse(config["baseline"]["rerun"])
        self.assertEqual(
            config["baseline"]["shugou_control_extension"][
                "expected_control_rows"
            ],
            21,
        )
        self.assertTrue(config["exclusions"]["shared_8task"])
        self.assertTrue(config["exclusions"]["move"])

    def test_onset_window_changes_only_delay_and_movement_internal_means(self) -> None:
        prediction = torch.zeros((1, 22, 2), dtype=torch.float32)
        target = torch.zeros_like(prediction)
        bounds = {
            "stable": (0, 2),
            "delay": (2, 14),
            "movement": (14, 20),
            "hold": (20, 22),
        }
        prediction[:, 4:14, 0] = 2.0
        prediction[:, 14:19, 0] = 3.0
        baseline = position_l1_metrics(prediction, target, bounds)
        onset = onset_window_position_l1(prediction, target, bounds)
        self.assertAlmostEqual(float(baseline["delay_mean_l1"]), 20.0 / 12.0)
        self.assertAlmostEqual(float(onset["delay_last_10_mean_l1"]), 2.0)
        self.assertAlmostEqual(
            float(onset["onset_delay_mean_l1"]),
            0.5 * (20.0 / 12.0) + 1.0,
            places=6,
        )
        self.assertAlmostEqual(float(baseline["movement_mean_l1"]), 2.5)
        self.assertAlmostEqual(float(onset["movement_first_5_mean_l1"]), 3.0)
        self.assertAlmostEqual(float(onset["onset_movement_mean_l1"]), 2.55)
        expected = 0.1 * (0.5 * (20.0 / 12.0) + 1.0) + 0.6 * 2.55
        self.assertAlmostEqual(float(onset["objective"]), expected, places=6)

    def test_runtime_uses_matched_initialization_and_two_learning_rates(self) -> None:
        config = load_config(CONFIG)
        phase1, _ = _runtime(config, "full_trial", "phase1", "heng")
        phase2, _ = _runtime(config, "full_trial", "phase2", "heng")
        onset, _ = _runtime(config, "onset_window", "phase1", "heng")
        self.assertEqual(phase1["active_rules"], ["heng"])
        self.assertEqual(phase2["active_rules"], ["heng"])
        self.assertEqual(phase1["optimizer"]["learning_rate"], 0.001)
        self.assertEqual(phase2["optimizer"]["learning_rate"], 0.0001)
        self.assertEqual(phase1["training"]["initial_review_updates"], 6000)
        self.assertEqual(phase2["training"]["initial_review_updates"], 2000)
        self.assertEqual(phase1["position_loss"], {"type": "full_trial_l1"})
        self.assertEqual(
            onset["position_loss"]["type"],
            "onset_window_phase_normalized_l1",
        )
        self.assertNotIn("move_sampler", phase1["training"])

    def test_training_accepts_both_new_position_objectives(self) -> None:
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as parent:
            for arm in ARM_NAMES:
                runtime, geometry = _runtime(config, arm, "phase1", "heng")
                runtime = copy.deepcopy(runtime)
                runtime["training"].update(
                    {
                        "initial_review_updates": 1,
                        "validation_interval": 1,
                        "log_interval": 1,
                    }
                )
                output = Path(parent) / arm
                objective = config["arms"][arm]["training_position_objective"]
                _train_task(
                    "heng",
                    dict(_task_conditions(geometry))["heng"],
                    runtime,
                    geometry,
                    output,
                    target_updates=1,
                    checkpoint_variant=config["arms"][arm]["phase1_variant"],
                    checkpoint_prefix=config["arms"][arm][
                        "phase1_checkpoint_prefix"
                    ],
                    training_position_objective=objective,
                    save_scheduled_checkpoints=True,
                )
                checkpoint = torch.load(
                    output / "review_checkpoint.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    checkpoint["training_position_objective"], objective
                )
                row = json.loads(
                    (output / "training_metrics.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                self.assertIn("full_trial_position_l1", row)
                if arm == "onset_window":
                    self.assertIn("delay_last_10_mean_l1", row)
                    self.assertIn("movement_first_5_mean_l1", row)
                phase2_runtime, _ = _runtime(
                    config, arm, "phase2", "heng"
                )
                phase2_runtime = copy.deepcopy(phase2_runtime)
                phase2_runtime["training"].update(
                    {
                        "initial_review_updates": 1,
                        "validation_interval": 1,
                        "log_interval": 1,
                    }
                )
                phase2_output = Path(parent) / f"{arm}_phase2"
                _train_task(
                    "heng",
                    dict(_task_conditions(geometry))["heng"],
                    phase2_runtime,
                    geometry,
                    phase2_output,
                    target_updates=1,
                    initial_checkpoint_path=output / "review_checkpoint.pt",
                    checkpoint_variant=config["arms"][arm]["phase2_variant"],
                    checkpoint_prefix=config["arms"][arm][
                        "phase2_checkpoint_prefix"
                    ],
                    initial_checkpoint_variant=config["arms"][arm][
                        "phase1_variant"
                    ],
                    initial_checkpoint_prefix=config["arms"][arm][
                        "phase1_checkpoint_prefix"
                    ],
                    training_position_objective=objective,
                    save_scheduled_checkpoints=True,
                )
                phase2_checkpoint = torch.load(
                    phase2_output / "review_checkpoint.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    {
                        group["lr"]
                        for group in phase2_checkpoint["optimizer_state_dict"][
                            "param_groups"
                        ]
                    },
                    {0.0001},
                )
                self.assertEqual(
                    phase2_checkpoint["training_position_objective"],
                    objective,
                )

    def test_server_runner_launches_exactly_sixteen_parallel_workers(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("canonical-start-loss-comparison-v1", script)
        self.assertIn("ARMS=(full_trial onset_window)", script)
        self.assertIn(
            "TASKS=(heng shu pie na dian ti hengzhe shugou)", script
        )
        self.assertIn('test "${#WORKER_PIDS[@]}" -eq 16', script)
        self.assertIn(
            "CANONICAL_START_LOSS_COMPARISON_WORKER_COUNT=16", script
        )
        self.assertIn('> "$OUTPUT/worker_logs/$label.log" 2>&1 &', script)
        self.assertIn("CANONICAL_EXISTING_BASELINE_RETRAINED=0", script)
        self.assertIn("SHARED_8TASK_STARTED=0", script)
        self.assertNotIn("TASKS=(heng shu pie na dian ti hengzhe shugou move)", script)


if __name__ == "__main__":
    unittest.main()
