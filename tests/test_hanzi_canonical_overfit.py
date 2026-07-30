from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from hanzi_writing.canonical_overfit import (
    _condition_for_update,
    _movement_metrics,
    _train_task,
    canonical_readonly_validation,
    load_canonical_overfit_config,
    require_approved_stage0,
    validate_frozen_shared_config,
)
from hanzi_writing.canonical_protocol import (
    canonical_stroke_conditions,
    canonical_target_trajectory,
    write_stage0_artifacts,
)
from hanzi_writing.envs import HanziComponentEnv
from hanzi_writing.geometry import move_conditions
from hanzi_writing.training import _hp_from_config, _make_effector
from train import _build_policy


ROOT = Path(__file__).resolve().parents[1]
OVERFIT_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
)
SHARED_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_shared_9task_v1.json"
)
SERVER_PATH = ROOT / "server" / "run_hanzi_canonical_single_task_overfit.sh"
STAGE0_SERVER_PATH = ROOT / "server" / "run_hanzi_canonical_stage0.sh"


class CanonicalOverfitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.geometry = load_canonical_overfit_config(OVERFIT_PATH)

    def test_initial_review_is_6000_without_an_update_cap(self) -> None:
        self.assertEqual(self.config["training"]["initial_review_updates"], 6000)
        self.assertNotIn("max_updates", self.config["training"])

    def test_move_sampler_is_exact_round_robin(self) -> None:
        moves = move_conditions(self.geometry, include_jitter=False)
        first_cycle = [
            _condition_for_update("move", moves, update).condition_id
            for update in range(12)
        ]
        second_cycle = [
            _condition_for_update("move", moves, update).condition_id
            for update in range(12, 24)
        ]
        self.assertEqual(first_cycle, [condition.condition_id for condition in moves])
        self.assertEqual(second_cycle, first_cycle)
        self.assertEqual(len(set(first_cycle)), 12)

    def test_exact_compound_metrics_are_zero_with_unit_path_ratio(self) -> None:
        condition = next(
            value
            for value in canonical_stroke_conditions(self.geometry)
            if value.rule == "hengzhe"
        )
        target = canonical_target_trajectory(condition, self.geometry)
        metrics = _movement_metrics(
            target["points_m"], target["points_m"], target["rounding"]
        )
        self.assertEqual(metrics["movement_mean_euclidean_m"], 0.0)
        self.assertEqual(metrics["endpoint_euclidean_m"], 0.0)
        self.assertEqual(metrics["transition_max_euclidean_m"], 0.0)
        self.assertAlmostEqual(metrics["path_length_ratio"], 1.0, places=12)
        self.assertAlmostEqual(metrics["exit_direction_error_deg"], 0.0, places=12)

    def test_final_validation_is_read_only_and_deterministic(self) -> None:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        hp = _hp_from_config(self.config)
        policy = _build_policy(hp, 6, torch.device("cpu"))
        optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
        env = HanziComponentEnv(
            effector=_make_effector(),
            geometry_config_path=self.config["geometry_config"],
            action_frame_stacking=0,
        )
        condition = (canonical_stroke_conditions(self.geometry)[0],)
        env.reset(
            options={
                "conditions": condition,
                "speed_name": "slow",
                "delay_steps": 50,
                "deterministic": True,
            }
        )
        first = canonical_readonly_validation(
            policy, optimizer, env, hp, condition, self.geometry
        )
        second = canonical_readonly_validation(
            policy, optimizer, env, hp, condition, self.geometry
        )
        self.assertEqual(first["validation"], second["validation"])
        self.assertTrue(all(first["read_only_checks"].values()))
        self.assertTrue(all(second["read_only_checks"].values()))

    def test_shared_config_cannot_be_enabled_without_schema_failure(self) -> None:
        raw = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
        raw["enabled"] = True
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "shared.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_frozen_shared_config(path)

    def test_server_runner_is_authorized_and_hanzi_only(self) -> None:
        script = SERVER_PATH.read_text(encoding="utf-8")
        stage0_script = STAGE0_SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "canonical-single-duration-stage1-parallel6000", script
        )
        self.assertIn("hanzi_writing.canonical_overfit", script)
        self.assertIn("--approved-stage0", script)
        self.assertIn("--target-updates 6000", script)
        self.assertIn("TASKS=(heng shu pie na dian ti hengzhe shugou move)", script)
        self.assertIn('> "$WORKER_LOG_DIR/$task.log" 2>&1 &', script)
        self.assertIn('CANONICAL_PARALLEL_WORKER_COUNT=${#WORKER_PIDS[@]}', script)
        self.assertIn("CANONICAL_SHARED_9TASK_STARTED=0", script)
        self.assertIn("canonical-single-duration-stage0-dev42", stage0_script)
        self.assertIn("hanzi_writing.canonical_protocol", stage0_script)
        for value in (script, stage0_script):
            self.assertNotIn("unittest discover", value)
            self.assertNotIn("test_digit", value)

    def test_stage1_requires_bitwise_identical_approved_stage0(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            approved = Path(parent) / "approved"
            generated = Path(parent) / "generated"
            write_stage0_artifacts(self.geometry, approved)
            write_stage0_artifacts(self.geometry, generated)
            hashes = require_approved_stage0(generated, approved)
            self.assertEqual(len(hashes), 3)
            with (approved / "canonical_target_audit.png").open("ab") as handle:
                handle.write(b"different")
            with self.assertRaises(RuntimeError):
                require_approved_stage0(generated, approved)

    def test_one_update_single_task_smoke_writes_required_checkpoint_files(self) -> None:
        config = copy.deepcopy(self.config)
        config["training"].update(
            {
                "initial_review_updates": 1,
                "validation_interval": 1,
                "log_interval": 1,
            }
        )
        condition = (canonical_stroke_conditions(self.geometry)[0],)
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "heng"
            summary = _train_task(
                "heng", condition, config, self.geometry, output
            )
            self.assertEqual(summary["best_update"], 0)
            self.assertEqual(summary["review_update"], 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "best_checkpoint.pt",
                    "continuation_checkpoint.pt",
                    "review_checkpoint.pt",
                    "training_metrics.jsonl",
                    "validation_metrics.jsonl",
                },
            )

    def test_task_resume_matches_uninterrupted_training_bitwise(self) -> None:
        config = copy.deepcopy(self.config)
        config["training"].update(
            {
                "initial_review_updates": 2,
                "validation_interval": 1,
                "log_interval": 1,
            }
        )
        condition = (canonical_stroke_conditions(self.geometry)[0],)
        with tempfile.TemporaryDirectory() as parent:
            uninterrupted = Path(parent) / "uninterrupted"
            resumed = Path(parent) / "resumed"
            _train_task(
                "heng",
                condition,
                config,
                self.geometry,
                uninterrupted,
                target_updates=2,
            )
            _train_task(
                "heng",
                condition,
                config,
                self.geometry,
                resumed,
                target_updates=1,
            )
            summary = _train_task(
                "heng",
                condition,
                config,
                self.geometry,
                resumed,
                target_updates=2,
            )
            self.assertEqual(summary["resumed_from_update"], 1)
            direct = torch.load(
                uninterrupted / "review_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            continued = torch.load(
                resumed / "review_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(direct["update"], 1)
            self.assertEqual(continued["update"], 1)
            for name, value in direct["agent_state_dict"].items():
                self.assertTrue(torch.equal(value, continued["agent_state_dict"][name]))
            training_rows = [
                json.loads(line)
                for line in (resumed / "training_metrics.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([row["update"] for row in training_rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
