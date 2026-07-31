from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from hanzi_writing.dual_rule_protocol import LOSS_ARMS, condition_manifest, conditions
from hanzi_writing.dual_rule_training import (
    JOINT_GRADIENT_CONFIG_PATH,
    JOINT_GRADIENT_VARIANT,
    _make_env,
    _optimizer_step_mode,
    _runtime_hp,
    _training_conditions,
    _training_step,
    _updates_per_rule,
    _validation_interval,
    load_config,
    validate_checkpoint_identity,
)
from train import _build_policy


ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "server" / "run_hanzi_dual_fixed_rule_joint_gradient.sh"


class DualFixedRuleJointGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.geometry = load_config(JOINT_GRADIENT_CONFIG_PATH)

    def test_configuration_freezes_joint_gradient_schedule(self) -> None:
        self.assertEqual(self.config["variant"], JOINT_GRADIENT_VARIANT)
        self.assertEqual(tuple(self.config["loss_arms"]), LOSS_ARMS)
        self.assertEqual(
            self.config["training"],
            {
                "microbatch_size_per_rule": 1,
                "optimizer_step_mode": "all_rules_mean_gradient",
                "scheduler": "all_rules_every_optimizer_step",
                "updates_per_rule": 8000,
                "effective_rules_per_optimizer_step": {
                    "stroke": 15,
                    "move": 12,
                },
                "network_noise": False,
                "deterministic_observation": True,
                "parallel_processes": 6,
            },
        )
        for model_kind, rule_count, validation_interval in (
            ("stroke", 15, 40),
            ("move", 12, 50),
        ):
            schedule = self.config["models"][model_kind]
            self.assertEqual(schedule["max_updates"], 8000)
            self.assertEqual(schedule["phase1_updates"], 6000)
            self.assertEqual(schedule["phase2_updates"], 2000)
            self.assertEqual(schedule["log_interval"], 10)
            self.assertEqual(schedule["validation_interval"], validation_interval)
            self.assertEqual(_updates_per_rule(self.config, model_kind), 8000)
            self.assertEqual(
                schedule["max_updates"] * rule_count,
                120000 if model_kind == "stroke" else 96000,
            )

    def test_every_optimizer_step_contains_every_rule_once(self) -> None:
        self.assertEqual(_optimizer_step_mode(self.config), "all_rules_mean_gradient")
        for model_kind, expected_count in (("stroke", 15), ("move", 12)):
            library = conditions(model_kind, self.geometry)
            by_rule = {condition.rule: condition for condition in library}
            selected = _training_conditions(
                self.config, model_kind, 137, library, by_rule
            )
            self.assertEqual(selected, library)
            self.assertEqual(len(selected), expected_count)
            self.assertEqual(
                _validation_interval(self.config, model_kind),
                600 // expected_count,
            )

    def test_duplicate_microtasks_are_meaned_not_summed(self) -> None:
        model_kind = "stroke"
        hp = _runtime_hp(self.config, model_kind)
        condition = conditions(model_kind, self.geometry)[7]
        torch.manual_seed(42)
        single_policy = _build_policy(hp, 6, torch.device("cpu"))
        duplicate_policy = _build_policy(hp, 6, torch.device("cpu"))
        duplicate_policy.load_state_dict(single_policy.state_dict())
        single_optimizer = torch.optim.Adam(single_policy.parameters(), lr=0.001)
        duplicate_optimizer = torch.optim.Adam(
            duplicate_policy.parameters(), lr=0.001
        )

        single = _training_step(
            single_policy,
            single_optimizer,
            _make_env(self.config, model_kind),
            hp,
            "baseline",
            (condition,),
        )
        duplicate = _training_step(
            duplicate_policy,
            duplicate_optimizer,
            _make_env(self.config, model_kind),
            hp,
            "baseline",
            (condition, condition),
        )
        self.assertTrue(np.isfinite(single["mean_total_loss"]))
        self.assertAlmostEqual(
            single["mean_total_loss"], duplicate["mean_total_loss"], places=8
        )
        for name, value in single_policy.state_dict().items():
            self.assertTrue(
                torch.allclose(value, duplicate_policy.state_dict()[name], atol=1e-7),
                name,
            )

    def test_joint_checkpoint_identity_is_isolated(self) -> None:
        manifest = condition_manifest("stroke", self.geometry)
        checkpoint = {
            "project": "hanzi_stroke_temporal_composition",
            "variant": JOINT_GRADIENT_VARIANT,
            "model_kind": "stroke",
            "loss_arm": "baseline",
            "rule_dim": 15,
            "input_size": 33,
            "rule_names": list(manifest["rule_names"]),
            "condition_manifest_sha256": manifest["condition_manifest_sha256"],
            "hp": {"inp_size": 33},
        }
        validate_checkpoint_identity(
            checkpoint,
            "stroke",
            "baseline",
            manifest,
            JOINT_GRADIENT_VARIANT,
        )
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_checkpoint_identity(
                checkpoint, "stroke", "baseline", manifest
            )

    def test_server_launcher_freezes_six_joint_workers(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("MODEL_KINDS=(stroke move)", script)
        self.assertIn("ARMS=(baseline full_trial onset_window)", script)
        self.assertIn('test "${#WORKER_PIDS[@]}" -eq 6', script)
        self.assertIn("OPTIMIZER_STEP_MODE=all_rules_mean_gradient", script)
        self.assertIn("STROKE_RULES_PER_STEP=15", script)
        self.assertIn("MOVE_RULES_PER_STEP=12", script)
        self.assertIn("RULE_EXPOSURES_PER_RULE=8000", script)


if __name__ == "__main__":
    unittest.main()
