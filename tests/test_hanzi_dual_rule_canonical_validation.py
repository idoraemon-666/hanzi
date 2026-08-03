from __future__ import annotations

from pathlib import Path
import unittest

from hanzi_writing.dual_rule_canonical_validation import CONFIG_PATH, load_config
from hanzi_writing.dual_rule_protocol import rule_names


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "hanzi_writing" / "dual_rule_canonical_validation.py"
SERVER_SCRIPT = ROOT / "server" / "run_hanzi_dual_rule_canonical_validation.sh"


class DualRuleCanonicalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.dual_config, cls.geometry = load_config(CONFIG_PATH)

    def test_config_selects_independent_canonical_best_macro_rollouts(self) -> None:
        contract = self.config["validation"]
        self.assertEqual(contract["candidate"], "best_macro")
        self.assertEqual(contract["loss_arm"], "onset_window")
        self.assertTrue(contract["independent_rollout_per_rule"])
        self.assertTrue(contract["canonical_start_state"])
        self.assertFalse(contract["cross_task_state_carryover"])
        self.assertEqual(
            contract["effector_state_reset_each_rule"], "canonical_standard_state"
        )
        self.assertTrue(contract["recurrent_state_reset_each_rule"])
        self.assertTrue(contract["feedback_buffers_reset_each_rule"])
        self.assertFalse(contract["network_noise"])
        self.assertTrue(contract["deterministic_observation"])
        self.assertEqual(len(rule_names("stroke")), 15)
        self.assertEqual(len(rule_names("move")), 12)

    def test_movement_is_sliced_from_the_same_complete_trial(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn(
            'actual_movement = actual_full[movement_start:movement_end]', source
        )
        self.assertIn(
            'target_movement = target_full[movement_start:movement_end]', source
        )
        self.assertIn('expected[f"best_macro__{condition.rule}__actual"]', source)
        self.assertIn("np.array_equal(actual_movement, expected_actual)", source)
        self.assertIn('bounds["stable"] != (0, 25)', source)
        self.assertIn('bounds["delay"] != (25, 75)', source)
        self.assertIn('bounds["hold"] != (movement_end, movement_end + 25)', source)

    def test_implementation_cannot_train_or_postprocess(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)
        self.assertIn('"trajectory_postprocessing_performed": False', source)
        self.assertIn('"training_started": False', source)

    def test_server_launcher_enforces_evidence_contract(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("frozen-dual-rule-canonical-validation-v1", script)
        self.assertIn("INDEPENDENT_CANONICAL_ROLLOUTS=27", script)
        self.assertIn("SOURCE_MOVEMENT_EXACT_MATCH=1", script)
        self.assertIn("TRAINING_STARTED=0", script)
        self.assertIn("TRAJECTORY_POSTPROCESSING_PERFORMED=0", script)
        self.assertIn("stroke_full_trial_trajectories.png", script)
        self.assertIn("move_full_trial_trajectories.png", script)


if __name__ == "__main__":
    unittest.main()
