from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from hanzi_writing.dual_controller_continuous_state_composition import (
    CONFIG_PATH,
    PHYSICAL_STATE_KEYS,
    ContinuousStateDualFixedRuleEnv,
    _clone_physical_state,
    load_config,
)
from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    FIXED_SPEED_NAME,
    conditions,
)
from hanzi_writing.training import _make_effector


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "hanzi_writing" / "dual_controller_continuous_state_composition.py"
SERVER_SCRIPT = (
    ROOT / "server" / "run_hanzi_frozen_dual_controller_continuous_state_composition.sh"
)


class FrozenDualControllerContinuousStateCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.dual_config, cls.geometry = load_config(CONFIG_PATH)

    def test_config_freezes_full_physical_state_contract(self) -> None:
        composition = self.config["composition"]
        self.assertEqual(
            composition["intertrial_effector_operation"], "none_shared_instance"
        )
        self.assertEqual(
            composition["physical_state_carryover"], list(PHYSICAL_STATE_KEYS)
        )
        self.assertTrue(composition["recurrent_state_reset_each_trial"])
        self.assertFalse(composition["translate_canonical_target"])
        self.assertEqual(
            self.config["render"]["trajectory"],
            "actual_complete_trial_including_physical_start_state",
        )
        self.assertFalse(self.config["render"]["postprocessing"])
        self.assertEqual(
            self.config["source"]["checkpoints"]["stroke"]["loss_arm"],
            "onset_window",
        )
        self.assertEqual(
            self.config["source"]["checkpoints"]["move"]["loss_arm"],
            "onset_window",
        )

    def test_trial_switch_preserves_all_effector_states_bitwise(self) -> None:
        effector = _make_effector()
        stroke_env = ContinuousStateDualFixedRuleEnv(
            effector=effector,
            model_kind="stroke",
            geometry_config_path=self.dual_config["geometry_config"],
            action_frame_stacking=0,
        )
        move_env = ContinuousStateDualFixedRuleEnv(
            effector=effector,
            model_kind="move",
            geometry_config_path=self.dual_config["geometry_config"],
            action_frame_stacking=0,
        )
        self.assertIs(stroke_env.effector, move_env.effector)
        stroke_condition = conditions("stroke", self.geometry)[0]
        move_condition = conditions("move", self.geometry)[0]
        stroke_env.reset(
            options={
                "conditions": (stroke_condition,),
                "speed_name": FIXED_SPEED_NAME,
                "delay_steps": FIXED_DELAY_STEPS,
                "deterministic": True,
            }
        )
        action = torch.full((1, effector.n_muscles), 0.2, dtype=torch.float32)
        stroke_env.step(0, action=action)
        before = _clone_physical_state(effector)
        _, info = move_env.continue_from_current_state(
            options={
                "conditions": (move_condition,),
                "speed_name": FIXED_SPEED_NAME,
                "delay_steps": FIXED_DELAY_STEPS,
                "deterministic": True,
            }
        )
        after = _clone_physical_state(effector)
        self.assertEqual(tuple(before), PHYSICAL_STATE_KEYS)
        self.assertEqual(tuple(after), PHYSICAL_STATE_KEYS)
        for key in PHYSICAL_STATE_KEYS:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        fingertip = info["states"]["fingertip"]
        self.assertTrue(torch.isfinite(fingertip).all())
        movement_start, movement_end = move_env.epoch_bounds["movement"]
        target = (
            move_env.traj[0, movement_start:movement_end].detach().cpu().numpy()
            - move_env.anchor_m
        )
        self.assertTrue(
            np.allclose(target, move_condition.points_m, rtol=0.0, atol=1e-6)
        )
        for name in ("vision", "proprioception"):
            values = move_env.obs_buffer[name]
            self.assertTrue(values)
            self.assertTrue(all(torch.equal(values[0], value) for value in values[1:]))

    def test_continuation_method_cannot_reset_effector(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        method_start = source.index("    def continue_from_current_state(")
        method_end = source.index("\n\n\ndef _clone_physical_state", method_start)
        method = source[method_start:method_end]
        self.assertNotIn("effector.reset", method)
        self.assertNotIn("_reset_effector", method)
        self.assertIn("_initialize_observation_buffers", method)

    def test_implementation_cannot_train_or_postprocess(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)
        self.assertIn('points = segment["result"]["actual_full_with_start"]', source)
        self.assertNotIn('points = segment["result"]["movement_actual"]', source)
        self.assertIn('x = torch.zeros((1, hp["hid_size"])', source)
        self.assertIn("EXPECTED_BOUNDARIES = 24", source)

    def test_server_launcher_gates_continuous_state_evidence(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("frozen-dual-controller-continuous-physical-state-v1", script)
        self.assertIn("CONTINUOUS_PHYSICAL_STATE_BOUNDARIES=24", script)
        self.assertIn("INTERTRIAL_EFFECTOR_RESETS=0", script)
        self.assertIn("PHYSICAL_STATE_BITWISE_CONTINUITY=1", script)
        self.assertIn("physical_state_boundary_audit.json", script)
        self.assertIn("FROZEN_DUAL_CONTROLLER_CONTINUOUS_STATE_COMPLETE=1", script)
        self.assertIn("TRAINING_STARTED=0", script)
        self.assertIn("TRAJECTORY_POSTPROCESSING_PERFORMED=0", script)


if __name__ == "__main__":
    unittest.main()
