from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from hanzi_writing.dual_controller_composition import (
    CONFIG_PATH,
    EXPECTED_RULE_SEQUENCES,
    ActualEndpointDualFixedRuleEnv,
    character_sequences,
    load_config,
)
from hanzi_writing.dual_rule_protocol import FIXED_DELAY_STEPS, conditions
from hanzi_writing.training import _make_effector


ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "server" / "run_hanzi_frozen_dual_controller_composition.sh"
MODULE = ROOT / "hanzi_writing" / "dual_controller_composition.py"


class FrozenDualControllerCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.dual_config, cls.geometry = load_config(CONFIG_PATH)

    def test_config_freezes_selected_joint_onset_window_checkpoints(self) -> None:
        self.assertEqual(
            self.config["source"]["checkpoints"]["stroke"]["loss_arm"],
            "onset_window",
        )
        self.assertEqual(
            self.config["source"]["checkpoints"]["move"]["loss_arm"],
            "onset_window",
        )
        self.assertEqual(
            self.config["source"]["checkpoints"]["stroke"]["update"], 7999
        )
        self.assertEqual(
            self.config["source"]["checkpoints"]["move"]["update"], 7599
        )
        self.assertFalse(self.config["composition"]["translate_canonical_target"])
        self.assertFalse(self.config["render"]["postprocessing"])
        self.assertEqual(
            self.config["render"]["trajectory"],
            "actual_complete_trial_including_reset_state",
        )

    def test_three_character_sequences_alternate_stroke_and_move(self) -> None:
        sequences = character_sequences(self.geometry)
        self.assertEqual(tuple(sequences), ("mu", "jiang", "ke"))
        self.assertEqual(
            {name: len(sequence) for name, sequence in sequences.items()},
            {"mu": 7, "jiang": 11, "ke": 9},
        )
        for character, sequence in sequences.items():
            actual = tuple((condition.model_kind, condition.rule) for condition in sequence)
            self.assertEqual(actual, EXPECTED_RULE_SEQUENCES[character])
            self.assertEqual(
                tuple(condition.model_kind for condition in sequence[::2]),
                ("stroke",) * len(sequence[::2]),
            )
            self.assertEqual(
                tuple(condition.model_kind for condition in sequence[1::2]),
                ("move",) * len(sequence[1::2]),
            )

    def test_actual_endpoint_reset_does_not_translate_canonical_target(self) -> None:
        condition = conditions("stroke", self.geometry)[0]
        actual_start = condition.points_m[0] + np.asarray((0.001, -0.001))
        env = ActualEndpointDualFixedRuleEnv(
            effector=_make_effector(),
            model_kind="stroke",
            geometry_config_path=self.dual_config["geometry_config"],
            action_frame_stacking=0,
        )
        _, info = env.reset(
            options={
                "conditions": (condition,),
                "speed_name": "slow",
                "delay_steps": FIXED_DELAY_STEPS,
                "deterministic": True,
                "actual_start_xy_m": actual_start,
            }
        )
        reset_xy = (
            info["states"]["fingertip"].detach().cpu().numpy()[0]
            - env.anchor_m
        )
        self.assertTrue(np.allclose(reset_xy, actual_start, rtol=0.0, atol=1e-6))
        movement_start, movement_end = env.epoch_bounds["movement"]
        target = (
            env.traj[0, movement_start:movement_end].detach().cpu().numpy()
            - env.anchor_m
        )
        self.assertTrue(np.allclose(target, condition.points_m, rtol=0.0, atol=1e-6))
        self.assertFalse(np.allclose(target[0], actual_start, rtol=0.0, atol=1e-6))
        for name in ("vision", "proprioception"):
            values = env.obs_buffer[name]
            self.assertTrue(values)
            self.assertTrue(all(torch.equal(values[0], value) for value in values[1:]))

    def test_implementation_cannot_train_or_postprocess(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)
        self.assertIn('style = "-" if model_kind == "stroke" else "--"', source)
        self.assertIn(
            'points = segment["result"]["actual_full_with_reset"]', source
        )
        self.assertNotIn(
            'points = segment["result"]["movement_actual"]', source
        )
        self.assertIn('hold_bounds[1] != timestep', source)

    def test_server_launcher_requires_frozen_authorization_and_archive(self) -> None:
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("frozen-dual-controller-full-trial-composition-v2", script)
        self.assertIn("test -z \"$(git -C \"$REPO\" status --short)\"", script)
        self.assertIn("test -f \"$SOURCE_RESULTS/provenance.json\"", script)
        self.assertIn("FROZEN_DUAL_CONTROLLER_COMPOSITION_COMPLETE=1", script)
        self.assertIn("TRAINING_STARTED=0", script)
        self.assertIn("TRAJECTORY_POSTPROCESSING_PERFORMED=0", script)
        self.assertIn("FULL_TRIAL_RENDER=1", script)
        self.assertIn("three_characters_actual_full_trial_composition.png", script)


if __name__ == "__main__":
    unittest.main()
