from __future__ import annotations

import unittest

import motornet as mn
import numpy as np
import torch

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.envs import HanziCharacterEnv, HanziComponentEnv
from hanzi_writing.geometry import (
    checkpoint_stroke_groups,
    cue_scale,
    load_geometry_config,
    move_conditions,
)


GEOMETRY_CONFIG = "configurations/hanzi_stroke_temporal_composition_geometry.json"


def make_effector():
    return mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())


class HanziEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_geometry_config(GEOMETRY_CONFIG)

    def test_isolated_stroke_keeps_28d_contract_and_zero_spatial_cue(self):
        condition = checkpoint_stroke_groups(self.config)[0][0]
        env = HanziComponentEnv(
            effector=make_effector(),
            geometry_config_path=GEOMETRY_CONFIG,
            action_frame_stacking=0,
        )
        obs, _ = env.reset(
            options={
                "conditions": (condition,),
                "speed_name": "medium",
                "delay_steps": 50,
                "deterministic": True,
            }
        )
        self.assertEqual(tuple(obs.shape), (1, 28))
        movement_start, movement_end = env.epoch_bounds["movement"]
        hold_start, hold_end = env.epoch_bounds["hold"]
        self.assertEqual(int(torch.argmax(env.rule_input[0, 0]).item()), authority.RULE_INDEX[condition.rule])
        self.assertTrue(bool(torch.all(env.rule_input[:, :, 9] == 0.0)))
        self.assertTrue(bool(torch.all(env.vis_inp == 0.0)))
        self.assertTrue(bool(torch.all(env.go_cue[:, :movement_start] == 0.0)))
        self.assertTrue(bool(torch.all(env.go_cue[:, movement_start:movement_end] == 1.0)))
        self.assertTrue(bool(torch.all(env.go_cue[:, hold_start:hold_end] == 0.0)))
        self.assertEqual(movement_end - movement_start, env.movement_intervals + 1)

    def test_space_construction_does_not_require_a_trial_condition(self):
        component = HanziComponentEnv(
            effector=make_effector(),
            geometry_config_path=GEOMETRY_CONFIG,
            action_frame_stacking=0,
        )
        character = HanziCharacterEnv(
            effector=make_effector(),
            geometry_config_path=GEOMETRY_CONFIG,
            action_frame_stacking=0,
        )
        self.assertEqual(component.observation_space.shape, (28,))
        self.assertEqual(character.observation_space.shape, (28,))
        self.assertEqual(component.action_space.shape, (6,))
        self.assertEqual(character.action_space.shape, (6,))
        self.assertFalse(hasattr(component, "conditions"))
        self.assertFalse(hasattr(character, "character_name"))

    def test_move_goal_cue_is_present_only_during_delay_and_movement(self):
        condition = move_conditions(self.config, include_jitter=False)[0]
        env = HanziComponentEnv(
            effector=make_effector(),
            geometry_config_path=GEOMETRY_CONFIG,
            action_frame_stacking=0,
        )
        env.reset(
            options={
                "conditions": (condition,),
                "speed_name": "medium",
                "delay_steps": 50,
                "deterministic": True,
            }
        )
        stable_end = env.epoch_bounds["stable"][1]
        movement_end = env.epoch_bounds["movement"][1]
        actual = env.vis_inp[0, stable_end:movement_end].cpu()
        expected = np.broadcast_to(
            condition.goal_xy_m / cue_scale(self.config), tuple(actual.shape)
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_allclose(env.vis_inp[0, :stable_end].cpu(), 0.0)
        np.testing.assert_allclose(env.vis_inp[0, movement_end:].cpu(), 0.0)
        self.assertEqual(int(torch.argmax(env.rule_input[0, 0]).item()), authority.RULE_INDEX["move"])

    def test_complete_character_is_validation_only_and_uses_dynamic_rules(self):
        env = HanziCharacterEnv(
            effector=make_effector(),
            geometry_config_path=GEOMETRY_CONFIG,
            action_frame_stacking=0,
        )
        with self.assertRaises(ValueError):
            env.reset(testing=False, options={"character": "mu"})
        obs, _ = env.reset(
            testing=True,
            options={"character": "ke", "speed_name": "medium", "deterministic": True},
        )
        self.assertEqual(tuple(obs.shape), (1, 28))
        self.assertGreater(len(torch.unique(torch.argmax(env.rule_input[0], dim=1))), 1)
        self.assertTrue(bool(torch.all(env.rule_input[:, :, 9] == 0.0)))
        self.assertTrue(bool(torch.all(env.go_cue[:, -self.config.hold_steps :] == 0.0)))
        move_mask = torch.argmax(env.rule_input[0], dim=1) == authority.RULE_INDEX["move"]
        self.assertFalse(bool(torch.any(env.writing_mask & move_mask)))


if __name__ == "__main__":
    unittest.main()
