from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hanzi_writing.boundary_intervention import (
    CONDITIONS,
    _new_character_env,
    _sample,
    compose_observation,
    enumerate_boundaries,
    finalize_trace,
    hengzhe_metrics,
    load_boundary_intervention_config,
    movement_metrics,
    plant_sensory_digest,
    restore_plant_sensory,
    snapshot_plant_sensory,
    state_block_changes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_boundary_intervention.json"
)
GEOMETRY = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_geometry.json"
)


class BoundaryInterventionConfigurationTests(unittest.TestCase):
    def test_frozen_configuration_is_accepted(self):
        config = load_boundary_intervention_config(CONFIG)
        self.assertEqual(tuple(config["conditions"]), CONDITIONS)
        self.assertEqual(config["behavioral_pass_fail"], "not_defined")
        self.assertFalse(config["network_noise"])

    def test_posthoc_behavior_threshold_is_rejected(self):
        with CONFIG.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config = copy.deepcopy(config)
        config["behavioral_pass_fail"] = "endpoint_error_below_observed_value"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_boundary_intervention_config(path)


class BoundaryScheduleTests(unittest.TestCase):
    def test_all_twelve_boundaries_are_derived_from_segments(self):
        counts = {}
        identifiers = []
        for character, expected in (("mu", 3), ("jiang", 5), ("ke", 4)):
            env, _ = _new_character_env(GEOMETRY, character, "medium", 1042)
            boundaries = enumerate_boundaries(env)
            counts[character] = len(boundaries)
            identifiers.extend(boundary["boundary_id"] for boundary in boundaries)
            for boundary in boundaries:
                self.assertEqual(
                    boundary["movement_intervals"] + 1,
                    boundary["movement_end_index_exclusive"]
                    - boundary["movement_sample_0_schedule_index"],
                )
                self.assertEqual(
                    boundary["movement_sample_0_schedule_index"]
                    - boundary["prepare_entry_schedule_index"],
                    25,
                )
            self.assertEqual(len(boundaries), expected)
        self.assertEqual(counts, {"mu": 3, "jiang": 5, "ke": 4})
        self.assertEqual(len(set(identifiers)), 12)


class PlantSensoryRestoreTests(unittest.TestCase):
    def test_same_snapshot_observation_and_action_reproduce_next_step(self):
        env, obs = _new_character_env(GEOMETRY, "mu", "medium", 1042)
        self.assertTrue(torch.equal(obs, compose_observation(env, 0)))
        snapshot = snapshot_plant_sensory(env)
        action = torch.full((1, env.action_space.shape[0]), 0.25, dtype=torch.float32)

        first_obs, _, _, _ = env.step(1, action=action)
        first_after = snapshot_plant_sensory(env)
        restore_plant_sensory(env, snapshot)
        second_obs, _, _, _ = env.step(1, action=action)
        second_after = snapshot_plant_sensory(env)

        self.assertTrue(torch.equal(first_obs, second_obs))
        self.assertEqual(
            plant_sensory_digest(first_after), plant_sensory_digest(second_after)
        )

    def test_trace_serializes_variable_action_buffer_manifest_without_object_arrays(self):
        env, _ = _new_character_env(GEOMETRY, "mu", "medium", 1042)
        x = torch.zeros((1, 4), dtype=torch.float32)
        h = torch.zeros_like(x)
        first = _sample(env, 0, x, h, None)
        action = torch.full((1, env.action_space.shape[0]), 0.25, dtype=torch.float32)
        env.step(1, action=action)
        second = _sample(env, 1, x, h, action)

        trace = finalize_trace([first, second], env.geometry_config.dt_seconds, None)
        manifests = trace["plant_sensory_field_manifest_json"]
        self.assertEqual(manifests.shape, (2,))
        self.assertNotEqual(manifests.dtype, object)
        self.assertNotEqual(len(json.loads(manifests[0])), len(json.loads(manifests[1])))


class BoundaryMetricTests(unittest.TestCase):
    def test_movement_sample_zero_is_the_first_of_intervals_plus_one_points(self):
        target = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        actual = target.copy()
        metrics = movement_metrics(actual, target)
        self.assertEqual(len(actual), 2 + 1)
        self.assertEqual(metrics["movement_sample_0_error_m"], 0.0)
        self.assertEqual(metrics["movement_sample_1_error_m"], 0.0)

    def test_state_block_interaction_uses_descriptive_nonadditivity_formula(self):
        rows = {}
        values = {
            "C0_continuous": 10.0,
            "C1_plant_sensory_clean": 7.0,
            "C2_neural_clean": 8.0,
            "C3_joint_clean": 4.0,
            "P1_plant_sensory_clean_before_prepare": 9.0,
        }
        for condition, value in values.items():
            rows[condition] = {
                "target_path_length_m": 2.0,
                "movement_mean_euclidean_m": value,
                "endpoint_euclidean_m": value,
                "movement_sample_0_error_m": value,
                "movement_sample_1_error_m": value,
                "path_length_ratio_absolute_deviation": value,
            }
        changes = state_block_changes(rows)
        result = changes["movement_mean_euclidean_m"]
        self.assertEqual(result["body_change"], 3.0)
        self.assertEqual(result["neural_change"], 2.0)
        self.assertEqual(result["joint_change"], 6.0)
        self.assertEqual(result["interaction"], 1.0)

    def test_hengzhe_metrics_require_turning_not_only_corner_proximity(self):
        target = np.asarray(
            [
                [-3.0, 0.0],
                [-2.0, 0.0],
                [-1.0, 0.0],
                [0.0, 0.0],
                [0.0, -1.0],
                [0.0, -2.0],
                [0.0, -3.0],
            ]
        )
        metrics = hengzhe_metrics(target, target, np.asarray([0.0, 0.0]), 3)
        self.assertEqual(metrics["corner_min_distance_m"], 0.0)
        self.assertEqual(metrics["corner_actual_turn_angle_deg"], 90.0)
        self.assertEqual(metrics["corner_turn_angle_difference_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
