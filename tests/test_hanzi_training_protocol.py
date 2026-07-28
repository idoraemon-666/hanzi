from __future__ import annotations

import json
import random
import unittest

from config import load_protocol_config
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    SPEED_NAMES,
    build_component_trajectory,
    cue_scale,
    load_geometry_config,
    training_conditions_by_rule,
)
from hanzi_writing.training import (
    _condition_manifest,
    _duration_compatibility_index,
    _sample_compatible_condition_batch,
    validate_training_config,
)


TRAIN_CONFIG = "configurations/hanzi_stroke_temporal_composition_train_dev42.json"


class HanziTrainingProtocolTests(unittest.TestCase):
    def test_config_is_cpu_project2_baseline_and_dispatchable(self):
        config = load_protocol_config(TRAIN_CONFIG)
        geometry = validate_training_config(config)
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["model"]["hidden_size"], 256)
        self.assertEqual(config["training"]["max_updates"], 75000)
        self.assertEqual(config["position_loss"]["movement_weight"], 0.6)
        self.assertEqual(geometry.validation_seed, 1042)

    def test_rule_first_sampler_and_checkpoint_grid_are_explicit(self):
        with open(TRAIN_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        geometry = load_geometry_config(config["geometry_config"])
        grouped = training_conditions_by_rule(geometry)
        manifest = _condition_manifest(geometry)
        self.assertEqual(tuple(grouped), ACTIVE_RULES)
        self.assertEqual(
            manifest["sampler"],
            "uniform_rule_speed_delay_reference_condition_then_compatible_batch",
        )
        self.assertFalse(manifest["batch_condition_sampling"]["direction_sampling"])
        self.assertEqual(manifest["checkpoint_validation"]["rollout_group_count"], 81)
        self.assertFalse(manifest["checkpoint_validation"]["includes_jitter"])
        self.assertFalse(manifest["checkpoint_validation"]["includes_complete_characters"])
        self.assertTrue(manifest["checkpoint_validation"]["network_noise"])

    def test_batch_conditions_share_rule_speed_and_movement_duration(self):
        with open(TRAIN_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        geometry = load_geometry_config(config["geometry_config"])
        grouped = training_conditions_by_rule(geometry)
        compatibility = _duration_compatibility_index(geometry, grouped)
        normalizer = cue_scale(geometry)
        rng = random.Random(42)

        for rule in ACTIVE_RULES:
            for speed_name in SPEED_NAMES:
                reference, batch = _sample_compatible_condition_batch(
                    grouped[rule], compatibility[rule][speed_name], 32, rng=rng
                )
                intervals = {
                    build_component_trajectory(condition, speed_name, normalizer).movement_intervals
                    for condition in batch
                }
                sampled_rules = {
                    build_component_trajectory(condition, speed_name, normalizer).rule
                    for condition in batch
                }
                self.assertEqual(sampled_rules, {rule})
                self.assertEqual(len(intervals), 1)
                self.assertIn(
                    reference.condition_id,
                    compatibility[rule][speed_name],
                )

    def test_all_stroke_duration_groups_allow_batch_condition_diversity(self):
        with open(TRAIN_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        geometry = load_geometry_config(config["geometry_config"])
        grouped = training_conditions_by_rule(geometry)
        compatibility = _duration_compatibility_index(geometry, grouped)

        for rule in (value for value in ACTIVE_RULES if value != "move"):
            for speed_name in SPEED_NAMES:
                group_sizes = {
                    len(group) for group in compatibility[rule][speed_name].values()
                }
                self.assertGreater(min(group_sizes), 1)

    def test_reference_group_sampling_preserves_uniform_condition_marginals(self):
        with open(TRAIN_CONFIG, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        geometry = load_geometry_config(config["geometry_config"])
        grouped = training_conditions_by_rule(geometry)
        compatibility = _duration_compatibility_index(geometry, grouped)

        for rule, conditions in grouped.items():
            expected = 1.0 / len(conditions)
            for speed_name in SPEED_NAMES:
                for condition in conditions:
                    group = compatibility[rule][speed_name][condition.condition_id]
                    marginal = (len(group) / len(conditions)) * (1.0 / len(group))
                    self.assertAlmostEqual(marginal, expected)


if __name__ == "__main__":
    unittest.main()
