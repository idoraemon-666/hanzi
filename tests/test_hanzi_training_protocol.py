from __future__ import annotations

import json
import unittest

from config import load_protocol_config
from hanzi_writing.geometry import ACTIVE_RULES, load_geometry_config, training_conditions_by_rule
from hanzi_writing.training import _condition_manifest, validate_training_config


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
        self.assertEqual(manifest["sampler"], "uniform_rule_then_uniform_condition_speed_delay")
        self.assertEqual(manifest["checkpoint_validation"]["rollout_group_count"], 81)
        self.assertFalse(manifest["checkpoint_validation"]["includes_jitter"])
        self.assertFalse(manifest["checkpoint_validation"]["includes_complete_characters"])
        self.assertTrue(manifest["checkpoint_validation"]["network_noise"])


if __name__ == "__main__":
    unittest.main()
