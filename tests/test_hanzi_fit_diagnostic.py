from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hanzi_writing.fit_diagnostic import (
    aggregate_metrics,
    load_fit_diagnostic_config,
    movement_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configurations" / "hanzi_stroke_temporal_composition_fit_diagnostic.json"


class FitDiagnosticMetricTests(unittest.TestCase):
    def test_exact_trajectory_has_zero_error_and_unit_path_ratio(self):
        target = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metrics = movement_metrics(target, target)
        self.assertEqual(metrics["target_path_length_m"], 2.0)
        self.assertEqual(metrics["path_length_ratio"], 1.0)
        self.assertEqual(metrics["movement_mean_l1_m"], 0.0)
        self.assertEqual(metrics["endpoint_euclidean_per_target_path_length"], 0.0)

    def test_stationary_actual_has_zero_path_ratio_and_normalized_endpoint_error(self):
        target = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        actual = np.zeros_like(target)
        metrics = movement_metrics(actual, target)
        self.assertEqual(metrics["path_length_ratio"], 0.0)
        self.assertEqual(metrics["path_length_ratio_absolute_deviation"], 1.0)
        self.assertEqual(metrics["endpoint_euclidean_per_target_path_length"], 1.0)

    def test_zero_length_target_is_rejected(self):
        target = np.zeros((3, 2))
        with self.assertRaisesRegex(ValueError, "positive"):
            movement_metrics(target, target)


class FitDiagnosticConfigurationTests(unittest.TestCase):
    def test_frozen_configuration_is_accepted(self):
        config = load_fit_diagnostic_config(CONFIG)
        self.assertFalse(config["network_noise"])
        self.assertFalse(config["include_jitter"])
        self.assertEqual(
            config["diagnostic_contract"]["behavioral_pass_fail"], "not_defined"
        )

    def test_unrequested_path_completion_definition_is_rejected(self):
        with CONFIG.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config = copy.deepcopy(config)
        config["diagnostic_contract"]["path_completion_metric"] = "endpoint_projection"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract"):
                load_fit_diagnostic_config(path)


class FitDiagnosticAggregationTests(unittest.TestCase):
    def test_equal_rule_mean_does_not_weight_more_frequent_rules_more_heavily(self):
        rows = []
        for rule_index, rule in enumerate(
            ("heng", "shu", "pie", "na", "dian", "ti", "hengzhe", "shugou", "move")
        ):
            repeats = 5 if rule == "heng" else 1
            for repeat in range(repeats):
                value = float(rule_index)
                row = {
                    "checkpoint": "best",
                    "speed": "medium",
                    "rule": rule,
                    "condition_id": f"{rule}_{repeat}",
                }
                row.update(
                    {
                        name: value
                        for name in (
                            "target_path_length_m",
                            "actual_path_length_m",
                            "path_length_ratio",
                            "path_length_ratio_absolute_deviation",
                            "movement_mean_l1_m",
                            "movement_mean_euclidean_m",
                            "endpoint_l1_m",
                            "endpoint_euclidean_m",
                            "movement_mean_l1_per_target_path_length",
                            "movement_mean_euclidean_per_target_path_length",
                            "endpoint_l1_per_target_path_length",
                            "endpoint_euclidean_per_target_path_length",
                        )
                    }
                )
                rows.append(row)
        summary = aggregate_metrics(rows)[0]
        self.assertEqual(summary["equal_rule_mean"]["movement_mean_l1_m"], 4.0)
        self.assertLess(summary["condition_weighted_mean"]["movement_mean_l1_m"], 4.0)


if __name__ == "__main__":
    unittest.main()
