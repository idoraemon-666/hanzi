from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hanzi_writing.fixed_duration_diagnostic import (
    fixed_movement_metrics,
    load_fixed_duration_diagnostic_config,
)
from hanzi_writing.geometry import build_component_trajectory, cue_scale, load_geometry_config
from hanzi_writing.fixed_duration_protocol import exact_conditions


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_fixed_duration_diagnostic_v1.json"
)
GEOMETRY = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_fixed_duration_geometry.json"
)


class FixedDurationDiagnosticTests(unittest.TestCase):
    def test_frozen_diagnostic_config_is_accepted(self):
        config = load_fixed_duration_diagnostic_config(CONFIG)
        self.assertEqual(config["diagnostic_contract"]["metric_rows"], 348)
        self.assertEqual(config["diagnostic_contract"]["trajectory_rows"], 35268)
        self.assertFalse(config["diagnostic_contract"]["complete_character_rollout"])

    def test_diagnostic_contract_cannot_define_behavioral_success(self):
        with CONFIG.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        changed = copy.deepcopy(config)
        changed["diagnostic_contract"]["behavioral_pass_fail"] = "endpoint_lt_1mm"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract"):
                load_fixed_duration_diagnostic_config(path)

    def test_exact_ordinary_trace_has_zero_error_and_unit_spatial_ratio(self):
        geometry = load_geometry_config(GEOMETRY)
        condition = next(
            item
            for item in exact_conditions(geometry)
            if item.condition_id == "primitive_00_heng_start_00_exact"
        )
        trajectory = build_component_trajectory(
            condition, "medium", cue_scale(geometry), geometry
        )
        metrics = fixed_movement_metrics(
            trajectory.points_m,
            trajectory.points_m,
            trajectory,
            None,
            geometry.dt_seconds,
        )
        self.assertEqual(metrics["spatial_movement_mean_euclidean_m"], 0.0)
        self.assertEqual(metrics["total_movement_mean_euclidean_m"], 0.0)
        self.assertEqual(metrics["spatial_path_length_ratio"], 1.0)
        self.assertEqual(metrics["endpoint_euclidean_m"], 0.0)

    def test_nonfinite_movement_trace_is_rejected(self):
        geometry = load_geometry_config(GEOMETRY)
        condition = exact_conditions(geometry)[0]
        trajectory = build_component_trajectory(
            condition, "fast", cue_scale(geometry), geometry
        )
        actual = np.asarray(trajectory.points_m).copy()
        actual[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            fixed_movement_metrics(
                actual,
                trajectory.points_m,
                trajectory,
                None,
                geometry.dt_seconds,
            )


if __name__ == "__main__":
    unittest.main()
