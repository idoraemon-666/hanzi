from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config import load_protocol_config
from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.fixed_duration_preflight import run_coverage
from hanzi_writing.fixed_duration_protocol import (
    EXPECTED_DIAGNOSTIC_METRIC_ROWS,
    EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS,
    HARD_COMPOUND_TIMING,
    build_fixed_condition_manifest,
    compound_movement_metrics,
    corner_reference_mapping,
    exact_conditions,
)
from hanzi_writing.geometry import (
    FIXED_DURATION_TIMING_MODE,
    LEGACY_TIMING_MODE,
    SPEED_NAMES,
    StrokeCondition,
    build_component_trajectory,
    characters,
    cue_scale,
    load_geometry_config,
)
from hanzi_writing.training import readonly_checkpoint_validation, validate_training_config


ROOT = Path(__file__).resolve().parents[1]
LEGACY_GEOMETRY = ROOT / "configurations" / "hanzi_stroke_temporal_composition_geometry.json"
FIXED_GEOMETRY = ROOT / "configurations" / "hanzi_stroke_temporal_composition_fixed_duration_geometry.json"
COVERAGE_CONFIG = ROOT / "configurations" / "hanzi_stroke_temporal_composition_fixed_duration_coverage_v1.json"
SMOKE_CONFIG = ROOT / "configurations" / "hanzi_stroke_temporal_composition_fixed_duration_smoke_v1.json"
PILOT_CONFIG = ROOT / "configurations" / "hanzi_stroke_temporal_composition_fixed_duration_pilot_v1.json"


def independent_resample(points: np.ndarray, intervals: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    lengths = np.sqrt(np.square(np.diff(points, axis=0)).sum(axis=1))
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(lengths)))
    positions = np.linspace(0.0, cumulative[-1], intervals + 1, dtype=np.float64)
    result = np.column_stack(
        [np.interp(positions, cumulative, points[:, dimension]) for dimension in range(2)]
    )
    result[0] = points[0]
    result[-1] = points[-1]
    return result


class FixedDurationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.legacy = load_geometry_config(LEGACY_GEOMETRY)
        self.fixed = load_geometry_config(FIXED_GEOMETRY)
        self.normalizer = cue_scale(self.fixed)

    def test_old_config_defaults_to_legacy_and_new_config_is_explicit(self):
        self.assertEqual(self.legacy.timing_mode, LEGACY_TIMING_MODE)
        self.assertEqual(self.fixed.timing_mode, FIXED_DURATION_TIMING_MODE)
        self.assertEqual(dict(self.fixed.movement_intervals), {"fast": 50, "medium": 100, "slow": 150})
        self.assertEqual(self.fixed.corner_dwell_intervals, 5)
        self.assertEqual(self.fixed.corner_dwell_rules, ("hengzhe", "shugou"))

    def test_fixed_ordinary_components_use_d_plus_one_samples(self):
        condition = next(
            item
            for item in exact_conditions(self.fixed)
            if not (isinstance(item, StrokeCondition) and item.rule in {"hengzhe", "shugou"})
        )
        for speed_name, intervals in (("fast", 50), ("medium", 100), ("slow", 150)):
            trajectory = build_component_trajectory(condition, speed_name, self.normalizer, self.fixed)
            self.assertEqual(trajectory.base_movement_intervals, intervals)
            self.assertEqual(trajectory.movement_intervals, intervals)
            self.assertEqual(len(trajectory.points_m), intervals + 1)
            self.assertEqual(trajectory.corner_dwell_intervals, 0)

    def test_compound_splice_matches_an_independent_oracle_and_hard_table(self):
        conditions = [
            item
            for item in exact_conditions(self.fixed)
            if isinstance(item, StrokeCondition)
            and item.rule in {"hengzhe", "shugou"}
            and item.condition_id.endswith("start_00_exact")
        ]
        self.assertEqual(len(conditions), 2)
        for condition in conditions:
            dense = condition.relative_points_m + condition.start_xy_m
            corner = dense[80]
            first_dense = dense[:81]
            second_dense = dense[80:]
            first_length = np.linalg.norm(np.diff(first_dense, axis=0), axis=1).sum()
            second_length = np.linalg.norm(np.diff(second_dense, axis=0), axis=1).sum()
            for speed_name in SPEED_NAMES:
                base, n1, n2, entry, exit_index, total, samples = HARD_COMPOUND_TIMING[
                    (condition.rule, speed_name)
                ]
                oracle_n1 = int(np.floor(base * first_length / (first_length + second_length) + 0.5))
                self.assertEqual((oracle_n1, base - oracle_n1), (n1, n2))
                first = independent_resample(first_dense, n1)
                second = independent_resample(second_dense, n2)
                expected = np.concatenate(
                    (first, np.repeat(corner[None, :], 5, axis=0), second[1:]), axis=0
                )
                actual = build_component_trajectory(
                    condition, speed_name, self.normalizer, self.fixed
                )
                np.testing.assert_array_equal(actual.points_m, expected)
                self.assertEqual(actual.dwell_entry_index, entry)
                self.assertEqual(actual.dwell_exit_index, exit_index)
                self.assertEqual(actual.movement_intervals, total)
                self.assertEqual(len(actual.points_m), samples)
                np.testing.assert_array_equal(actual.points_m[entry : exit_index + 1], np.repeat(corner[None, :], 6, axis=0))
                np.testing.assert_array_equal(np.diff(actual.points_m[entry : exit_index + 1], axis=0), np.zeros((5, 2)))

    def test_character_schedule_default_remains_legacy_and_fixed_mode_propagates(self):
        character = characters(self.fixed)["ke"]
        legacy_default = authority.assemble_character_schedule(
            character, speed_name="medium", cue_normalizer_m=self.normalizer
        )
        legacy_explicit = authority.assemble_character_schedule(
            character,
            speed_name="medium",
            cue_normalizer_m=self.normalizer,
            timing_mode=LEGACY_TIMING_MODE,
        )
        self.assertEqual(set(legacy_default), set(legacy_explicit))
        for key in legacy_default:
            if isinstance(legacy_default[key], np.ndarray):
                np.testing.assert_array_equal(legacy_default[key], legacy_explicit[key])
            else:
                self.assertEqual(legacy_default[key], legacy_explicit[key])
        fixed = authority.assemble_character_schedule(
            character,
            speed_name="medium",
            cue_normalizer_m=self.normalizer,
            timing_mode=FIXED_DURATION_TIMING_MODE,
            movement_intervals=dict(self.fixed.movement_intervals),
            corner_dwell_intervals=5,
            corner_dwell_rules=("hengzhe", "shugou"),
        )
        compound_segments = [
            row for row in fixed["segments"] if row["rule"] in {"hengzhe", "shugou"} and "movement" in row["phase"]
        ]
        self.assertEqual(len(compound_segments), 2)
        self.assertTrue(all(row["end_index_exclusive"] - row["start_index"] == 106 for row in compound_segments))
        self.assertTrue(all(row["corner_dwell_intervals"] == 5 for row in compound_segments))


class FixedDurationContractTests(unittest.TestCase):
    def test_three_strict_variants_do_not_cross_accept(self):
        coverage = load_protocol_config(COVERAGE_CONFIG)
        smoke = load_protocol_config(SMOKE_CONFIG)
        pilot = load_protocol_config(PILOT_CONFIG)
        validate_training_config(coverage)
        validate_training_config(smoke)
        validate_training_config(pilot)
        wrong = copy.deepcopy(smoke)
        wrong["variant"] = pilot["variant"]
        with self.assertRaises(ValueError):
            validate_training_config(wrong)
        wrong = copy.deepcopy(pilot)
        wrong["training"]["max_updates"] = 100
        with self.assertRaises(ValueError):
            validate_training_config(wrong)
        wrong = copy.deepcopy(coverage)
        wrong["timing_mode"] = LEGACY_TIMING_MODE
        with self.assertRaises(ValueError):
            validate_training_config(wrong)
        legacy = load_protocol_config(
            ROOT / "configurations" / "hanzi_stroke_temporal_composition_train_dev42.json"
        )
        legacy["geometry_config"] = str(FIXED_GEOMETRY)
        with self.assertRaises(ValueError):
            validate_training_config(legacy)

    def test_geometry_schema_rejects_cross_mode_fields(self):
        with LEGACY_GEOMETRY.open("r", encoding="utf-8") as handle:
            legacy = json.load(handle)
        legacy["timing_mode"] = FIXED_DURATION_TIMING_MODE
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_geometry_config(path)

    def test_manifest_freezes_counts_corner_mappings_and_diagnostic_gates(self):
        geometry = load_geometry_config(FIXED_GEOMETRY)
        rows, audit = build_fixed_condition_manifest(geometry)
        self.assertTrue(rows)
        self.assertEqual(audit["atomic_exact_condition_count"], 58)
        self.assertEqual(audit["N_corner"], 4)
        self.assertEqual(audit["future_diagnostic_metric_rows"], EXPECTED_DIAGNOSTIC_METRIC_ROWS)
        self.assertEqual(audit["future_diagnostic_trajectory_rows"], EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS)
        self.assertEqual(len(audit["corner_reference_mappings"]), 6)

    def test_compound_metrics_use_frozen_spatial_and_dwell_indices(self):
        geometry = load_geometry_config(FIXED_GEOMETRY)
        condition = next(
            item
            for item in exact_conditions(geometry)
            if isinstance(item, StrokeCondition)
            and item.rule == "hengzhe"
            and item.condition_id.endswith("start_00_exact")
        )
        trajectory = build_component_trajectory(
            condition, "medium", cue_scale(geometry), geometry
        )
        mapping = corner_reference_mapping(trajectory)
        metrics = compound_movement_metrics(
            trajectory.points_m,
            trajectory.points_m,
            trajectory,
            mapping,
            geometry.dt_seconds,
        )
        self.assertEqual(metrics["spatial_movement_mean_euclidean_m"], 0.0)
        self.assertEqual(metrics["total_movement_mean_euclidean_m"], 0.0)
        self.assertEqual(metrics["dwell_path_length_m"], 0.0)
        self.assertEqual(metrics["dwell_max_speed_mps"], 0.0)
        self.assertAlmostEqual(metrics["target_corner_angle_deg"], -100.0)

    def test_existing_output_directory_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_coverage(COVERAGE_CONFIG, directory)

    def test_readonly_validation_restores_all_visible_state_and_repeats(self):
        policy = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        policy(torch.ones(1, 2)).sum().backward()

        class GeneratorHolder:
            def __init__(self):
                self.generator = np.random.default_rng(42)

        validation_value = {
            "aggregate": {"phase_normalized_position_l1": 1.0},
            "rollout_group_count_by_rule": {rule: 9 for rule in authority.RULE_INDEX},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.bin"
            checkpoint.write_bytes(b"unchanged")
            with patch(
                "hanzi_writing.training.checkpoint_validation",
                return_value=validation_value,
            ):
                first = readonly_checkpoint_validation(
                    policy,
                    optimizer,
                    {},
                    None,
                    training_env=GeneratorHolder(),
                    checkpoint_paths=(checkpoint,),
                )
                second = readonly_checkpoint_validation(
                    policy,
                    optimizer,
                    {},
                    None,
                    training_env=GeneratorHolder(),
                    checkpoint_paths=(checkpoint,),
                )
        self.assertEqual(first["validation"], second["validation"])
        self.assertTrue(all(value for value in first["read_only_checks"].values() if isinstance(value, bool)))


if __name__ == "__main__":
    unittest.main()
