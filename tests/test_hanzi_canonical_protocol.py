from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.canonical_overfit import (
    load_canonical_overfit_config,
    validate_frozen_shared_config,
)
from hanzi_writing.canonical_protocol import (
    EXPECTED_TARGET_ROWS,
    build_canonical_manifest,
    canonical_occurrence_manifest,
    canonical_stroke_conditions,
    canonical_target_trajectory,
    write_stage0_artifacts,
)
from hanzi_writing.geometry import (
    CANONICAL_GEOMETRY_VARIANT,
    CANONICAL_TIMING_MODE,
    build_component_trajectory,
    cue_scale,
    load_geometry_config,
    move_conditions,
)


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_geometry.json"
)
OVERFIT_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json"
)
SHARED_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_canonical_shared_9task_v1.json"
)
FIXED_DURATION_GEOMETRY_PATH = (
    ROOT
    / "configurations"
    / "hanzi_stroke_temporal_composition_fixed_duration_geometry.json"
)


class CanonicalProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = load_geometry_config(GEOMETRY_PATH)

    def test_strict_identity_and_frozen_shared_stage(self) -> None:
        self.assertEqual(self.geometry.timing_mode, CANONICAL_TIMING_MODE)
        self.assertEqual(self.geometry.geometry_variant, CANONICAL_GEOMETRY_VARIANT)
        self.assertEqual(self.geometry.delay_steps, (25, 50, 75))
        self.assertEqual(self.geometry.validation_delay_steps, 50)
        config, geometry = load_canonical_overfit_config(OVERFIT_PATH)
        self.assertTrue(config["enabled"])
        self.assertEqual(config["training"]["delay_steps"], [25, 50, 75])
        self.assertEqual(geometry, self.geometry)
        shared = validate_frozen_shared_config(SHARED_PATH)
        self.assertFalse(shared["enabled"])

    def test_canonical_occurrence_selection_is_frozen(self) -> None:
        selected = canonical_occurrence_manifest(self.geometry)
        actual = {
            row["rule"]: (
                row["source_character"],
                row["source_stroke_index"],
                row["original_condition_id"],
            )
            for row in selected
        }
        self.assertEqual(
            actual,
            {
                "heng": ("ke", 0, "primitive_10_heng_start_00_exact"),
                "shu": ("mu", 1, "primitive_01_shu_start_01_exact"),
                "pie": ("mu", 2, "primitive_02_pie_start_01_exact"),
                "na": ("mu", 3, "primitive_03_na_start_01_exact"),
                "dian": ("jiang", 0, "primitive_04_dian_start_00_exact"),
                "ti": ("jiang", 2, "primitive_06_ti_start_01_exact"),
                "hengzhe": ("ke", 2, "primitive_12_hengzhe_start_01_exact"),
                "shugou": ("ke", 4, "primitive_14_shugou_start_01_exact"),
            },
        )
        characters, _ = authority.physical_characters(
            self.geometry.target_long_medium_steps
        )
        occurrences = authority.primitive_occurrences(characters)
        for row in selected:
            maximum = max(
                float(value["length_m"])
                for value in occurrences
                if value["rule"] == row["rule"]
            )
            self.assertEqual(row["original_path_length_m"], maximum)

    def test_targets_have_frozen_counts_and_no_zero_intervals(self) -> None:
        strokes = canonical_stroke_conditions(self.geometry)
        moves = move_conditions(self.geometry, include_jitter=False)
        self.assertEqual(len(strokes), 8)
        self.assertEqual(len(moves), 12)
        for condition in (*strokes, *moves):
            target = canonical_target_trajectory(condition, self.geometry)
            expected = 201 if getattr(condition, "rule", None) in {"hengzhe", "shugou"} else 151
            self.assertEqual(len(target["points_m"]), expected)
            self.assertTrue(np.isfinite(target["points_m"]).all())
            self.assertTrue(
                np.all(np.linalg.norm(np.diff(target["points_m"], axis=0), axis=1) > 0.0)
            )

    def test_simple_targets_equal_resampled_authority_occurrences(self) -> None:
        for condition in canonical_stroke_conditions(self.geometry):
            if condition.rule in {"hengzhe", "shugou"}:
                continue
            source = condition.relative_points_m + condition.start_xy_m
            expected = authority.resample_by_arclength(source, 150)
            target = canonical_target_trajectory(condition, self.geometry)
            self.assertTrue(np.array_equal(target["points_m"], expected))

    def test_compound_rounding_is_tangent_and_uses_frozen_allocation(self) -> None:
        manifest = build_canonical_manifest(self.geometry)
        _, expected_scale = authority.physical_characters(
            self.geometry.target_long_medium_steps
        )
        self.assertEqual(
            manifest["physical_scale_m_per_design_unit"], expected_scale
        )
        for row in manifest["stroke_conditions"]:
            self.assertLessEqual(row["start_difference_m"], 1e-12)
            self.assertLessEqual(row["end_difference_m"], 1e-12)
            self.assertGreater(row["canonical_target_path_length_m"], 0.0)
        compounds = {
            row["rule"]: row["rounding"]
            for row in manifest["stroke_conditions"]
            if row["rounding"] is not None
        }
        self.assertEqual(set(compounds), {"hengzhe", "shugou"})
        self.assertEqual(
            (
                compounds["hengzhe"]["n_before"],
                compounds["hengzhe"]["transition_intervals"],
                compounds["hengzhe"]["n_after"],
            ),
            (88, 20, 92),
        )
        self.assertEqual(
            (
                compounds["shugou"]["n_before"],
                compounds["shugou"]["transition_intervals"],
                compounds["shugou"]["n_after"],
            ),
            (158, 20, 22),
        )
        for rounding in compounds.values():
            tangent = rounding["analytic_tangency"]
            self.assertAlmostEqual(tangent["start_dot"], 1.0, places=12)
            self.assertAlmostEqual(tangent["end_dot"], 1.0, places=12)
            self.assertAlmostEqual(tangent["start_cross"], 0.0, places=12)
            self.assertAlmostEqual(tangent["end_cross"], 0.0, places=12)
            self.assertEqual(rounding["start_difference_m"], 0.0)
            self.assertEqual(rounding["end_difference_m"], 0.0)

    def test_component_trajectory_uses_slow_cue_without_new_dimensions(self) -> None:
        condition = canonical_stroke_conditions(self.geometry)[0]
        trajectory = build_component_trajectory(
            condition, "slow", cue_scale(self.geometry), self.geometry
        )
        self.assertEqual(trajectory.speed_scalar, authority.TRAIN_SPEED_SCALAR["slow"])
        self.assertEqual(trajectory.movement_intervals, 150)
        self.assertTrue(np.array_equal(trajectory.spatial_cue, np.zeros(2)))
        with self.assertRaises(ValueError):
            build_component_trajectory(
                condition, "medium", cue_scale(self.geometry), self.geometry
            )

    def test_stage0_writes_only_three_frozen_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "stage0"
            result = write_stage0_artifacts(self.geometry, output)
            self.assertEqual(result["target_rows"], EXPECTED_TARGET_ROWS)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "canonical_condition_manifest.json",
                    "canonical_target_trajectories.npz",
                    "canonical_target_audit.png",
                },
            )
            with np.load(output / "canonical_target_trajectories.npz") as arrays:
                self.assertEqual(len(arrays["sample_index"]), EXPECTED_TARGET_ROWS)
                self.assertFalse(any(arrays[name].dtype == object for name in arrays.files))
            with self.assertRaises(FileExistsError):
                write_stage0_artifacts(self.geometry, output)

    def test_stage0_artifacts_are_cross_process_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            outputs = [Path(parent) / name for name in ("first", "second")]
            for output in outputs:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "hanzi_writing.canonical_protocol",
                        "--geometry-config",
                        str(GEOMETRY_PATH),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            for name in (
                "canonical_condition_manifest.json",
                "canonical_target_trajectories.npz",
                "canonical_target_audit.png",
            ):
                hashes = [
                    hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for output in outputs
                ]
                self.assertEqual(hashes[0], hashes[1], name)

    def test_canonical_and_old_schemas_do_not_cross_accept(self) -> None:
        raw = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
        raw["timing_mode"] = "fixed_movement_duration"
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "wrong.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_geometry_config(path)
        raw = json.loads(FIXED_DURATION_GEOMETRY_PATH.read_text(encoding="utf-8"))
        raw["timing_mode"] = CANONICAL_TIMING_MODE
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "wrong.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_geometry_config(path)


if __name__ == "__main__":
    unittest.main()
