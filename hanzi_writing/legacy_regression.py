"""Build the new code's legacy candidate and compare it with old-HEAD evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.geometry import (
    SPEED_NAMES,
    build_component_trajectory,
    checkpoint_stroke_groups,
    cue_scale,
    load_geometry_config,
    move_conditions,
    stroke_conditions,
)
from hanzi_writing.training import _condition_manifest


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def collect_legacy_candidate_arrays(
    geometry_path: str | Path,
) -> dict[str, np.ndarray]:
    geometry = load_geometry_config(geometry_path)
    characters, scale = authority.physical_characters(geometry.target_long_medium_steps)
    arrays: dict[str, np.ndarray] = {
        "authority/global_scale_m_per_design_unit": np.asarray(scale, dtype=np.float64),
        "authority/dt_seconds": np.asarray(authority.DT_S, dtype=np.float64),
    }
    for character_name, character in characters.items():
        for stroke_index, stroke in enumerate(character.strokes):
            arrays[f"geometry/{character_name}/stroke_{stroke_index:02d}/points_m"] = np.asarray(
                stroke.points, dtype=np.float64
            )

    strokes = stroke_conditions(geometry, include_jitter=True)
    moves = move_conditions(geometry, include_jitter=True)
    arrays["conditions/stroke_ids"] = np.asarray([row.condition_id for row in strokes])
    arrays["conditions/move_ids"] = np.asarray([row.condition_id for row in moves])
    normalizer = cue_scale(geometry)
    for category, conditions in (("stroke", strokes), ("move", moves)):
        for condition in conditions:
            for speed_name in SPEED_NAMES:
                trajectory = build_component_trajectory(
                    condition, speed_name, normalizer, geometry
                )
                prefix = f"trajectory/{category}/{condition.condition_id}/{speed_name}"
                arrays[f"{prefix}/target_xy_m"] = np.asarray(
                    trajectory.points_m, dtype=np.float64
                )
                arrays[f"{prefix}/movement_intervals"] = np.asarray(
                    trajectory.movement_intervals, dtype=np.int64
                )
                arrays[f"{prefix}/movement_samples"] = np.asarray(
                    len(trajectory.points_m), dtype=np.int64
                )

    groups = [
        [condition.condition_id for condition in group]
        for group in checkpoint_stroke_groups(geometry)
    ]
    validation_groups = [
        {"kind": "stroke", "speed": speed, "condition_ids": group}
        for speed in SPEED_NAMES
        for group in groups
    ] + [
        {"kind": "move", "speed": speed, "condition_ids": [condition.condition_id]}
        for speed in SPEED_NAMES
        for condition in move_conditions(geometry, include_jitter=False)
    ]
    arrays["manifest/training_condition_manifest_json"] = _canonical_json(
        _condition_manifest(geometry)
    )
    arrays["manifest/checkpoint_validation_groups_json"] = _canonical_json(
        validation_groups
    )

    for character_name, character in characters.items():
        schedule = authority.assemble_character_schedule(
            character,
            speed_name="medium",
            stable_steps=geometry.stable_steps,
            prepare_steps=geometry.prepare_steps,
            final_hold_steps=geometry.hold_steps,
            cue_normalizer_m=normalizer,
        )
        prefix = f"schedule/{character_name}"
        for source_key, target_key in (
            ("target_xy_m", "target_xy_m"),
            ("rule_input", "rule_input"),
            ("speed_scalar", "speed_cue"),
            ("go_cue", "go_cue"),
            ("spatial_goal_cue", "spatial_cue"),
            ("writing_mask", "writing_mask"),
            ("phase", "phase"),
        ):
            arrays[f"{prefix}/{target_key}"] = np.asarray(schedule[source_key])
        arrays[f"{prefix}/time_index"] = np.arange(
            len(schedule["target_xy_m"]), dtype=np.int64
        )
        arrays[f"{prefix}/segments_json"] = _canonical_json(schedule["segments"])
    return arrays


def compare_legacy_reference(
    reference_json: str | Path,
    reference_npz: str | Path,
    legacy_geometry_path: str | Path,
) -> dict[str, Any]:
    with Path(reference_json).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    candidate = collect_legacy_candidate_arrays(legacy_geometry_path)
    mismatches = []
    max_abs_difference = 0.0
    with np.load(reference_npz, allow_pickle=False) as reference:
        reference_keys = set(reference.files)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            mismatches.append(
                {
                    "key_set": {
                        "missing": sorted(reference_keys - candidate_keys),
                        "extra": sorted(candidate_keys - reference_keys),
                    }
                }
            )
        for key in sorted(reference_keys & candidate_keys):
            expected = reference[key]
            actual = candidate[key]
            expected_manifest = metadata["arrays"][key]
            if _array_sha256(expected) != expected_manifest["sha256"]:
                raise RuntimeError(f"legacy reference array manifest failed: {key}")
            equal = np.array_equal(expected, actual)
            if not equal:
                difference = None
                if np.issubdtype(expected.dtype, np.number) and expected.shape == actual.shape:
                    difference = float(
                        np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
                    )
                    max_abs_difference = max(max_abs_difference, difference)
                mismatches.append({"key": key, "max_abs_difference": difference})
    return {
        "legacy_head": metadata["legacy_head"],
        "submodule_head": metadata["submodule_head"],
        "reference_array_count": len(metadata["arrays"]),
        "candidate_array_count": len(candidate),
        "all_arrays_np_array_equal": not mismatches,
        "max_abs_difference": max_abs_difference,
        "mismatches": mismatches,
        "passed": not mismatches and max_abs_difference == 0.0,
    }
