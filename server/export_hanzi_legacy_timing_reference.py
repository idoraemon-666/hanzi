#!/usr/bin/env python3
"""Export timing evidence from an explicitly named untouched legacy worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SPEED_NAMES = ("fast", "medium", "slow")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _import_legacy(repo: Path):
    for name in tuple(sys.modules):
        if name == "hanzi_writing" or name.startswith("hanzi_writing."):
            del sys.modules[name]
    sys.path[:] = [str(repo)] + [entry for entry in sys.path if Path(entry or os.curdir).resolve() != repo]
    os.chdir(repo)
    from hanzi_writing import hanzi_geometry_final as authority
    from hanzi_writing.geometry import (
        build_component_trajectory,
        checkpoint_stroke_groups,
        cue_scale,
        load_geometry_config,
        move_conditions,
        stroke_conditions,
    )
    from hanzi_writing.training import _condition_manifest

    imported = {}
    for name, module in sys.modules.items():
        if name == "hanzi_writing" or name.startswith("hanzi_writing."):
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            resolved = Path(module_file).resolve()
            if repo not in resolved.parents and resolved != repo:
                raise RuntimeError(f"legacy import escaped worktree: {name}={resolved}")
            imported[name] = str(resolved)
    return (
        authority,
        build_component_trajectory,
        checkpoint_stroke_groups,
        cue_scale,
        load_geometry_config,
        move_conditions,
        stroke_conditions,
        _condition_manifest,
        imported,
    )


def collect_arrays(repo: Path, geometry_relative: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    (
        authority,
        build_component_trajectory,
        checkpoint_stroke_groups,
        cue_scale,
        load_geometry_config,
        move_conditions,
        stroke_conditions,
        condition_manifest,
        imported,
    ) = _import_legacy(repo)
    geometry = load_geometry_config(repo / geometry_relative)
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
                trajectory = build_component_trajectory(condition, speed_name, normalizer)
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
        condition_manifest(geometry)
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
    return arrays, imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--geometry-config",
        default="configurations/hanzi_stroke_temporal_composition_geometry.json",
    )
    arguments = parser.parse_args()
    repo = arguments.legacy_repo.resolve()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite legacy reference directory: {output}")
    if _git(repo, "status", "--short"):
        raise RuntimeError("legacy reference requires a clean worktree")
    output.mkdir(parents=True, exist_ok=False)
    arrays, imported = collect_arrays(repo, arguments.geometry_config)
    npz_path = output / "legacy_reference.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        key: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for key, value in sorted(arrays.items())
    }
    geometry_path = repo / arguments.geometry_config
    metadata = {
        "legacy_head": _git(repo, "rev-parse", "HEAD"),
        "submodule_head": _git(repo / "mRNNTorch", "rev-parse", "HEAD"),
        "legacy_worktree": str(repo),
        "legacy_worktree_clean": True,
        "geometry_config": str(geometry_path),
        "geometry_config_sha256": _sha256_file(geometry_path),
        "exporter": str(Path(__file__).resolve()),
        "exporter_sha256": _sha256_file(Path(__file__).resolve()),
        "imported_hanzi_modules": imported,
        "array_count": len(arrays),
        "arrays": manifest,
        "npz_sha256": _sha256_file(npz_path),
        "npz_allow_pickle_required": False,
    }
    with (output / "legacy_reference.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"completed": True, "array_count": len(arrays)}, sort_keys=True))


if __name__ == "__main__":
    main()
