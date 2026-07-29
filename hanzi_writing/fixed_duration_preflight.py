"""Server-only gates for the fixed-duration nine-task pilot implementation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover - the accepted runner is Linux only
    resource = None

from hanzi_writing.audit import _workspace_audit
from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.fixed_duration_protocol import (
    EXPECTED_DIAGNOSTIC_METRIC_ROWS,
    EXPECTED_DIAGNOSTIC_PLOTS,
    EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS,
    build_fixed_condition_manifest,
    condition_rule,
    exact_conditions,
    validate_fixed_trajectory,
    write_condition_manifest,
)
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    SPEED_NAMES,
    build_component_trajectory,
    cue_scale,
    load_geometry_config,
    training_conditions_by_rule,
)
from hanzi_writing.legacy_regression import compare_legacy_reference
from hanzi_writing.training import (
    _duration_compatibility_index,
    _hp_from_config,
    _make_effector,
    _nested_equal,
    _rollout,
    _sample_compatible_condition_batch,
    readonly_checkpoint_validation,
    validate_training_config,
)
from losses import l1_muscle_act, l1_rate, l1_weight, position_l1_metrics, simple_dynamics
from train import _build_policy


WARMUP_UPDATES = 10


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _require_absent(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    return output


def _peak_rss_mb() -> float:
    if resource is None:
        raise RuntimeError("fixed-duration preflight requires Linux resource accounting")
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _loss(policy, hp: dict[str, Any], result: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    position = position_l1_metrics(result["xy"], result["target"], result["epoch_bounds"])
    components = {
        "position": position["phase_normalized_position_l1"],
        "rate": l1_rate(result["hidden"], hp["l1_rate"]),
        "weight": l1_weight(policy, hp["l1_weight"]),
        "muscle": l1_muscle_act(result["muscle"], hp["l1_muscle_act"]),
        "simple_dynamics": simple_dynamics(
            result["hidden"], policy.mrnn, weight=hp["simple_dynamics_weight"]
        ),
    }
    total = sum(components.values())
    return total, {
        name: float(value.detach().cpu()) for name, value in components.items()
    }


def _gradients_finite(policy) -> tuple[bool, int]:
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    return bool(gradients and all(torch.isfinite(value).all() for value in gradients)), len(gradients)


def _assert_run_kind(config: dict[str, Any], expected: str) -> None:
    if config["run_kind"] != expected:
        raise ValueError(f"expected run_kind={expected}")


def run_target_audit(
    config_path: str | Path,
    legacy_geometry_path: str | Path,
    legacy_json: str | Path,
    legacy_npz: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = _require_absent(output_dir)
    config = _load_json(config_path)
    _assert_run_kind(config, "coverage_test")
    geometry = validate_training_config(config)
    output = _new_directory(output)
    legacy_geometry = load_geometry_config(legacy_geometry_path)
    fixed_characters, fixed_scale = authority.physical_characters(
        geometry.target_long_medium_steps
    )
    legacy_characters, legacy_scale = authority.physical_characters(
        legacy_geometry.target_long_medium_steps
    )
    geometry_equal = fixed_scale == legacy_scale
    geometry_max_abs_difference = 0.0
    for name in fixed_characters:
        for fixed_stroke, legacy_stroke in zip(
            fixed_characters[name].strokes, legacy_characters[name].strokes
        ):
            geometry_equal = geometry_equal and np.array_equal(
                fixed_stroke.points, legacy_stroke.points
            )
            geometry_max_abs_difference = max(
                geometry_max_abs_difference,
                float(np.max(np.abs(fixed_stroke.points - legacy_stroke.points))),
            )
    if not geometry_equal or geometry_max_abs_difference != 0.0:
        raise RuntimeError("fixed and legacy physical geometry are not elementwise identical")
    legacy = compare_legacy_reference(legacy_json, legacy_npz, legacy_geometry_path)
    if not legacy["passed"]:
        raise RuntimeError("new code failed the old-HEAD legacy reference comparison")
    rows, manifest_audit = build_fixed_condition_manifest(geometry)
    manifest_path = output / "fixed_duration_condition_manifest.csv"
    write_condition_manifest(manifest_path, rows)
    workspace = _workspace_audit(geometry)
    if not workspace["passed"]:
        raise RuntimeError("existing workspace audit failed under fixed-duration geometry")
    result = {
        "passed": True,
        "legacy_regression": legacy,
        "physical_geometry_np_array_equal": geometry_equal,
        "physical_geometry_max_abs_difference": geometry_max_abs_difference,
        "global_scale_m_per_design_unit": fixed_scale,
        "fixed_duration_manifest": manifest_audit,
        "workspace_audit": {
            "passed": workspace["passed"],
            "minimum_inner_radial_margin_m": workspace["minimum_inner_radial_margin_m"],
            "minimum_outer_radial_margin_m": workspace["minimum_outer_radial_margin_m"],
            "minimum_joint_margin_rad": workspace["minimum_joint_margin_rad"],
            "maximum_motor_fk_error_m": workspace["maximum_motor_fk_error_m"],
        },
    }
    _write_json(output / "target_audit.json", result)
    return result


def run_coverage(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = _require_absent(output_dir)
    config = _load_json(config_path)
    _assert_run_kind(config, "coverage_test")
    geometry = validate_training_config(config)
    output = _new_directory(output)
    hp = _hp_from_config(config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    policy = _build_policy(hp, config["model"]["output_size"], torch.device("cpu"))
    from hanzi_writing.envs import HanziComponentEnv

    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    conditions = training_conditions_by_rule(geometry)
    normalizer = cue_scale(geometry)
    cases = []
    for rule in ACTIVE_RULES:
        condition = conditions[rule][0]
        for speed_name in SPEED_NAMES:
            trajectory = build_component_trajectory(
                condition, speed_name, normalizer, geometry
            )
            target_audit = validate_fixed_trajectory(trajectory)
            batch = (condition,) * hp["batch_size"]
            policy.zero_grad(set_to_none=True)
            result = _rollout(
                policy,
                env,
                hp,
                batch,
                speed_name,
                geometry.validation_delay_steps,
                network_noise=False,
                deterministic_observation=True,
                track_gradients=True,
            )
            loss, components = _loss(policy, hp, result)
            if not torch.isfinite(loss):
                raise RuntimeError(f"coverage loss is not finite: {rule}/{speed_name}")
            loss.backward()
            gradient_finite, gradient_count = _gradients_finite(policy)
            if not gradient_finite:
                raise RuntimeError(f"coverage gradients are not finite: {rule}/{speed_name}")
            movement_start, movement_end = result["epoch_bounds"]["movement"]
            if movement_end - movement_start != trajectory.movement_intervals + 1:
                raise RuntimeError("coverage environment target length is off by one")
            cases.append(
                {
                    "rule": rule,
                    "duration_condition": speed_name,
                    "condition_id": condition.condition_id,
                    "batch_size": hp["batch_size"],
                    "movement_samples": movement_end - movement_start,
                    "loss": float(loss.detach().cpu()),
                    "loss_components": components,
                    "gradient_tensor_count": gradient_count,
                    "gradient_finite": gradient_finite,
                    **target_audit,
                }
            )
    policy.zero_grad(set_to_none=True)
    if len(cases) != 27 or {
        (row["rule"], row["duration_condition"]) for row in cases
    } != {(rule, speed) for rule in ACTIVE_RULES for speed in SPEED_NAMES}:
        raise RuntimeError("deterministic coverage must contain all 27 rule-duration cases")
    result = {
        "passed": True,
        "case_count": len(cases),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "training_checkpoint_written": False,
        "cases": cases,
        "peak_rss_mb": _peak_rss_mb(),
    }
    _write_json(output / "coverage_result.json", result)
    return result


def run_smoke(
    config_path: str | Path, output_dir: str | Path, state_path: str | Path
) -> dict[str, Any]:
    output = _require_absent(output_dir)
    state_file = Path(state_path)
    if state_file.exists():
        raise FileExistsError(f"refusing to overwrite technical smoke state: {state_file}")
    config = _load_json(config_path)
    _assert_run_kind(config, "technical_smoke")
    geometry = validate_training_config(config)
    output = _new_directory(output)
    hp = _hp_from_config(config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    policy = _build_policy(hp, config["model"]["output_size"], torch.device("cpu"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    from hanzi_writing.envs import HanziComponentEnv

    env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    conditions = training_conditions_by_rule(geometry)
    compatibility = _duration_compatibility_index(geometry, conditions)
    timings = []
    sample_counts: Counter[str] = Counter()
    rule_duration_counts: Counter[str] = Counter()
    longest_sequence = 0
    all_losses_finite = True
    all_gradients_finite = True
    started = time.perf_counter()
    for update in range(config["training"]["max_updates"]):
        update_started = time.perf_counter()
        rule = random.choice(ACTIVE_RULES)
        speed_name = random.choice(SPEED_NAMES)
        delay_steps = random.choice(geometry.delay_steps)
        _, batch = _sample_compatible_condition_batch(
            conditions[rule], compatibility[rule][speed_name], hp["batch_size"]
        )
        result = _rollout(
            policy,
            env,
            hp,
            batch,
            speed_name,
            delay_steps,
            network_noise=True,
            deterministic_observation=False,
            track_gradients=True,
        )
        loss, _ = _loss(policy, hp, result)
        optimizer.zero_grad()
        loss_finite = bool(torch.isfinite(loss))
        if not loss_finite:
            raise RuntimeError(f"technical smoke loss is not finite at update {update}")
        loss.backward()
        gradient_finite, _ = _gradients_finite(policy)
        if not gradient_finite:
            raise RuntimeError(f"technical smoke gradient is not finite at update {update}")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), hp["grad_clip_norm"])
        optimizer.step()
        timings.append(time.perf_counter() - update_started)
        sample_counts[speed_name] += 1
        rule_duration_counts[f"{rule}:{speed_name}"] += 1
        longest_sequence = max(longest_sequence, int(result["timesteps"]))
        all_losses_finite = all_losses_finite and loss_finite
        all_gradients_finite = all_gradients_finite and gradient_finite
    total_time = time.perf_counter() - started
    post_warmup = timings[WARMUP_UPDATES:]
    if not post_warmup:
        raise RuntimeError("technical smoke has no post-warmup timing samples")
    if set(sample_counts) != set(SPEED_NAMES):
        raise RuntimeError("technical smoke did not sample all three duration conditions")
    sorted_times = sorted(post_warmup)
    p95_index = int(math.ceil(0.95 * len(sorted_times))) - 1
    state_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "technical_only": True,
            "not_valid_for_pilot_initialization": True,
            "config": config,
            "hp": hp,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "gradients": {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in policy.named_parameters()
            },
            "completed_update": 99,
        },
        state_file,
    )
    result = {
        "passed": all_losses_finite and all_gradients_finite,
        "updates": len(timings),
        "warmup_updates": WARMUP_UPDATES,
        "post_warmup_update_wall_time_median_s": statistics.median(post_warmup),
        "post_warmup_update_wall_time_p95_s": sorted_times[p95_index],
        "total_100_update_wall_time_s": total_time,
        "peak_rss_mb": _peak_rss_mb(),
        "duration_sample_counts": dict(sample_counts),
        "rule_duration_sample_counts": dict(rule_duration_counts),
        "all_losses_finite": all_losses_finite,
        "all_gradients_finite": all_gradients_finite,
        "longest_episode_timesteps": longest_sequence,
        "oom_observed": False,
        "smoke_state_is_technical_only": True,
        "smoke_state_must_not_initialize_pilot": True,
        "final_validation_update": None,
    }
    _write_json(output / "smoke_result.json", result)
    return result


def run_validation_benchmark(
    config_path: str | Path,
    state_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = _require_absent(output_dir)
    config = _load_json(config_path)
    _assert_run_kind(config, "technical_smoke")
    geometry = validate_training_config(config)
    output = _new_directory(output)
    state_file = Path(state_path)
    state_hash_before = hashlib.sha256(state_file.read_bytes()).hexdigest()
    state = torch.load(state_file, map_location=torch.device("cpu"), weights_only=False)
    if state.get("technical_only") is not True:
        raise ValueError("validation benchmark requires the technical smoke state")
    hp = state["hp"]
    policy = _build_policy(hp, config["model"]["output_size"], torch.device("cpu"))
    policy.load_state_dict(state["policy_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for name, parameter in policy.named_parameters():
        gradient = state["gradients"][name]
        parameter.grad = None if gradient is None else gradient.detach().clone()
    from hanzi_writing.envs import HanziComponentEnv

    training_env = HanziComponentEnv(
        effector=_make_effector(),
        geometry_config_path=config["geometry_config"],
        action_frame_stacking=0,
    )
    training_env.reset(
        options={
            "conditions": (exact_conditions(geometry)[0],),
            "speed_name": "medium",
            "delay_steps": geometry.validation_delay_steps,
            "deterministic": True,
        }
    )
    calls = []
    for _ in range(2):
        started = time.perf_counter()
        value = readonly_checkpoint_validation(
            policy,
            optimizer,
            hp,
            geometry,
            training_env=training_env,
            checkpoint_paths=(state_file,),
        )
        calls.append({"wall_time_s": time.perf_counter() - started, "result": value})
    if not _nested_equal(calls[0]["result"]["validation"], calls[1]["result"]["validation"]):
        raise RuntimeError("two read-only validation calls returned different results")
    state_hash_after = hashlib.sha256(state_file.read_bytes()).hexdigest()
    if state_hash_before != state_hash_after:
        raise RuntimeError("technical state file changed during validation")
    validation = calls[0]["result"]["validation"]
    group_count = sum(validation["rollout_group_count_by_rule"].values())
    if group_count != 81:
        raise RuntimeError("read-only validation benchmark must contain 81 rollout groups")
    checks = calls[0]["result"]["read_only_checks"]
    checks_second = calls[1]["result"]["read_only_checks"]
    if not all(value for value in checks.values() if isinstance(value, bool)):
        raise RuntimeError("first read-only validation state check failed")
    if not all(value for value in checks_second.values() if isinstance(value, bool)):
        raise RuntimeError("second read-only validation state check failed")
    result = {
        "passed": True,
        "rollout_group_count": group_count,
        "validation_wall_time_s": calls[0]["wall_time_s"],
        "repeat_validation_wall_time_s": calls[1]["wall_time_s"],
        "peak_rss_mb": _peak_rss_mb(),
        "validation_loss": validation["aggregate"]["phase_normalized_position_l1"],
        "two_calls_numerically_identical": True,
        "readonly_validation_function_tested": True,
        "read_only_checks": checks,
        "technical_state_sha256_unchanged": True,
        "optimizer_created_before_validation_function": True,
        "optimizer_created_during_validation_function": False,
        "final_validation_update": None,
    }
    _write_json(output / "validation_result.json", result)
    return result


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def finalize_preflight(
    repo: str | Path,
    config_path: str | Path,
    audit_json: str | Path,
    coverage_json: str | Path,
    smoke_json: str | Path,
    validation_json: str | Path,
    legacy_json: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    repository = Path(repo).resolve()
    output = Path(output_root).resolve()
    audit = _load_json(audit_json)
    coverage = _load_json(coverage_json)
    smoke = _load_json(smoke_json)
    validation = _load_json(validation_json)
    legacy = _load_json(legacy_json)
    required = (audit, coverage, smoke, validation)
    if not all(value.get("passed") is True for value in required):
        raise RuntimeError("cannot finalize a failed preflight gate")
    scheduled_validation_count = len(range(0, 10000, 500))
    median = smoke["post_warmup_update_wall_time_median_s"]
    validation_time = validation["validation_wall_time_s"]
    cost = {
        "estimated_10000_update_pure_training_s": 10000 * median,
        "scheduled_validation_count": scheduled_validation_count,
        "estimated_scheduled_validation_extra_s": scheduled_validation_count * validation_time,
        "estimated_final_validation_extra_s": validation_time,
        "estimated_complete_pilot_s": 10000 * median
        + (scheduled_validation_count + 1) * validation_time,
        "automatic_cost_acceptance_threshold_defined": False,
        "automatic_cost_decision_made": False,
    }
    summary = {
        "completed": True,
        "project": "hanzi_stroke_temporal_composition",
        "experiment": "fixed base movement duration with fixed compound-corner dwell nine-task pilot preflight",
        "target_audit": audit,
        "coverage": coverage,
        "technical_smoke": smoke,
        "readonly_validation_benchmark": validation,
        "cost_estimate": cost,
        "hard_integrity_counts": {
            "N_corner": 4,
            "metric_rows": EXPECTED_DIAGNOSTIC_METRIC_ROWS,
            "expected_trajectory_rows": EXPECTED_DIAGNOSTIC_TRAJECTORY_ROWS,
            "plots": EXPECTED_DIAGNOSTIC_PLOTS,
        },
        "final_validation_update": None,
        "formal_10k_pilot_started": False,
        "formal_75k_training_started": False,
        "complete_character_rollout_started": False,
        "boundary_intervention_started": False,
        "next_stage_requires_user_approval": True,
    }
    _write_json(output / "preflight_summary.json", summary)
    config_file = Path(config_path).resolve()
    configuration_paths = [
        repository / "configurations/hanzi_stroke_temporal_composition_fixed_duration_geometry.json",
        repository / "configurations/hanzi_stroke_temporal_composition_fixed_duration_coverage_v1.json",
        repository / "configurations/hanzi_stroke_temporal_composition_fixed_duration_smoke_v1.json",
        repository / "configurations/hanzi_stroke_temporal_composition_fixed_duration_pilot_v1.json",
    ]
    implementation_paths = [
        repository / "hanzi_writing/geometry.py",
        repository / "hanzi_writing/hanzi_geometry_final.py",
        repository / "hanzi_writing/envs.py",
        repository / "hanzi_writing/training.py",
        repository / "hanzi_writing/fixed_duration_protocol.py",
        repository / "hanzi_writing/fixed_duration_preflight.py",
        repository / "hanzi_writing/legacy_regression.py",
        repository / "server/export_hanzi_legacy_timing_reference.py",
        repository / "server/run_hanzi_fixed_duration_preflight.sh",
    ]
    provenance = {
        "git_head": _git(repository, "rev-parse", "HEAD"),
        "git_branch": _git(repository, "branch", "--show-current"),
        "submodule_head": _git(repository / "mRNNTorch", "rev-parse", "HEAD"),
        "legacy_head": legacy["legacy_head"],
        "legacy_submodule_head": legacy["submodule_head"],
        "legacy_exporter_sha256": legacy["exporter_sha256"],
        "fixed_duration_config": str(config_file),
        "fixed_duration_config_sha256": _sha256_file(config_file),
        "configuration_sha256": {
            str(path.relative_to(repository)): _sha256_file(path)
            for path in configuration_paths
        },
        "implementation_sha256": {
            str(path.relative_to(repository)): _sha256_file(path)
            for path in implementation_paths
        },
        "seed": 42,
        "validation_seed": 1042,
        "timing_mode": "fixed_movement_duration",
        "movement_intervals": {"fast": 50, "medium": 100, "slow": 150},
        "corner_dwell_intervals": 5,
        "corner_dwell_rules": ["hengzhe", "shugou"],
        "policy_read_only_validation": validation["read_only_checks"],
        "actual_coverage_case_count": coverage["case_count"],
        "actual_smoke_updates": smoke["updates"],
        "actual_validation_rollout_groups": validation["rollout_group_count"],
        "formal_training_started": False,
    }
    _write_json(output / "provenance.json", provenance)
    report = f"""# Fixed-duration nine-task pilot preflight

All implementation gates completed on the server. Legacy timing arrays match the old-HEAD reference element by element with maximum absolute difference 0.0. The fixed mode uses 50/100/150 base spatial intervals; hengzhe and shugou add exactly five continuous corner-dwell intervals.

- Legacy HEAD: `{legacy['legacy_head']}`
- Running HEAD: `{provenance['git_head']}`
- Deterministic coverage: {coverage['case_count']} cases, no optimizer update
- Technical smoke: {smoke['updates']} updates; loss and gradients finite; OOM observed = {smoke['oom_observed']}
- Post-warmup update median / p95: {median:.6f} s / {smoke['post_warmup_update_wall_time_p95_s']:.6f} s
- Smoke peak RSS: {smoke['peak_rss_mb']:.3f} MB
- 81-group read-only validation: {validation_time:.6f} s; peak RSS {validation['peak_rss_mb']:.3f} MB
- Estimated complete 10k pilot: {cost['estimated_complete_pilot_s']:.3f} s; no automatic cost threshold was applied
- Frozen future diagnostic gates: N_corner=4, metric rows=348, trajectory rows=35268, plots=18

This preflight is technical evidence only. It does not define behavioral success, does not compare fit with the old protocol, and does not isolate the causal effect of duration or dwell. No complete-character rollout, 10k pilot, 75k training, or boundary intervention was run. Starting the single-seed 10k pilot still requires explicit user approval.
"""
    with (output / "PREFLIGHT_REPORT.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--config", required=True)
    audit_parser.add_argument("--legacy-geometry", required=True)
    audit_parser.add_argument("--legacy-json", required=True)
    audit_parser.add_argument("--legacy-npz", required=True)
    audit_parser.add_argument("--output-dir", required=True)
    coverage_parser = subparsers.add_parser("coverage")
    coverage_parser.add_argument("--config", required=True)
    coverage_parser.add_argument("--output-dir", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--output-dir", required=True)
    smoke_parser.add_argument("--state", required=True)
    validation_parser = subparsers.add_parser("validation")
    validation_parser.add_argument("--config", required=True)
    validation_parser.add_argument("--state", required=True)
    validation_parser.add_argument("--output-dir", required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--repo", required=True)
    finalize_parser.add_argument("--config", required=True)
    finalize_parser.add_argument("--audit-json", required=True)
    finalize_parser.add_argument("--coverage-json", required=True)
    finalize_parser.add_argument("--smoke-json", required=True)
    finalize_parser.add_argument("--validation-json", required=True)
    finalize_parser.add_argument("--legacy-json", required=True)
    finalize_parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    if arguments.command == "audit":
        result = run_target_audit(
            arguments.config,
            arguments.legacy_geometry,
            arguments.legacy_json,
            arguments.legacy_npz,
            arguments.output_dir,
        )
    elif arguments.command == "coverage":
        result = run_coverage(arguments.config, arguments.output_dir)
    elif arguments.command == "smoke":
        result = run_smoke(arguments.config, arguments.output_dir, arguments.state)
    elif arguments.command == "validation":
        result = run_validation_benchmark(
            arguments.config, arguments.state, arguments.output_dir
        )
    else:
        result = finalize_preflight(
            arguments.repo,
            arguments.config,
            arguments.audit_json,
            arguments.coverage_json,
            arguments.smoke_json,
            arguments.validation_json,
            arguments.legacy_json,
            arguments.output_root,
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
