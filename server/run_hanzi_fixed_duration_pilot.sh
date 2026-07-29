#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_hanzi_fixed_duration_pilot.sh REPO RUN_ROOT EXPECTED_HEAD PREFLIGHT_ROOT" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "fixed-duration-9task-10k-pilot-dev42" ]]; then
  echo "STOP: explicit authorization for the fixed-duration single-seed 10k pilot is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
EXPECTED_HEAD="$3"
PREFLIGHT_ROOT="$(realpath "$4")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
PILOT_CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_fixed_duration_pilot_v1.json
DIAGNOSTIC_CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_fixed_duration_diagnostic_v1.json
PILOT_CONFIG="$REPO/$PILOT_CONFIG_RELATIVE"
DIAGNOSTIC_CONFIG="$REPO/$DIAGNOSTIC_CONFIG_RELATIVE"
TRAIN_OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/fixed_duration_pilot/dev42"
DIAGNOSTIC_OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/fixed_duration_pilot_diagnostic/dev42_best_final"
ARCHIVE="${RUN_ROOT}.tar.gz"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$PILOT_CONFIG"
test -f "$DIAGNOSTIC_CONFIG"
test -d "$PREFLIGHT_ROOT"
test "$(git -C "$REPO" branch --show-current)" = "codex/hanzi-stroke-temporal-composition"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in "$RUN_ROOT" "$TRAIN_OUTPUT" "$DIAGNOSTIC_OUTPUT" "$ARCHIVE" "${ARCHIVE}.sha256"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: refusing to overwrite $path" >&2
    exit 1
  fi
done

(
  cd "$PREFLIGHT_ROOT"
  sha256sum -c SHA256SUMS
)
PREFLIGHT_ROOT="$PREFLIGHT_ROOT" REPO="$REPO" "$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO"])
root = Path(os.environ["PREFLIGHT_ROOT"])

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = json.load((root / "preflight_summary.json").open(encoding="utf-8"))
provenance = json.load((root / "provenance.json").open(encoding="utf-8"))
assert sha256(root / "preflight_summary.json") == "4062348d356261e0f5425792327e626f4f425f8cb2829638e96bcff541d44cc9"
assert sha256(root / "provenance.json") == "896e167da7b0869ceab5d3b286f253ae180ad872dbbced5543462f0c017d91cf"
assert provenance["git_head"] == "711c4f12daf2f74d6c703806c207ca11147a183a"
assert provenance["submodule_head"] == "ac0c4f589eae37bbde63968912925de99232e306"
assert summary["completed"] is True
assert summary["formal_10k_pilot_started"] is False
assert summary["formal_75k_training_started"] is False
assert summary["coverage"]["passed"] is True
assert summary["technical_smoke"]["passed"] is True
assert summary["readonly_validation_benchmark"]["passed"] is True
checks = summary["readonly_validation_benchmark"]["read_only_checks"]
assert checks["captured_environment_generator_paths"] == [
    "HanziComponentEnv.effector._np_random"
]
assert all(value for value in checks.values() if isinstance(value, bool))
assert summary["target_audit"]["legacy_regression"]["all_arrays_np_array_equal"] is True
assert summary["target_audit"]["legacy_regression"]["max_abs_difference"] == 0.0
assert summary["target_audit"]["physical_geometry_np_array_equal"] is True
assert summary["hard_integrity_counts"] == {
    "N_corner": 4,
    "expected_trajectory_rows": 35268,
    "metric_rows": 348,
    "plots": 18,
}
for relative in (
    "hanzi_writing/envs.py",
    "hanzi_writing/fixed_duration_protocol.py",
    "hanzi_writing/geometry.py",
    "hanzi_writing/hanzi_geometry_final.py",
    "hanzi_writing/training.py",
):
    assert sha256(repo / relative) == provenance["implementation_sha256"][relative]
for relative in (
    "configurations/hanzi_stroke_temporal_composition_fixed_duration_geometry.json",
    "configurations/hanzi_stroke_temporal_composition_fixed_duration_pilot_v1.json",
):
    assert sha256(repo / relative) == provenance["configuration_sha256"][relative]
assert sha256(root / "fixed_duration_condition_manifest.csv") == (
    "19928754f9ce75e5f30917b173eeaf1791a9c13c95607e0fa133795879f7d211"
)
PY

mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/start_utc.txt"
cp "$PILOT_CONFIG" "$RUN_ROOT/pilot_configuration.json"
cp "$DIAGNOSTIC_CONFIG" "$RUN_ROOT/diagnostic_configuration.json"
cp "$PREFLIGHT_ROOT/PREFLIGHT_REPORT.md" "$RUN_ROOT/PREFLIGHT_REPORT.md"
cp "$PREFLIGHT_ROOT/provenance.json" "$RUN_ROOT/preflight_provenance.json"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/git_head.txt"
git -C "$REPO/mRNNTorch" rev-parse HEAD > "$RUN_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"

set +e
(
  set -euo pipefail
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES=''
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONHASHSEED=0
  "$PYTHON" -m unittest discover -s tests
  echo "FORMAL_10K_PILOT_STARTED=1"
  echo "FORMAL_75K_TRAINING_STARTED=0"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_start_utc.txt"
  "$PYTHON" -m hanzi_writing.training --config "$PILOT_CONFIG_RELATIVE"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_end_utc.txt"
  "$PYTHON" -m hanzi_writing.fixed_duration_diagnostic \
    --config "$DIAGNOSTIC_CONFIG_RELATIVE"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/diagnostic_end_utc.txt"

  REPO="$REPO" RUN_ROOT="$RUN_ROOT" TRAIN_OUTPUT="$TRAIN_OUTPUT" \
    DIAGNOSTIC_OUTPUT="$DIAGNOSTIC_OUTPUT" "$PYTHON" - <<'PY'
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch

repo = Path(os.environ["REPO"])
run_root = Path(os.environ["RUN_ROOT"])
training = Path(os.environ["TRAIN_OUTPUT"])
diagnostic = Path(os.environ["DIAGNOSTIC_OUTPUT"])

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = json.load((training / "training_summary.json").open(encoding="utf-8"))
assert summary["project"] == "hanzi_stroke_temporal_composition"
assert summary["variant"] == "fixed_duration_9task_pilot_v1"
assert summary["seed"] == 42
assert summary["updates"] == 10000
assert summary["final_validation_update"] == 9999
assert summary["final_validation_read_only"] is True
assert math.isfinite(summary["best_validation_loss"])
assert math.isfinite(summary["final_validation_loss"])
checks = summary["final_validation_read_only_checks"]
assert checks["captured_environment_generator_paths"] == [
    "HanziComponentEnv.effector._np_random"
]
assert all(value for value in checks.values() if isinstance(value, bool))

with (training / "checkpoint_validation_metrics.jsonl").open(encoding="utf-8") as handle:
    scheduled = [json.loads(line) for line in handle if line.strip()]
assert [row["update"] for row in scheduled] == list(range(0, 10000, 500))
assert len(scheduled) == 20

best_path = training / "best_checkpoint.pt"
final_path = training / "final_continuation_checkpoint.pt"
best = torch.load(best_path, map_location="cpu", weights_only=False)
final = torch.load(final_path, map_location="cpu", weights_only=False)
assert best["checkpoint_kind"] == "best_shared9"
assert best["variant"] == "fixed_duration_9task_pilot_v1"
assert 0 <= int(best["update"]) <= 9500 and int(best["update"]) % 500 == 0
assert final["checkpoint_kind"] == "final_continuation"
assert final["variant"] == "fixed_duration_9task_pilot_v1"
assert int(final["update"]) == 9999

fit = json.load(
    (diagnostic / "fixed_duration_diagnostic_summary.json").open(encoding="utf-8")
)
assert fit["completed"] is True
assert fit["behavioral_pass_fail_defined"] is False
assert fit["causal_effect_claimed"] is False
assert fit["complete_character_rollout_started"] is False
assert fit["boundary_intervention_started"] is False
assert fit["integrity"]["exact_stroke_placement_count"] == 46
assert fit["integrity"]["exact_move_count"] == 12
assert fit["integrity"]["N_corner"] == 4
assert fit["integrity"]["metric_rows"] == 348
assert fit["integrity"]["trajectory_rows"] == 35268
assert fit["integrity"]["plots"] == 18
assert fit["integrity"]["rollout_groups"] == 162
assert fit["integrity"]["checkpoint_files_unchanged"] is True

head = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
submodule = subprocess.check_output(
    ["git", "-C", str(repo / "mRNNTorch"), "rev-parse", "HEAD"], text=True
).strip()
provenance = {
    "project": "hanzi_stroke_temporal_composition",
    "run_kind": "fixed_duration_9task_pilot_v1",
    "completed": True,
    "git_head": head,
    "submodule_head": submodule,
    "seed": 42,
    "validation_seed": 1042,
    "formal_10k_pilot_started": True,
    "formal_10k_pilot_completed": True,
    "formal_75k_training_started": False,
    "complete_character_rollout_started": False,
    "boundary_intervention_started": False,
    "preflight_head": "711c4f12daf2f74d6c703806c207ca11147a183a",
    "preflight_provenance_sha256": sha256(run_root / "preflight_provenance.json"),
    "pilot_config_sha256": sha256(run_root / "pilot_configuration.json"),
    "diagnostic_config_sha256": sha256(run_root / "diagnostic_configuration.json"),
    "best_checkpoint": {
        "path": str(best_path),
        "sha256": sha256(best_path),
        "update": int(best["update"]),
        "validation_loss": float(best["validation_loss"]),
    },
    "final_checkpoint": {
        "path": str(final_path),
        "sha256": sha256(final_path),
        "update": int(final["update"]),
        "validation_loss": float(final["validation_loss"]),
    },
    "final_validation_update": summary["final_validation_update"],
    "final_validation_loss": summary["final_validation_loss"],
    "final_validation_read_only_checks": checks,
    "diagnostic_integrity": fit["integrity"],
}
with (run_root / "provenance.json").open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")

report = f"""# Fixed-duration nine-task single-seed 10k pilot execution

- Git HEAD: `{head}`
- Submodule HEAD: `{submodule}`
- Seed: `42`
- Updates: `10000`
- Best checkpoint update: `{int(best['update'])}`
- Best scheduled-validation loss: `{float(best['validation_loss']):.12g}`
- Final checkpoint update: `9999`
- Final read-only validation loss: `{float(summary['final_validation_loss']):.12g}`
- Exact diagnostic rows: `348`
- Trajectory rows: `35268`
- Plots: `18`

No 75k training, complete-character rollout, or boundary intervention was run.
No behavioral pass/fail threshold or independent causal duration/dwell claim is defined.
"""
(run_root / "PILOT_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")
PY
) 2>&1 | tee "$RUN_ROOT/execution.log"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${RUN_CODES[0]}" > "$RUN_ROOT/exit_code.txt"
test "${RUN_CODES[0]}" -eq 0
test "${RUN_CODES[1]}" -eq 0

cp -a "$TRAIN_OUTPUT" "$RUN_ROOT/training_output"
cp -a "$DIAGNOSTIC_OUTPUT" "$RUN_ROOT/diagnostic_output"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/end_utc.txt"
(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
echo "FORMAL_10K_PILOT_STARTED=1"
echo "FORMAL_10K_PILOT_COMPLETE=1"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "COMPLETE_CHARACTER_ROLLOUT_STARTED=0"
echo "BOUNDARY_INTERVENTION_STARTED=0"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
