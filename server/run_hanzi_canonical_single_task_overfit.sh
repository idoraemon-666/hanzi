#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_hanzi_canonical_single_task_overfit.sh REPO ARCHIVE_BASE EXPECTED_HEAD APPROVED_STAGE0" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "canonical-single-duration-stage1-parallel6000" ]]; then
  echo "STOP: explicit authorization for canonical Stage 1 parallel 6000-update review is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
ARCHIVE_BASE="$(realpath -m "$2")"
EXPECTED_HEAD="$3"
APPROVED_STAGE0="$(realpath "$4")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
APPROVED_STAGE0_HEAD=2cf251ce828d6e50384a359dfa15d1de65c53726
CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_canonical_single_task_overfit_v1.json
SHARED_RELATIVE=configurations/hanzi_stroke_temporal_composition_canonical_shared_9task_v1.json
OUTPUT_DIR="$REPO/runs/hanzi_stroke_temporal_composition/canonical_single_task_overfit/dev42"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$REPO/$CONFIG_RELATIVE"
test -f "$REPO/$SHARED_RELATIVE"
test -d "$APPROVED_STAGE0"
test "$(git -C "$REPO" branch --show-current)" = "codex/hanzi-stroke-temporal-composition"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in "$OUTPUT_DIR" "$LOG_TMP" "$ARCHIVE" "$ARCHIVE_SHA"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: refusing to overwrite $path" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$ARCHIVE_BASE")"

set +e
(
  set -euo pipefail
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES=''
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONHASHSEED=0
  (
    cd "$APPROVED_STAGE0"
    sha256sum -c SHA256SUMS
  )
  APPROVED_STAGE0="$APPROVED_STAGE0" APPROVED_STAGE0_HEAD="$APPROVED_STAGE0_HEAD" \
    SUBMODULE_HEAD="$SUBMODULE_HEAD" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["APPROVED_STAGE0"])
with (root / "provenance.json").open(encoding="utf-8") as handle:
    provenance = json.load(handle)
assert provenance["completed"] is True
assert provenance["formal_stage0_completed"] is True
assert provenance["git_head"] == os.environ["APPROVED_STAGE0_HEAD"]
assert provenance["submodule_head"] == os.environ["SUBMODULE_HEAD"]
assert provenance["target_rows"] == 3120
assert provenance["canonical_stage1_started"] is False
assert provenance["shared_9task_started"] is False
assert provenance["formal_75k_started"] is False
assert provenance["complete_character_rollout_started"] is False
print("APPROVED_CANONICAL_STAGE0_VERIFIED=1")
PY
  "$PYTHON" - <<'PY'
import importlib.metadata
import torch

assert torch.__version__ == "2.6.0+cpu", torch.__version__
assert torch.cuda.is_available() is False
assert importlib.metadata.version("motornet") == "0.2.0"
print("SERVER_RUNTIME_PREFLIGHT=1")
PY
  "$PYTHON" -m unittest \
    tests.test_hanzi_geometry \
    tests.test_hanzi_environment \
    tests.test_hanzi_training_protocol \
    tests.test_hanzi_fit_diagnostic \
    tests.test_hanzi_boundary_intervention \
    tests.test_hanzi_fixed_duration_protocol \
    tests.test_hanzi_fixed_duration_diagnostic \
    tests.test_phase_normalized_loss \
    tests.test_hanzi_canonical_protocol \
    tests.test_hanzi_canonical_overfit
  echo "CANONICAL_STAGE1_PARALLEL6000_STARTED=1"
  echo "CANONICAL_SHARED_9TASK_STARTED=0"
  "$PYTHON" -m hanzi_writing.canonical_overfit \
    --config "$CONFIG_RELATIVE" \
    --prepare \
    --approved-stage0 "$APPROVED_STAGE0"
  TASKS=(heng shu pie na dian ti hengzhe shugou move)
  WORKER_LOG_DIR="$OUTPUT_DIR/worker_logs"
  mkdir "$WORKER_LOG_DIR"
  WORKER_PIDS=()
  terminate_workers() {
    for pid in "${WORKER_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap terminate_workers INT TERM EXIT
  for task in "${TASKS[@]}"; do
    "$PYTHON" -m hanzi_writing.canonical_overfit \
      --config "$CONFIG_RELATIVE" \
      --task "$task" \
      --target-updates 6000 \
      > "$WORKER_LOG_DIR/$task.log" 2>&1 &
    WORKER_PIDS+=("$!")
    echo "CANONICAL_TASK_LAUNCHED=$task PID=$!"
  done
  echo "CANONICAL_PARALLEL_WORKER_COUNT=${#WORKER_PIDS[@]}"
  WORKER_FAILURE=0
  for index in "${!WORKER_PIDS[@]}"; do
    task="${TASKS[$index]}"
    pid="${WORKER_PIDS[$index]}"
    if wait "$pid"; then
      echo "CANONICAL_TASK_COMPLETE=$task PID=$pid"
    else
      status="$?"
      echo "CANONICAL_TASK_FAILED=$task PID=$pid EXIT=$status" >&2
      WORKER_FAILURE=1
    fi
  done
  trap - INT TERM EXIT
  if [[ "$WORKER_FAILURE" -ne 0 ]]; then
    exit 1
  fi
  "$PYTHON" -m hanzi_writing.canonical_overfit \
    --config "$CONFIG_RELATIVE" \
    --finalize \
    --approved-stage0 "$APPROVED_STAGE0"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Canonical Stage 1 parallel 6000-update review stopped; see $LOG_TMP and worker logs." >&2
  exit 1
fi

TEST_COUNT="$(sed -nE 's/^Ran ([0-9]+) tests? in .*/\1/p' "$LOG_TMP" | tail -n 1)"
test "$TEST_COUNT" = 65
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi
test -d "$OUTPUT_DIR"
mv "$LOG_TMP" "$OUTPUT_DIR/execution.log"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$OUTPUT_DIR/exit_code.txt"

REPO="$REPO" OUTPUT_DIR="$OUTPUT_DIR" APPROVED_STAGE0="$APPROVED_STAGE0" \
TEST_COUNT="$TEST_COUNT" PROGRAM_EXIT="$PROGRAM_EXIT" TEE_EXIT="$TEE_EXIT" \
"$PYTHON" - <<'PY'
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

repo = Path(os.environ["REPO"])
output = Path(os.environ["OUTPUT_DIR"])

def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = load_json(output / "canonical_condition_manifest.json")
assert manifest["integrity"]["stroke_rule_count"] == 8
assert manifest["integrity"]["exact_move_count"] == 12
assert manifest["integrity"]["adjacent_duplicate_targets"] == 0
assert manifest["integrity"]["all_targets_finite"] is True
assert manifest["integrity"]["maximum_stroke_start_difference_m"] <= 1e-12
assert manifest["integrity"]["maximum_stroke_end_difference_m"] <= 1e-12
with np.load(output / "canonical_target_trajectories.npz", allow_pickle=False) as data:
    assert len(data["sample_index"]) == 3120
    assert all(data[name].dtype != object for name in data.files)
with np.load(output / "single_task_overfit_trajectories.npz", allow_pickle=False) as data:
    assert len(data["sample_index"]) == 3120
    assert all(data[name].dtype != object for name in data.files)

with (output / "single_task_overfit_metrics.csv").open(
    encoding="utf-8", newline=""
) as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 40
assert len({(row["checkpoint"], row["condition_id"]) for row in rows}) == 40
for row in rows:
    for key, value in row.items():
        if value and key.endswith(("_m", "_deg", "_ratio", "_loss")):
            assert math.isfinite(float(value)), (key, value)

models = output / "models"
expected_model_files = {
    "best_checkpoint.pt",
    "continuation_checkpoint.pt",
    "review_checkpoint.pt",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
}
assert {path.name for path in models.iterdir()} == {
    "heng", "shu", "pie", "na", "dian", "ti", "hengzhe", "shugou", "move"
}
for directory in models.iterdir():
    assert {path.name for path in directory.iterdir()} == expected_model_files
    assert sum(1 for line in (directory / "training_metrics.jsonl").open()) == 60
    validation_rows = [
        json.loads(line) for line in (directory / "validation_metrics.jsonl").open()
    ]
    assert len(validation_rows) == 25
    assert validation_rows[-1]["validation_kind"] == "review_read_only"
    assert validation_rows[-1]["update"] == 5999
    assert all(validation_rows[-1]["read_only_checks"].values())

assert len(list((output / "plots").glob("*.png"))) == 9
provenance_path = output / "provenance.json"
provenance = load_json(provenance_path)
assert provenance["completed"] is True
assert provenance["integrity"] == {
    "all_numeric_values_finite": True,
    "best_trajectory_rows": 3120,
    "metrics_rows": 40,
    "plots": 9,
    "single_tasks_completed": 9,
}
assert len(provenance["checkpoint_sha256"]) == 27
assert provenance["initial_review_updates"] == 6000
assert provenance["update_cap"] is None
assert provenance["execution_mode"] == "nine_independent_parallel_processes"
assert {
    summary["completed_updates"] for summary in provenance["task_summaries"]
} == {6000}
assert provenance["approved_stage0_directory"] == os.environ["APPROVED_STAGE0"]
assert set(provenance["approved_stage0_sha256"]) == {
    "canonical_condition_manifest.json",
    "canonical_target_trajectories.npz",
    "canonical_target_audit.png",
}
for relative, expected in provenance["checkpoint_sha256"].items():
    assert sha256(output / relative) == expected
assert provenance["shared_9task_started"] is False
assert provenance["formal_75k_started"] is False
assert provenance["complete_character_rollout_started"] is False
provenance.update(
    {
        "test_count": int(os.environ["TEST_COUNT"]),
        "skipped_test_count": 0,
        "program_exit_code": int(os.environ["PROGRAM_EXIT"]),
        "tee_exit_code": int(os.environ["TEE_EXIT"]),
    }
)
with provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

(
  cd "$OUTPUT_DIR"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$OUTPUT_DIR")" -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"

echo "CANONICAL_STAGE1_PARALLEL6000_COMPLETE=1"
echo "CANONICAL_SHARED_9TASK_STARTED=0"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "COMPLETE_CHARACTER_ROLLOUT_STARTED=0"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ARCHIVE=$ARCHIVE"
