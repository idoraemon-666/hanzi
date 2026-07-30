#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: bash server/run_hanzi_canonical_parallel_lr_lines.sh REPO EXPECTED_HEAD APPROVED_STAGE0 SOURCE_REFINEMENT SCRATCH_ARCHIVE_BASE FOLLOWUP_ARCHIVE_BASE" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "canonical-parallel-scratch1e4-10000-selected-followup2000" ]]; then
  echo "STOP: explicit authorization for the two canonical low-LR lines is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
EXPECTED_HEAD="$2"
APPROVED_STAGE0="$(realpath "$3")"
SOURCE_REFINEMENT="$(realpath "$4")"
SCRATCH_ARCHIVE_BASE="$(realpath -m "$5")"
FOLLOWUP_ARCHIVE_BASE="$(realpath -m "$6")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
APPROVED_STAGE0_HEAD=2cf251ce828d6e50384a359dfa15d1de65c53726
SCRATCH_CONFIG=configurations/hanzi_stroke_temporal_composition_canonical_scratch_lr1e4_v1.json
FOLLOWUP_CONFIG=configurations/hanzi_stroke_temporal_composition_canonical_selected_refinement_followup_v1.json
SCRATCH_OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/canonical_stroke_scratch_lr1e4_10000/dev42"
FOLLOWUP_OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/canonical_stroke_refinement_followup/dev42"
SCRATCH_ARCHIVE="${SCRATCH_ARCHIVE_BASE}.tar.gz"
SCRATCH_ARCHIVE_SHA="${SCRATCH_ARCHIVE}.sha256"
FOLLOWUP_ARCHIVE="${FOLLOWUP_ARCHIVE_BASE}.tar.gz"
FOLLOWUP_ARCHIVE_SHA="${FOLLOWUP_ARCHIVE}.sha256"
LOG_TMP="${SCRATCH_ARCHIVE_BASE}.combined.execution.log.tmp"

test -d "$REPO/.git"
test -x "$PYTHON"
test -d "$APPROVED_STAGE0"
test -d "$SOURCE_REFINEMENT"
test -f "$REPO/$SCRATCH_CONFIG"
test -f "$REPO/$FOLLOWUP_CONFIG"
test -f "$APPROVED_STAGE0/SHA256SUMS"
test -f "$SOURCE_REFINEMENT/SHA256SUMS"
test "$(git -C "$REPO" branch --show-current)" = "codex/hanzi-stroke-temporal-composition"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in \
  "$SCRATCH_OUTPUT" "$FOLLOWUP_OUTPUT" "$LOG_TMP" \
  "$SCRATCH_ARCHIVE" "$SCRATCH_ARCHIVE_SHA" \
  "$FOLLOWUP_ARCHIVE" "$FOLLOWUP_ARCHIVE_SHA"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: refusing to overwrite $path" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "$SCRATCH_ARCHIVE_BASE")"
mkdir -p "$(dirname "$FOLLOWUP_ARCHIVE_BASE")"

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
  (
    cd "$SOURCE_REFINEMENT"
    sha256sum -c SHA256SUMS
  )
  APPROVED_STAGE0="$APPROVED_STAGE0" APPROVED_STAGE0_HEAD="$APPROVED_STAGE0_HEAD" \
    SOURCE_REFINEMENT="$SOURCE_REFINEMENT" SUBMODULE_HEAD="$SUBMODULE_HEAD" \
    "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

stage0 = Path(os.environ["APPROVED_STAGE0"])
with (stage0 / "provenance.json").open(encoding="utf-8") as handle:
    stage0_provenance = json.load(handle)
assert stage0_provenance["completed"] is True
assert stage0_provenance["formal_stage0_completed"] is True
assert stage0_provenance["git_head"] == os.environ["APPROVED_STAGE0_HEAD"]
assert stage0_provenance["submodule_head"] == os.environ["SUBMODULE_HEAD"]
assert stage0_provenance["target_rows"] == 3120
assert stage0_provenance["canonical_stage1_started"] is False

source = Path(os.environ["SOURCE_REFINEMENT"])
with (source / "provenance.json").open(encoding="utf-8") as handle:
    source_provenance = json.load(handle)
assert source_provenance["completed"] is True
assert source_provenance["variant"] == "canonical_stroke_refinement_v1"
assert source_provenance["git_head"] == "d763b59ccc09d6e8673f518f0aead89094fb2226"
assert source_provenance["submodule_head"] == os.environ["SUBMODULE_HEAD"]
assert source_provenance["learning_rate"] == 0.0001
assert source_provenance["additional_updates"] == 2000
assert source_provenance["move_started"] is False
assert source_provenance["shared_9task_started"] is False
assert source_provenance["formal_75k_started"] is False
assert source_provenance["complete_character_rollout_started"] is False
print("APPROVED_CANONICAL_STAGE0_VERIFIED=1")
print("SOURCE_CANONICAL_REFINEMENT_VERIFIED=1")
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
    tests.test_hanzi_canonical_overfit \
    tests.test_hanzi_canonical_refinement \
    tests.test_hanzi_canonical_parallel_lines

  echo "CANONICAL_SCRATCH_LOW_LR_STARTED=1"
  echo "CANONICAL_SELECTED_FOLLOWUP_STARTED=1"
  echo "MOVE_TRAINING_STARTED=0"
  "$PYTHON" -m hanzi_writing.canonical_parallel_lines scratch-prepare \
    --config "$SCRATCH_CONFIG" \
    --approved-stage0 "$APPROVED_STAGE0"
  "$PYTHON" -m hanzi_writing.canonical_parallel_lines followup-prepare \
    --config "$FOLLOWUP_CONFIG" \
    --source "$SOURCE_REFINEMENT"

  SCRATCH_TASKS=(heng shu pie na dian ti hengzhe shugou)
  FOLLOWUP_TASKS=(heng pie na shugou)
  mkdir "$SCRATCH_OUTPUT/worker_logs" "$FOLLOWUP_OUTPUT/worker_logs"
  WORKER_PIDS=()
  WORKER_LABELS=()
  terminate_workers() {
    for pid in "${WORKER_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap terminate_workers INT TERM EXIT
  for task in "${SCRATCH_TASKS[@]}"; do
    "$PYTHON" -m hanzi_writing.canonical_parallel_lines scratch-task \
      --config "$SCRATCH_CONFIG" \
      --task "$task" \
      > "$SCRATCH_OUTPUT/worker_logs/$task.log" 2>&1 &
    WORKER_PIDS+=("$!")
    WORKER_LABELS+=("scratch:$task")
    echo "CANONICAL_SCRATCH_TASK_LAUNCHED=$task PID=$!"
  done
  for task in "${FOLLOWUP_TASKS[@]}"; do
    "$PYTHON" -m hanzi_writing.canonical_parallel_lines followup-task \
      --config "$FOLLOWUP_CONFIG" \
      --source "$SOURCE_REFINEMENT" \
      --task "$task" \
      > "$FOLLOWUP_OUTPUT/worker_logs/$task.log" 2>&1 &
    WORKER_PIDS+=("$!")
    WORKER_LABELS+=("followup:$task")
    echo "CANONICAL_FOLLOWUP_TASK_LAUNCHED=$task PID=$!"
  done
  test "${#WORKER_PIDS[@]}" -eq 12
  echo "CANONICAL_TOTAL_PARALLEL_WORKER_COUNT=12"
  WORKER_FAILURE=0
  for index in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$index]}"
    label="${WORKER_LABELS[$index]}"
    if wait "$pid"; then
      echo "CANONICAL_PARALLEL_TASK_COMPLETE=$label PID=$pid"
    else
      status="$?"
      echo "CANONICAL_PARALLEL_TASK_FAILED=$label PID=$pid EXIT=$status" >&2
      WORKER_FAILURE=1
    fi
  done
  trap - INT TERM EXIT
  test "$WORKER_FAILURE" -eq 0

  "$PYTHON" -m hanzi_writing.canonical_parallel_lines scratch-finalize \
    --config "$SCRATCH_CONFIG" \
    --approved-stage0 "$APPROVED_STAGE0"
  "$PYTHON" -m hanzi_writing.canonical_parallel_lines followup-finalize \
    --config "$FOLLOWUP_CONFIG" \
    --source "$SOURCE_REFINEMENT"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Canonical parallel low-LR lines stopped; see $LOG_TMP and worker logs." >&2
  exit 1
fi

TEST_COUNT="$(sed -nE 's/^Ran ([0-9]+) tests? in .*/\1/p' "$LOG_TMP" | tail -n 1)"
test "$TEST_COUNT" = 75
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi
test -d "$SCRATCH_OUTPUT"
test -d "$FOLLOWUP_OUTPUT"
cp "$LOG_TMP" "$SCRATCH_OUTPUT/execution.log"
cp "$LOG_TMP" "$FOLLOWUP_OUTPUT/execution.log"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$SCRATCH_OUTPUT/exit_code.txt"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$FOLLOWUP_OUTPUT/exit_code.txt"

SCRATCH_OUTPUT="$SCRATCH_OUTPUT" FOLLOWUP_OUTPUT="$FOLLOWUP_OUTPUT" \
TEST_COUNT="$TEST_COUNT" PROGRAM_EXIT="$PROGRAM_EXIT" TEE_EXIT="$TEE_EXIT" \
"$PYTHON" - <<'PY'
import csv
import hashlib
import json
import math
import os
from pathlib import Path

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

scratch = Path(os.environ["SCRATCH_OUTPUT"])
followup = Path(os.environ["FOLLOWUP_OUTPUT"])
scratch_provenance_path = scratch / "provenance.json"
followup_provenance_path = followup / "provenance.json"
scratch_provenance = load_json(scratch_provenance_path)
followup_provenance = load_json(followup_provenance_path)

assert scratch_provenance["completed"] is True
assert scratch_provenance["variant"] == "canonical_eight_stroke_scratch_lr1e4_v1"
assert scratch_provenance["from_scratch"] is True
assert scratch_provenance["learning_rate"] == 0.0001
assert scratch_provenance["updates"] == 10000
assert scratch_provenance["execution_mode"] == "eight_independent_parallel_processes"
assert scratch_provenance["integrity"] == {
    "all_numeric_values_finite": True,
    "metrics_rows": 16,
    "plots": 9,
    "stroke_tasks_completed": 8,
    "trajectory_rows": 2616,
}
assert len(scratch_provenance["checkpoint_sha256"]) == 24
assert {row["completed_updates"] for row in scratch_provenance["task_summaries"]} == {10000}
for task in ("heng", "shu", "pie", "na", "dian", "ti", "hengzhe", "shugou"):
    model = scratch / "models" / task
    assert sum(1 for _ in (model / "training_metrics.jsonl").open()) == 100
    rows = [json.loads(line) for line in (model / "validation_metrics.jsonl").open()]
    assert len(rows) == 101
    assert rows[-1]["validation_kind"] == "review_read_only"
    assert rows[-1]["update"] == 9999
    assert all(rows[-1]["read_only_checks"].values())

assert followup_provenance["completed"] is True
assert followup_provenance["variant"] == "canonical_selected_refinement_followup_v1"
assert followup_provenance["checkpoint_variant"] == "canonical_stroke_refinement_v1"
assert followup_provenance["exact_continuation"] is True
assert followup_provenance["learning_rate"] == 0.0001
assert followup_provenance["source_completed_additional_updates"] == 2000
assert followup_provenance["additional_updates"] == 2000
assert followup_provenance["target_total_refinement_updates"] == 4000
assert followup_provenance["selected_tasks"] == ["heng", "pie", "na", "shugou"]
assert followup_provenance["execution_mode"] == "four_independent_parallel_processes"
assert followup_provenance["integrity"] == {
    "all_numeric_values_finite": True,
    "metrics_rows": 12,
    "plots": 5,
    "stroke_tasks_completed": 4,
    "trajectory_rows": 1962,
}
assert len(followup_provenance["checkpoint_sha256"]) == 12
assert len(followup_provenance["source_checkpoint_sha256"]) == 4
assert {row["completed_updates"] for row in followup_provenance["task_summaries"]} == {4000}
for task in ("heng", "pie", "na", "shugou"):
    model = followup / "models" / task
    assert sum(1 for _ in (model / "training_metrics.jsonl").open()) == 40
    rows = [json.loads(line) for line in (model / "validation_metrics.jsonl").open()]
    assert len(rows) == 42
    assert rows[-1]["validation_kind"] == "review_read_only"
    assert rows[-1]["update"] == 3999
    assert all(rows[-1]["read_only_checks"].values())

for output, provenance, metrics_name in (
    (scratch, scratch_provenance, "scratch_low_lr_metrics.csv"),
    (followup, followup_provenance, "selected_followup_metrics.csv"),
):
    with (output / metrics_name).open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == provenance["integrity"]["metrics_rows"]
    for row in metric_rows:
        for key, value in row.items():
            if value and key.endswith(("_m", "_deg", "_ratio", "_loss")):
                assert math.isfinite(float(value)), (key, value)
    for relative, expected in provenance["checkpoint_sha256"].items():
        assert sha256(output / relative) == expected
    assert provenance["move_started"] is False
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

with scratch_provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(scratch_provenance, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
with followup_provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(followup_provenance, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

for output in "$SCRATCH_OUTPUT" "$FOLLOWUP_OUTPUT"; do
  (
    cd "$output"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS
  )
done
tar -C "$(dirname "$SCRATCH_OUTPUT")" -czf "$SCRATCH_ARCHIVE" "$(basename "$SCRATCH_OUTPUT")"
sha256sum "$SCRATCH_ARCHIVE" > "$SCRATCH_ARCHIVE_SHA"
tar -C "$(dirname "$FOLLOWUP_OUTPUT")" -czf "$FOLLOWUP_ARCHIVE" "$(basename "$FOLLOWUP_OUTPUT")"
sha256sum "$FOLLOWUP_ARCHIVE" > "$FOLLOWUP_ARCHIVE_SHA"
rm "$LOG_TMP"

echo "CANONICAL_PARALLEL_LOW_LR_LINES_COMPLETE=1"
echo "CANONICAL_SCRATCH_LOW_LR_COMPLETE=1"
echo "CANONICAL_SELECTED_FOLLOWUP_COMPLETE=1"
echo "MOVE_TRAINING_STARTED=0"
echo "SHARED_9TASK_STARTED=0"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "COMPLETE_CHARACTER_ROLLOUT_STARTED=0"
echo "SCRATCH_OUTPUT=$SCRATCH_OUTPUT"
echo "SCRATCH_ARCHIVE=$SCRATCH_ARCHIVE"
echo "FOLLOWUP_OUTPUT=$FOLLOWUP_OUTPUT"
echo "FOLLOWUP_ARCHIVE=$FOLLOWUP_ARCHIVE"
