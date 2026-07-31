#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_canonical_start_loss_comparison.sh REPO EXPECTED_HEAD ARCHIVE_BASE" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "canonical-start-loss-comparison-v1" ]]; then
  echo "STOP: explicit authorization for the canonical start-loss comparison is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
EXPECTED_HEAD="$2"
ARCHIVE_BASE="$(realpath -m "$3")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
CONFIG=configurations/hanzi_stroke_temporal_composition_canonical_start_loss_comparison_v1.json
OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/canonical_start_loss_comparison/dev42"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"
SOURCE_DIRECTORIES=(
  "$REPO/runs/hanzi_stroke_temporal_composition/canonical_single_task_overfit/dev42"
  "$REPO/runs/hanzi_stroke_temporal_composition/canonical_stroke_refinement/dev42"
  "$REPO/runs/hanzi_stroke_temporal_composition/canonical_stroke_scratch_lr1e4_10000/dev42"
  "$REPO/runs/hanzi_stroke_temporal_composition/canonical_stroke_refinement_followup/dev42"
  "$REPO/runs/hanzi_stroke_temporal_composition/canonical_shugou_checkpoint_experiment/dev42"
)
ARMS=(full_trial onset_window)
TASKS=(heng shu pie na dian ti hengzhe shugou)

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$REPO/$CONFIG"
test "$(git -C "$REPO" branch --show-current)" = "codex/hanzi-stroke-temporal-composition"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for source in "${SOURCE_DIRECTORIES[@]}"; do
  test -d "$source"
  test -f "$source/SHA256SUMS"
done
for path in "$OUTPUT" "$ARCHIVE" "$ARCHIVE_SHA" "$LOG_TMP"; do
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
  for source in "${SOURCE_DIRECTORIES[@]}"; do
    (
      cd "$source"
      sha256sum -c SHA256SUMS
    )
  done
  "$PYTHON" - <<'PY'
import importlib.metadata
import torch

assert torch.__version__ == "2.6.0+cpu", torch.__version__
assert torch.cuda.is_available() is False
assert importlib.metadata.version("motornet") == "0.2.0"
print("SERVER_RUNTIME_PREFLIGHT=1")
PY
  "$PYTHON" -m unittest \
    tests.test_hanzi_canonical_start_loss_comparison \
    tests.test_hanzi_canonical_shugou_checkpoint_experiment

  "$PYTHON" -m hanzi_writing.canonical_start_loss_comparison \
    --config "$CONFIG" prepare
  echo "CANONICAL_EXISTING_BASELINE_REUSED=1"
  echo "CANONICAL_EXISTING_BASELINE_RETRAINED=0"
  echo "CANONICAL_START_LOSS_COMPARISON_WORKER_COUNT=16"
  echo "MOVE_TRAINING_STARTED=0"
  echo "SHARED_8TASK_STARTED=0"
  echo "SHARED_9TASK_STARTED=0"

  mkdir "$OUTPUT/worker_logs"
  WORKER_PIDS=()
  WORKER_LABELS=()
  terminate_workers() {
    for pid in "${WORKER_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap terminate_workers INT TERM EXIT
  for arm in "${ARMS[@]}"; do
    for task in "${TASKS[@]}"; do
      label="${arm}_${task}"
      "$PYTHON" -m hanzi_writing.canonical_start_loss_comparison \
        --config "$CONFIG" train-worker --arm "$arm" --task "$task" \
        > "$OUTPUT/worker_logs/$label.log" 2>&1 &
      WORKER_PIDS+=("$!")
      WORKER_LABELS+=("$label")
      echo "CANONICAL_START_LOSS_WORKER_LAUNCHED=$label PID=$!"
    done
  done
  test "${#WORKER_PIDS[@]}" -eq 16
  WORKER_FAILURE=0
  for index in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$index]}"
    label="${WORKER_LABELS[$index]}"
    if wait "$pid"; then
      echo "CANONICAL_START_LOSS_WORKER_COMPLETE=$label PID=$pid"
    else
      status="$?"
      echo "CANONICAL_START_LOSS_WORKER_FAILED=$label PID=$pid EXIT=$status" >&2
      WORKER_FAILURE=1
    fi
  done
  trap - INT TERM EXIT
  test "$WORKER_FAILURE" -eq 0

  "$PYTHON" -m hanzi_writing.canonical_start_loss_comparison \
    --config "$CONFIG" finalize
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Canonical start-loss comparison stopped; see $LOG_TMP and worker logs." >&2
  exit 1
fi

TEST_COUNT="$(sed -nE 's/^Ran ([0-9]+) tests? in .*/\1/p' "$LOG_TMP" | tail -n 1)"
test "$TEST_COUNT" = 10
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi
test -d "$OUTPUT"
cp "$LOG_TMP" "$OUTPUT/execution.log"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$OUTPUT/exit_code.txt"

OUTPUT="$OUTPUT" TEST_COUNT="$TEST_COUNT" PROGRAM_EXIT="$PROGRAM_EXIT" \
TEE_EXIT="$TEE_EXIT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT"])
path = output / "provenance.json"
with path.open(encoding="utf-8") as handle:
    provenance = json.load(handle)
with (output / "baseline_selection.json").open(encoding="utf-8") as handle:
    baseline = json.load(handle)
assert baseline["shugou_control_extension"]["control_candidate_rows"] == 21
assert baseline["shugou_control_extension"]["selected_candidate"] == (
    "control:review:u1999"
)
assert provenance["completed"] is True
assert provenance["variant"] == "canonical_start_loss_comparison_v1"
assert provenance["baseline_retrained"] is False
assert provenance["champions"]["baseline:shugou"]["candidate_id"] == (
    "control:review:u1999"
)
assert provenance["parallel_workers"] == 16
assert provenance["integrity"] == {
    "all_numeric_values_finite": True,
    "baseline_selected_tasks": 8,
    "champion_rows": 24,
    "new_candidate_rows": 736,
    "parallel_workers_completed": 16,
    "plots": 8,
}
assert provenance["automatic_checkpoint_selection_performed"] is True
assert provenance["behavioral_pass_fail_defined"] is False
assert provenance["training_started"] is True
assert provenance["move_started"] is False
assert provenance["shared_8task_started"] is False
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
with path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

(
  cd "$OUTPUT"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$OUTPUT")" -czf "$ARCHIVE" "$(basename "$OUTPUT")"
sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"
rm "$LOG_TMP"

echo "CANONICAL_START_LOSS_COMPARISON_COMPLETE=1"
echo "AUTOMATIC_CHECKPOINT_SELECTION_PERFORMED=1"
echo "BEHAVIORAL_PASS_FAIL_DEFINED=0"
echo "MOVE_TRAINING_STARTED=0"
echo "SHARED_8TASK_STARTED=0"
echo "SHARED_9TASK_STARTED=0"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "COMPLETE_CHARACTER_ROLLOUT_STARTED=0"
echo "OUTPUT=$OUTPUT"
echo "ARCHIVE=$ARCHIVE"
