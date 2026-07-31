#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_dual_fixed_rule_joint_gradient.sh REPO EXPECTED_HEAD ARCHIVE_BASE" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "dual-fixed-rule-joint-gradient-loss-comparison-v1" ]]; then
  echo "STOP: explicit authorization for the joint-gradient experiment is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
EXPECTED_HEAD="$2"
ARCHIVE_BASE="$(realpath -m "$3")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
CONFIG=configurations/hanzi_stroke_temporal_composition_dual_fixed_rule_joint_gradient_v1.json
OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/dual_fixed_rule_joint_gradient_loss_comparison/dev42"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"
MODEL_KINDS=(stroke move)
ARMS=(baseline full_trial onset_window)

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$REPO/$CONFIG"
test "$(git -C "$REPO" branch --show-current)" = "codex/hanzi-stroke-temporal-composition"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
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
  export OMP_THREAD_LIMIT=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONHASHSEED=0

  "$PYTHON" - <<'PY'
import importlib.metadata
import torch

assert torch.__version__ == "2.6.0+cpu", torch.__version__
assert torch.cuda.is_available() is False
assert importlib.metadata.version("motornet") == "0.2.0"
print("SERVER_RUNTIME_PREFLIGHT=1")
PY
  "$PYTHON" -m unittest discover -s tests -p 'test_hanzi*.py'
  "$PYTHON" -m unittest tests.test_phase_normalized_loss

  "$PYTHON" -m hanzi_writing.dual_rule_training --config "$CONFIG" prepare
  printf '%s\n' "$(git rev-parse HEAD)" > "$OUTPUT/git_head.txt"
  printf '%s\n' "$(git -C mRNNTorch rev-parse HEAD)" > "$OUTPUT/submodule_head.txt"
  "$PYTHON" -m pip freeze > "$OUTPUT/environment.freeze.txt"
  {
    lscpu
    echo "logical_cpus=$(nproc)"
  } > "$OUTPUT/cpu_info.txt"

  echo "DUAL_FIXED_RULE_JOINT_WORKER_COUNT=6"
  echo "OPTIMIZER_STEP_MODE=all_rules_mean_gradient"
  echo "STROKE_RULES_PER_STEP=15"
  echo "MOVE_RULES_PER_STEP=12"
  echo "MICROBATCH_SIZE_PER_RULE=1"
  echo "RULE_EXPOSURES_PER_RULE=8000"
  echo "STROKE_OPTIMIZER_STEPS=8000"
  echo "MOVE_OPTIMIZER_STEPS=8000"
  echo "FIXED_DELAY_STEPS=50"
  echo "COMPLETE_CHARACTER_STARTED=0"
  echo "CHAINED_CONTROLLER_STARTED=0"

  WORKER_PIDS=()
  WORKER_LABELS=()
  terminate_workers() {
    for pid in "${WORKER_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap terminate_workers INT TERM EXIT
  for model_kind in "${MODEL_KINDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      label="${model_kind}_${arm}"
      (
        start_epoch="$(date +%s)"
        set +e
        "$PYTHON" -m hanzi_writing.dual_rule_training \
          --config "$CONFIG" train-worker \
          --model-kind "$model_kind" --arm "$arm" \
          > "$OUTPUT/worker_logs/$label.log" 2>&1
        status="$?"
        set -e
        end_epoch="$(date +%s)"
        printf 'start_epoch=%s\nend_epoch=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
          "$start_epoch" "$end_epoch" "$((end_epoch - start_epoch))" "$status" \
          > "$OUTPUT/worker_logs/$label.runtime.txt"
        exit "$status"
      ) &
      WORKER_PIDS+=("$!")
      WORKER_LABELS+=("$label")
      echo "DUAL_FIXED_RULE_JOINT_WORKER_LAUNCHED=$label PID=$!"
    done
  done
  test "${#WORKER_PIDS[@]}" -eq 6
  WORKER_FAILURE=0
  for index in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$index]}"
    label="${WORKER_LABELS[$index]}"
    if wait "$pid"; then
      echo "DUAL_FIXED_RULE_JOINT_WORKER_COMPLETE=$label PID=$pid"
    else
      status="$?"
      echo "DUAL_FIXED_RULE_JOINT_WORKER_FAILED=$label PID=$pid EXIT=$status" >&2
      WORKER_FAILURE=1
    fi
  done
  trap - INT TERM EXIT
  test "$WORKER_FAILURE" -eq 0

  "$PYTHON" -m hanzi_writing.dual_rule_training --config "$CONFIG" finalize
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Joint-gradient experiment stopped; see $LOG_TMP and worker logs." >&2
  exit 1
fi
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi

test -d "$OUTPUT"
cp "$LOG_TMP" "$OUTPUT/execution.log"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$OUTPUT/exit_code.txt"

OUTPUT="$OUTPUT" PROGRAM_EXIT="$PROGRAM_EXIT" TEE_EXIT="$TEE_EXIT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT"])
path = output / "provenance.json"
with path.open(encoding="utf-8") as handle:
    provenance = json.load(handle)
assert provenance["completed"] is True
assert provenance["variant"] == "dual_fixed_rule_joint_gradient_loss_comparison_v1"
assert provenance["parallel_workers"] == 6
assert provenance["models"] == ["stroke", "move"]
assert provenance["loss_arms"] == ["baseline", "full_trial", "onset_window"]
assert provenance["fixed_delay_steps"] == 50
assert provenance["microbatch_size_per_rule"] == 1
assert provenance["optimizer_step_mode"] == "all_rules_mean_gradient"
assert provenance["effective_rules_per_optimizer_step"] == {"stroke": 15, "move": 12}
assert provenance["optimizer_steps_per_model"] == {"stroke": 8000, "move": 8000}
assert provenance["rule_exposures_per_rule"] == 8000
assert provenance["integrity"] == {
    "all_numeric_values_finite": True,
    "candidate_metric_rows": 243,
    "plots": 18,
    "workers_completed": 6,
}
assert provenance["automatic_checkpoint_selection_performed"] is True
assert provenance["behavioral_pass_fail_defined"] is False
assert provenance["complete_character_started"] is False
assert provenance["chained_controller_started"] is False
provenance.update(
    {
        "program_exit_code": int(os.environ["PROGRAM_EXIT"]),
        "tee_exit_code": int(os.environ["TEE_EXIT"]),
        "skipped_test_count": 0,
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

echo "DUAL_FIXED_RULE_JOINT_GRADIENT_COMPLETE=1"
echo "AUTOMATIC_CHECKPOINT_SELECTION_PERFORMED=1"
echo "BEHAVIORAL_PASS_FAIL_DEFINED=0"
echo "COMPLETE_CHARACTER_STARTED=0"
echo "CHAINED_CONTROLLER_STARTED=0"
echo "OUTPUT=$OUTPUT"
echo "ARCHIVE=$ARCHIVE"
