#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_hanzi_dual_rule_canonical_validation.sh REPO EXPECTED_HEAD SOURCE_RESULTS ARCHIVE_BASE" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "frozen-dual-rule-canonical-validation-v1" ]]; then
  echo "STOP: explicit authorization for frozen dual-rule canonical validation is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
EXPECTED_HEAD="$2"
SOURCE_RESULTS="$(realpath "$3")"
ARCHIVE_BASE="$(realpath -m "$4")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
CONFIG=configurations/hanzi_stroke_temporal_composition_dual_rule_canonical_validation_v1.json
OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/frozen_dual_rule_canonical_validation/dev42"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$REPO/$CONFIG"
test -f "$SOURCE_RESULTS/provenance.json"
test -f "$SOURCE_RESULTS/SHA256SUMS"
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

  "$PYTHON" -m hanzi_writing.dual_rule_canonical_validation \
    --config "$CONFIG" --source-results "$SOURCE_RESULTS"
  printf '%s\n' "$(git rev-parse HEAD)" > "$OUTPUT/git_head.txt"
  printf '%s\n' "$(git -C mRNNTorch rev-parse HEAD)" > "$OUTPUT/submodule_head.txt"
  "$PYTHON" -m pip freeze > "$OUTPUT/environment.freeze.txt"
  {
    lscpu
    echo "logical_cpus=$(nproc)"
  } > "$OUTPUT/cpu_info.txt"

  echo "FROZEN_STROKE_CHECKPOINT=joint_gradient/onset_window/best_macro"
  echo "FROZEN_MOVE_CHECKPOINT=joint_gradient/onset_window/best_macro"
  echo "INDEPENDENT_CANONICAL_ROLLOUTS=27"
  echo "SOURCE_MOVEMENT_EXACT_MATCH=1"
  echo "FULL_TRIAL_RENDER=1"
  echo "TRAINING_STARTED=0"
  echo "TRAJECTORY_POSTPROCESSING_PERFORMED=0"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Frozen dual-rule canonical validation stopped; see $LOG_TMP." >&2
  exit 1
fi
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi

test -d "$OUTPUT"
test -z "$(git -C "$REPO" status --short)"
for plot in \
  stroke_movement_trajectories.png \
  stroke_full_trial_trajectories.png \
  move_movement_trajectories.png \
  move_full_trial_trajectories.png; do
  test -f "$OUTPUT/plots/$plot"
done
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
assert provenance["variant"] == "frozen_joint_gradient_onset_window_canonical_validation_full_trial_v1"
assert provenance["selected_loss_arms"] == {"stroke": "onset_window", "move": "onset_window"}
assert provenance["selected_candidate"] == "best_macro"
assert provenance["policy_state_bitwise_unchanged"] == {"stroke": True, "move": True}
assert provenance["source_movement_arrays_matched_exactly"] is True
assert provenance["training_started"] is False
assert provenance["optimizer_created"] is False
assert provenance["backward_executed"] is False
assert provenance["trajectory_postprocessing_performed"] is False
assert provenance["validation_contract"]["independent_rollout_per_rule"] is True
assert provenance["validation_contract"]["canonical_start_state"] is True
assert provenance["validation_contract"]["cross_task_state_carryover"] is False
assert provenance["validation_contract"]["effector_state_reset_each_rule"] == "canonical_standard_state"
assert provenance["validation_contract"]["recurrent_state_reset_each_rule"] is True
assert provenance["validation_contract"]["feedback_buffers_reset_each_rule"] is True
assert provenance["integrity"]["total_tasks"] == 27
assert provenance["integrity"]["stroke_tasks"] == 15
assert provenance["integrity"]["move_tasks"] == 12
assert provenance["integrity"]["source_movement_arrays_matched_exactly"] == 27
assert provenance["integrity"]["plots"] == 4
assert provenance["integrity"]["all_numeric_values_finite"] is True
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

echo "FROZEN_DUAL_RULE_CANONICAL_VALIDATION_COMPLETE=1"
echo "SOURCE_MOVEMENT_EXACT_MATCH=1"
echo "FULL_TRIAL_RENDER=1"
echo "TRAINING_STARTED=0"
echo "TRAJECTORY_POSTPROCESSING_PERFORMED=0"
echo "OUTPUT=$OUTPUT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_SHA=$ARCHIVE_SHA"
