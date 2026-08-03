#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_hanzi_frozen_dual_controller_continuous_state_composition.sh REPO EXPECTED_HEAD SOURCE_RESULTS ARCHIVE_BASE" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "frozen-dual-controller-continuous-physical-state-v1" ]]; then
  echo "STOP: explicit authorization for continuous physical-state composition is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
EXPECTED_HEAD="$2"
SOURCE_RESULTS="$(realpath "$3")"
ARCHIVE_BASE="$(realpath -m "$4")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
CONFIG=configurations/hanzi_stroke_temporal_composition_frozen_dual_controller_continuous_state_v1.json
OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/frozen_dual_controller_continuous_physical_state/dev42"
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

  "$PYTHON" -m hanzi_writing.dual_controller_continuous_state_composition \
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
  echo "CHARACTER_COUNT=3"
  echo "TRIAL_COUNT=27"
  echo "CONTINUOUS_PHYSICAL_STATE_BOUNDARIES=24"
  echo "CHARACTER_INITIAL_EFFECTOR_RESETS=3"
  echo "INTERTRIAL_EFFECTOR_RESETS=0"
  echo "PHYSICAL_STATE_BITWISE_CONTINUITY=1"
  echo "RNN_STATE_RESET_EACH_TRIAL=1"
  echo "FULL_TRIAL_RENDER=1"
  echo "TRAINING_STARTED=0"
  echo "TRAJECTORY_POSTPROCESSING_PERFORMED=0"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Continuous physical-state composition stopped; see $LOG_TMP." >&2
  exit 1
fi
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi

test -d "$OUTPUT"
test -z "$(git -C "$REPO" status --short)"
for plot in \
  mu_continuous_physical_state.png \
  jiang_continuous_physical_state.png \
  ke_continuous_physical_state.png \
  three_characters_continuous_physical_state.png; do
  test -f "$OUTPUT/plots/$plot"
done
test -f "$OUTPUT/physical_state_boundary_audit.json"
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
assert provenance["variant"] == "frozen_joint_gradient_onset_window_continuous_physical_state_v1"
assert provenance["selected_loss_arms"] == {"stroke": "onset_window", "move": "onset_window"}
assert provenance["policy_state_bitwise_unchanged"] == {"stroke": True, "move": True}
assert provenance["training_started"] is False
assert provenance["optimizer_created"] is False
assert provenance["backward_executed"] is False
assert provenance["trajectory_postprocessing_performed"] is False
assert provenance["render_contract"]["trajectory"] == "actual_complete_trial_including_physical_start_state"
integrity = provenance["integrity"]
assert integrity["total_trials"] == 27
assert integrity["stroke_trials"] == 15
assert integrity["move_trials"] == 12
assert integrity["character_initial_effector_resets"] == 3
assert integrity["intertrial_effector_resets"] == 0
assert integrity["continuous_physical_state_boundaries"] == 24
assert integrity["physical_state_bitwise_continuity"] is True
assert integrity["maximum_physical_state_boundary_difference"] == 0.0
assert integrity["shared_effector_instance"] is True
assert integrity["full_trial_render_verified"] is True
assert integrity["physical_state_keys"] == ["joint", "cartesian", "muscle", "geometry"]
assert integrity["plots"] == 4
assert integrity["boundary_audit_rows"] == 24
assert integrity["all_numeric_values_finite"] is True
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

echo "FROZEN_DUAL_CONTROLLER_CONTINUOUS_STATE_COMPLETE=1"
echo "PHYSICAL_STATE_BITWISE_CONTINUITY=1"
echo "INTERTRIAL_EFFECTOR_RESETS=0"
echo "FULL_TRIAL_RENDER=1"
echo "TRAINING_STARTED=0"
echo "TRAJECTORY_POSTPROCESSING_PERFORMED=0"
echo "OUTPUT=$OUTPUT"
echo "ARCHIVE=$ARCHIVE"
