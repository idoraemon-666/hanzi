#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_canonical_checkpoint_metric_review.sh REPO EXPECTED_HEAD ARCHIVE_BASE" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "canonical-checkpoint-metric-review" ]]; then
  echo "STOP: explicit authorization for the canonical checkpoint metric review is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
EXPECTED_HEAD="$2"
ARCHIVE_BASE="$(realpath -m "$3")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
CONFIG=configurations/hanzi_stroke_temporal_composition_canonical_checkpoint_metric_review_v1.json
OUTPUT="$REPO/runs/hanzi_stroke_temporal_composition/canonical_checkpoint_metric_review/dev42"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"

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
  export MKL_NUM_THREADS=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONHASHSEED=0
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py'
  "$PYTHON" -m hanzi_writing.canonical_checkpoint_review --config "$CONFIG"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Canonical checkpoint metric review stopped; see $LOG_TMP." >&2
  exit 1
fi

TEST_COUNT="$(sed -nE 's/^Ran ([0-9]+) tests? in .*/\1/p' "$LOG_TMP" | tail -n 1)"
test "$TEST_COUNT" = 108
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
assert provenance["completed"] is True
assert provenance["variant"] == "canonical_checkpoint_metric_review_v1"
assert provenance["integrity"] == {
    "all_numeric_values_finite": True,
    "candidate_rows": 56,
    "leader_rows": 58,
    "source_count": 4,
    "task_count": 8,
}
assert provenance["behavioral_pass_fail_defined"] is False
assert provenance["automatic_checkpoint_selection_performed"] is False
assert provenance["manual_review_required"] is True
assert provenance["training_started"] is False
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

echo "CANONICAL_CHECKPOINT_METRIC_REVIEW_COMPLETE=1"
echo "BEHAVIORAL_PASS_FAIL_DEFINED=0"
echo "AUTOMATIC_CHECKPOINT_SELECTION_PERFORMED=0"
echo "MANUAL_REVIEW_REQUIRED=1"
echo "TRAINING_STARTED=0"
echo "MOVE_TRAINING_STARTED=0"
echo "SHARED_9TASK_STARTED=0"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "COMPLETE_CHARACTER_ROLLOUT_STARTED=0"
echo "OUTPUT=$OUTPUT"
echo "ARCHIVE=$ARCHIVE"
