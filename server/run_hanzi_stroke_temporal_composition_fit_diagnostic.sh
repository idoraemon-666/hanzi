#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_stroke_temporal_composition_fit_diagnostic.sh REPO RUN_ROOT EXPECTED_PARENT_HEAD" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "best-final-fit-diagnostic" ]]; then
  echo "STOP: explicit authorization for best-final-fit-diagnostic is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
EXPECTED_PARENT_HEAD="$3"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_fit_diagnostic.json
CONFIG="$REPO/$CONFIG_RELATIVE"
OUTPUT_DIR="$REPO/runs/hanzi_stroke_temporal_composition/fit_diagnostic/dev42_best_final"
BEST="$REPO/runs/hanzi_stroke_temporal_composition/train/dev42/best_checkpoint.pt"
FINAL="$REPO/runs/hanzi_stroke_temporal_composition/train/dev42/final_continuation_checkpoint.pt"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$CONFIG"
test -f "$BEST"
test -f "$FINAL"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_PARENT_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in "$RUN_ROOT" "$OUTPUT_DIR" "${RUN_ROOT}.tar.gz" "${RUN_ROOT}.tar.gz.sha256"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: refusing to overwrite $path" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/start_utc.txt"
cp "$CONFIG" "$RUN_ROOT/configuration.json"
sha256sum "$BEST" "$FINAL" > "$RUN_ROOT/source_checkpoints.sha256"
set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -m hanzi_writing.fit_diagnostic --config "$CONFIG_RELATIVE"
) 2>&1 | tee "$RUN_ROOT/diagnostic.log"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
printf 'RUN_EXIT=%s\nTEE_EXIT=%s\n' "${RUN_CODES[0]}" "${RUN_CODES[1]}" \
  > "$RUN_ROOT/exit_codes.txt"
test "${RUN_CODES[0]}" -eq 0
test "${RUN_CODES[1]}" -eq 0
"$PYTHON" - <<PY
import json
report = json.load(open("$OUTPUT_DIR/fit_diagnostic_summary.json"))
assert report["project"] == "hanzi_stroke_temporal_composition"
assert report["diagnostic_only"] is True
assert report["formal_training_started"] is False
assert report["optimizer_created"] is False
assert report["behavioral_pass_fail_defined"] is False
assert report["path_completion_metric_defined"] is False
assert report["environment_observation_and_action_noise_verified_zero"] is True
assert report["condition_grid"]["primitive_count"] == 15
assert report["condition_grid"]["exact_stroke_placement_count"] == 46
assert report["condition_grid"]["exact_move_count"] == 12
assert report["condition_grid"]["metric_row_count"] == 348
assert report["condition_grid"]["rollout_group_count"] == 162
assert set(report["checkpoints"]) == {"best", "final"}
assert all(value["policy_state_bitwise_unchanged"] for value in report["checkpoints"].values())
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/end_utc.txt"
cp -a "$OUTPUT_DIR" "$RUN_ROOT/output"
(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$RUN_ROOT")" -czf "${RUN_ROOT}.tar.gz" "$(basename "$RUN_ROOT")"
sha256sum "${RUN_ROOT}.tar.gz" > "${RUN_ROOT}.tar.gz.sha256"
echo "FORMAL_TRAINING_STARTED=0"
echo "HANZI_FIT_DIAGNOSTIC_COMPLETE=1"
echo "RUN_ROOT=$RUN_ROOT"
