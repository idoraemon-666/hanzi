#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_stroke_temporal_composition_train_dev42.sh REPO RUN_ROOT EXPECTED_PARENT_HEAD" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "shared9-dev42-75000" ]]; then
  echo "STOP: explicit authorization for shared9-dev42-75000 is required." >&2
  exit 2
fi
if [[ -z "${HANZI_AUDIT_REPORT:-}" || ! -f "$HANZI_AUDIT_REPORT" ]]; then
  echo "STOP: HANZI_AUDIT_REPORT must name a completed passing audit_report.json." >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
EXPECTED_PARENT_HEAD="$3"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_train_dev42.json
CONFIG="$REPO/$CONFIG_RELATIVE"
OUTPUT_DIR="$REPO/runs/hanzi_stroke_temporal_composition/train/dev42"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$CONFIG"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_PARENT_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in "$RUN_ROOT" "$OUTPUT_DIR" "${RUN_ROOT}.tar.gz" "${RUN_ROOT}.tar.gz.sha256"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: refusing to overwrite $path" >&2
    exit 1
  fi
done
"$PYTHON" - <<PY
import json
report = json.load(open("$HANZI_AUDIT_REPORT"))
assert report["overall_passed"] is True
assert report["formal_training_started"] is False
PY

mkdir -p "$RUN_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/start_utc.txt"
cp "$CONFIG" "$RUN_ROOT/configuration.json"
sha256sum "$RUN_ROOT/configuration.json" > "$RUN_ROOT/configuration.json.sha256"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/parent_head.txt"
git -C "$REPO/mRNNTorch" rev-parse HEAD > "$RUN_ROOT/submodule_head.txt"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" config.py --protocol_config "$CONFIG_RELATIVE"
) 2>&1 | tee "$RUN_ROOT/training.log"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
printf 'RUN_EXIT=%s\nTEE_EXIT=%s\n' "${RUN_CODES[0]}" "${RUN_CODES[1]}" \
  > "$RUN_ROOT/exit_codes.txt"
test "${RUN_CODES[0]}" -eq 0
test "${RUN_CODES[1]}" -eq 0
test -f "$OUTPUT_DIR/best_checkpoint.pt"
test -f "$OUTPUT_DIR/final_continuation_checkpoint.pt"
"$PYTHON" - <<PY
import json
summary = json.load(open("$OUTPUT_DIR/training_summary.json"))
assert summary["updates"] == 75000
assert summary["project"] == "hanzi_stroke_temporal_composition"
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
echo "FORMAL_TRAINING_COMPLETE=1"
echo "RUN_ROOT=$RUN_ROOT"
