#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_stroke_temporal_composition_validate_characters.sh REPO RUN_ROOT EXPECTED_PARENT_HEAD" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "mu-jiang-ke-frozen" ]]; then
  echo "STOP: explicit authorization for mu-jiang-ke-frozen is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
EXPECTED_PARENT_HEAD="$3"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_validate_characters.json
CONFIG="$REPO/$CONFIG_RELATIVE"
SOURCE="$REPO/runs/hanzi_stroke_temporal_composition/train/dev42/best_checkpoint.pt"
OUTPUT_DIR="$REPO/runs/hanzi_stroke_temporal_composition/character_validation/dev42"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$CONFIG"
test -f "$SOURCE"
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
sha256sum "$SOURCE" > "$RUN_ROOT/source_checkpoint.sha256"
set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" config.py --protocol_config "$CONFIG_RELATIVE"
) 2>&1 | tee "$RUN_ROOT/validation.log"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
printf 'RUN_EXIT=%s\nTEE_EXIT=%s\n' "${RUN_CODES[0]}" "${RUN_CODES[1]}" \
  > "$RUN_ROOT/exit_codes.txt"
test "${RUN_CODES[0]}" -eq 0
test "${RUN_CODES[1]}" -eq 0
"$PYTHON" - <<PY
import json
report = json.load(open("$OUTPUT_DIR/frozen_character_validation.json"))
assert report["overall_passed"] is True
assert report["policy_state_bitwise_unchanged"] is True
assert report["optimizer_created"] is False
assert set(report["characters"]) == {"mu", "jiang", "ke"}
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
echo "FROZEN_CHARACTER_VALIDATION_COMPLETE=1"
echo "RUN_ROOT=$RUN_ROOT"
