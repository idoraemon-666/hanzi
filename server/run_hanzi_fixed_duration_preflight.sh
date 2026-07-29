#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_hanzi_fixed_duration_preflight.sh REPO OUTPUT_ROOT LEGACY_REFERENCE_DIR EXPECTED_HEAD" >&2
  exit 2
fi

REPO="$(realpath "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"
LEGACY_REFERENCE_DIR="$(realpath "$3")"
EXPECTED_HEAD="$4"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
COVERAGE_CONFIG=configurations/hanzi_stroke_temporal_composition_fixed_duration_coverage_v1.json
SMOKE_CONFIG=configurations/hanzi_stroke_temporal_composition_fixed_duration_smoke_v1.json
LEGACY_GEOMETRY=configurations/hanzi_stroke_temporal_composition_geometry.json

test -d "$REPO/.git"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
test -f "$LEGACY_REFERENCE_DIR/legacy_reference.json"
test -f "$LEGACY_REFERENCE_DIR/legacy_reference.npz"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ABORT: refusing to reuse preflight output root $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT"
WORK="$(mktemp -d /root/autodl-tmp/hanzi-fixed-duration-preflight.XXXXXX)"
case "$WORK" in
  /root/autodl-tmp/hanzi-fixed-duration-preflight.*) ;;
  *) echo "ABORT: unexpected temporary path $WORK" >&2; exit 1 ;;
esac
trap 'rm -rf -- "$WORK"' EXIT
cp "$LEGACY_REFERENCE_DIR/legacy_reference.json" "$OUTPUT_ROOT/legacy_reference.json"
cp "$LEGACY_REFERENCE_DIR/legacy_reference.npz" "$OUTPUT_ROOT/legacy_reference.npz"

set +e
(
  set -euo pipefail
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES=''
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONHASHSEED=0
  "$PYTHON" -m unittest discover -s tests
  "$PYTHON" -m hanzi_writing.fixed_duration_preflight audit \
    --config "$COVERAGE_CONFIG" \
    --legacy-geometry "$LEGACY_GEOMETRY" \
    --legacy-json "$OUTPUT_ROOT/legacy_reference.json" \
    --legacy-npz "$OUTPUT_ROOT/legacy_reference.npz" \
    --output-dir "$WORK/audit"
  cp "$WORK/audit/fixed_duration_condition_manifest.csv" \
    "$OUTPUT_ROOT/fixed_duration_condition_manifest.csv"
  "$PYTHON" -m hanzi_writing.fixed_duration_preflight coverage \
    --config "$COVERAGE_CONFIG" \
    --output-dir "$WORK/coverage"
  "$PYTHON" -m hanzi_writing.fixed_duration_preflight smoke \
    --config "$SMOKE_CONFIG" \
    --output-dir "$WORK/smoke" \
    --state "$WORK/technical_smoke_state.pt"
  "$PYTHON" -m hanzi_writing.fixed_duration_preflight validation \
    --config "$SMOKE_CONFIG" \
    --state "$WORK/technical_smoke_state.pt" \
    --output-dir "$WORK/validation"
  "$PYTHON" -m hanzi_writing.fixed_duration_preflight finalize \
    --repo "$REPO" \
    --config "$REPO/$SMOKE_CONFIG" \
    --audit-json "$WORK/audit/target_audit.json" \
    --coverage-json "$WORK/coverage/coverage_result.json" \
    --smoke-json "$WORK/smoke/smoke_result.json" \
    --validation-json "$WORK/validation/validation_result.json" \
    --legacy-json "$OUTPUT_ROOT/legacy_reference.json" \
    --output-root "$OUTPUT_ROOT"
) 2>&1 | tee "$OUTPUT_ROOT/execution.log"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${RUN_CODES[0]}" > "$OUTPUT_ROOT/exit_code.txt"
test "${RUN_CODES[0]}" -eq 0
test "${RUN_CODES[1]}" -eq 0

EXPECTED_FILES=(
  PREFLIGHT_REPORT.md
  execution.log
  exit_code.txt
  fixed_duration_condition_manifest.csv
  legacy_reference.json
  legacy_reference.npz
  preflight_summary.json
  provenance.json
)
for name in "${EXPECTED_FILES[@]}"; do
  test -f "$OUTPUT_ROOT/$name"
done
test "$(find "$OUTPUT_ROOT" -maxdepth 1 -type f | wc -l)" -eq 8
(
  cd "$OUTPUT_ROOT"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
test "$(find "$OUTPUT_ROOT" -maxdepth 1 -type f | wc -l)" -eq 9
echo "FORMAL_10K_PILOT_STARTED=0"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "FIXED_DURATION_PREFLIGHT_COMPLETE=1"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
