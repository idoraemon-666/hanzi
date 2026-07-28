#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_stroke_temporal_composition_audit.sh REPO RUN_ROOT EXPECTED_PARENT_HEAD" >&2
  exit 2
fi

REPO="$(realpath "$1")"
RUN_ROOT="$(realpath -m "$2")"
EXPECTED_PARENT_HEAD="$3"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
ENV_MARKER=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/.hanzi_cpu_environment_complete
BASELINE=f76067232fac8add756b04ea75a8825cca156330
SUBMODULE=ac0c4f589eae37bbde63968912925de99232e306
GEOMETRY_CONFIG="$REPO/configurations/hanzi_stroke_temporal_composition_geometry.json"
TRAINING_CONFIG="$REPO/configurations/hanzi_stroke_temporal_composition_train_dev42.json"
ARCHIVE="${RUN_ROOT}.tar.gz"
ARCHIVE_HASH="${ARCHIVE}.sha256"

test -d "$REPO/.git"
test -f "$GEOMETRY_CONFIG"
test -f "$TRAINING_CONFIG"
for path in "$RUN_ROOT" "$ARCHIVE" "$ARCHIVE_HASH"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: output already exists: $path" >&2
    exit 1
  fi
done

bash "$REPO/server/create_hanzi_cpu_environment.sh" "$REPO"
test -x "$PYTHON"
test -f "$ENV_MARKER"
grep -qx 'device=cpu' "$ENV_MARKER"

PARENT_HEAD="$(git -C "$REPO" rev-parse HEAD)"
SUBMODULE_HEAD="$(git -C "$REPO/mRNNTorch" rev-parse HEAD)"
test "$PARENT_HEAD" = "$EXPECTED_PARENT_HEAD"
test "$SUBMODULE_HEAD" = "$SUBMODULE"
git -C "$REPO" merge-base --is-ancestor "$BASELINE" "$PARENT_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"

mkdir -p "$RUN_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/start_utc.txt"
printf '%s\n' "$PARENT_HEAD" > "$RUN_ROOT/parent_head.txt"
printf '%s\n' "$SUBMODULE_HEAD" > "$RUN_ROOT/submodule_head.txt"
printf '%s\n' "$BASELINE" > "$RUN_ROOT/project2_baseline.txt"
cp "$GEOMETRY_CONFIG" "$RUN_ROOT/geometry_configuration.json"
cp "$TRAINING_CONFIG" "$RUN_ROOT/training_configuration.json"
sha256sum "$RUN_ROOT/geometry_configuration.json" "$RUN_ROOT/training_configuration.json" \
  > "$RUN_ROOT/configuration.sha256"
(
  cd "$REPO"
  git ls-files | while IFS= read -r path; do
    if [[ -f "$path" ]]; then
      sha256sum "$path"
    fi
  done
) > "$RUN_ROOT/repository_tracked_sha256.txt"
{
  printf 'PROJECT=hanzi_stroke_temporal_composition\n'
  printf 'DEVICE=cpu\n'
  printf 'PARENT_HEAD=%s\n' "$PARENT_HEAD"
  printf 'SUBMODULE_HEAD=%s\n' "$SUBMODULE_HEAD"
  df -h /root/autodl-tmp
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
  fi
  "$PYTHON" --version
  "$PYTHON" -m pip check
  "$PYTHON" -c "import motornet,numpy,torch; print(torch.__version__,motornet.__version__,numpy.__version__,torch.cuda.is_available())"
} 2>&1 | tee "$RUN_ROOT/preflight.log"
"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -m unittest discover -s tests -v
) 2>&1 | tee "$RUN_ROOT/tests.log"
TEST_CODES=("${PIPESTATUS[@]}")
set -e
printf 'TEST_EXIT=%s\nTEST_TEE_EXIT=%s\n' "${TEST_CODES[0]}" "${TEST_CODES[1]}" \
  > "$RUN_ROOT/test_exit_codes.txt"
test "${TEST_CODES[0]}" -eq 0
test "${TEST_CODES[1]}" -eq 0
grep -Eq '^Ran [1-9][0-9]* tests in ' "$RUN_ROOT/tests.log"
grep -q '^OK$' "$RUN_ROOT/tests.log"
if grep -q 'skipped=' "$RUN_ROOT/tests.log"; then
  echo "ABORT: required server tests contain skipped tests." >&2
  exit 1
fi

set +e
(
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -m hanzi_writing.audit \
      --geometry-config "$GEOMETRY_CONFIG" \
      --training-config "$TRAINING_CONFIG" \
      --output-dir "$RUN_ROOT/audit"
) 2>&1 | tee "$RUN_ROOT/audit.log"
AUDIT_CODES=("${PIPESTATUS[@]}")
set -e
printf 'AUDIT_EXIT=%s\nAUDIT_TEE_EXIT=%s\n' "${AUDIT_CODES[0]}" "${AUDIT_CODES[1]}" \
  > "$RUN_ROOT/audit_exit_codes.txt"
test "${AUDIT_CODES[0]}" -eq 0
test "${AUDIT_CODES[1]}" -eq 0
"$PYTHON" - <<PY
import json
from pathlib import Path

report = json.loads(Path("$RUN_ROOT/audit/audit_report.json").read_text())
assert report["overall_passed"] is True
assert report["formal_training_started"] is False
assert report["sampler_and_checkpoint_grid"]["checkpoint_rollout_group_count_three_speeds"] == 81
assert report["frozen_character_smoke"]["network_state_dict_bitwise_unchanged"] is True
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/end_utc.txt"
(
  cd "$RUN_ROOT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")"
(
  cd "$(dirname "$ARCHIVE")"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE_HASH")"
)
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
echo "PARENT_HEAD=$PARENT_HEAD"
echo "RUN_ROOT=$RUN_ROOT"
echo "ARCHIVE=$ARCHIVE"
echo "ARCHIVE_HASH=$ARCHIVE_HASH"
echo "FORMAL_TRAINING_STARTED=0"
echo "HANZI_AUDIT_COMPLETE=1"
