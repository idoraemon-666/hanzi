#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash server/run_hanzi_stroke_temporal_composition_boundary_intervention.sh REPO ARCHIVE_BASE EXPECTED_HEAD" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "boundary-state-block-intervention" ]]; then
  echo "STOP: explicit authorization for boundary-state-block-intervention is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
ARCHIVE_BASE="$(realpath -m "$2")"
EXPECTED_HEAD="$3"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_boundary_intervention.json
CONFIG="$REPO/$CONFIG_RELATIVE"
CHECKPOINT="$REPO/runs/hanzi_stroke_temporal_composition/train/dev42/best_checkpoint.pt"
ISOLATED="$REPO/runs/hanzi_stroke_temporal_composition/fit_diagnostic/dev42_best_final/component_metrics.csv"
OUTPUT_DIR="$REPO/runs/hanzi_stroke_temporal_composition/boundary_intervention/dev42_best_medium"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"
EXPECTED_CHECKPOINT_SHA=76a7c2ca021a69ea99d68c143773307000890d72f7c7d6233614c6a5fab9ca39
EXPECTED_ISOLATED_SHA=d4e76993600e906df6bb25a6e2a67d0065cfe18874018841d6c9b453633bcecc

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$CONFIG"
test -f "$CHECKPOINT"
test -f "$ISOLATED"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in "$OUTPUT_DIR" "$LOG_TMP" "$ARCHIVE" "$ARCHIVE_SHA"; do
  if [[ -e "$path" ]]; then
    echo "ABORT: refusing to overwrite $path" >&2
    exit 1
  fi
done
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
test "$(sha256sum "$ISOLATED" | awk '{print $1}')" = "$EXPECTED_ISOLATED_SHA"
mkdir -p "$(dirname "$ARCHIVE_BASE")"

set +e
(
  set -e
  cd "$REPO"
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" - <<'PY'
import importlib.metadata
import torch

assert torch.__version__ == "2.6.0+cpu", torch.__version__
assert torch.cuda.is_available() is False
assert importlib.metadata.version("motornet") == "0.2.0"
print("SERVER_RUNTIME_PREFLIGHT=1")
PY
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -m unittest discover -s tests -p 'test_*.py'
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -m hanzi_writing.boundary_intervention --config "$CONFIG_RELATIVE"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Boundary intervention stopped; see $LOG_TMP" >&2
  exit 1
fi

TEST_COUNT="$(sed -nE 's/^Ran ([0-9]+) tests? in .*/\1/p' "$LOG_TMP" | tail -n 1)"
test -n "$TEST_COUNT"
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi
SKIPPED_COUNT=0
test -d "$OUTPUT_DIR"
mv "$LOG_TMP" "$OUTPUT_DIR/execution.log"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$OUTPUT_DIR/exit_code.txt"

"$PYTHON" - "$OUTPUT_DIR" "$TEST_COUNT" "$SKIPPED_COUNT" "$PROGRAM_EXIT" "$TEE_EXIT" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

output = Path(sys.argv[1])
test_count = int(sys.argv[2])
skipped_count = int(sys.argv[3])
program_exit = int(sys.argv[4])
tee_exit = int(sys.argv[5])

summary_path = output / "boundary_intervention_summary.json"
with summary_path.open("r", encoding="utf-8") as handle:
    summary = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
assert summary["completed"] is True
assert summary["boundary_count"] == 12
assert summary["actual_executed_condition_count"] == 60
assert summary["behavioral_pass_fail_defined"] is False
assert all(summary["integrity_checks"].values())

with (output / "boundary_intervention_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 60
counts = {}
for row in rows:
    counts[row["boundary_id"]] = counts.get(row["boundary_id"], 0) + 1
    for key, value in row.items():
        if value == "":
            continue
        if key.endswith("_m") or key.endswith("_mps") or "euclidean" in key:
            assert math.isfinite(float(value)), (key, value)
assert len(counts) == 12 and set(counts.values()) == {5}

with np.load(output / "boundary_trace.npz", allow_pickle=False) as archive:
    assert archive.files
    assert all(archive[name].dtype != object for name in archive.files)

provenance_path = output / "provenance.json"
with provenance_path.open("r", encoding="utf-8") as handle:
    provenance = json.load(handle)
provenance.update(
    {
        "test_count": test_count,
        "skipped_count": skipped_count,
        "program_exit_code": program_exit,
        "tee_exit_code": tee_exit,
    }
)
assert provenance["checkpoint_kind"] == "best"
assert provenance["speed"] == "medium"
assert provenance["network_noise"] is False
assert provenance["environment_noise"] is False
assert provenance["policy_requires_grad"] is False
assert provenance["policy_frozen_and_bitwise_unchanged"] is True
assert provenance["optimizer_created"] is False
assert provenance["backward_executed"] is False
with provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
PY

test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
EXPECTED_FILES=(
  BOUNDARY_INTERVENTION_REPORT.md
  boundary_intervention_metrics.csv
  boundary_intervention_summary.json
  boundary_trace.npz
  execution.log
  exit_code.txt
  provenance.json
)
for name in "${EXPECTED_FILES[@]}"; do
  test -f "$OUTPUT_DIR/$name"
done
test "$(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l)" -eq 7
(
  cd "$OUTPUT_DIR"
  sha256sum "${EXPECTED_FILES[@]}" > SHA256SUMS
  sha256sum -c SHA256SUMS
)
test "$(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l)" -eq 8
tar -C "$(dirname "$OUTPUT_DIR")" -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"

echo "FORMAL_TRAINING_STARTED=0"
echo "BOUNDARY_STATE_BLOCK_INTERVENTION_COMPLETE=1"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ARCHIVE=$ARCHIVE"
