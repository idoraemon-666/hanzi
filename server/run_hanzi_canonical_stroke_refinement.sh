#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash server/run_hanzi_canonical_stroke_refinement.sh REPO ARCHIVE_BASE EXPECTED_HEAD SOURCE_STAGE1" >&2
  exit 2
fi
if [[ "${HANZI_TEMPORAL_COMPOSITION_AUTHORIZED_RUN:-}" != "canonical-eight-stroke-refinement-lr1e4-2000" ]]; then
  echo "STOP: explicit authorization for the eight-stroke refinement is required." >&2
  exit 2
fi

REPO="$(realpath "$1")"
ARCHIVE_BASE="$(realpath -m "$2")"
EXPECTED_HEAD="$3"
SOURCE_STAGE1="$(realpath "$4")"
PYTHON=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu/bin/python
SUBMODULE_HEAD=ac0c4f589eae37bbde63968912925de99232e306
CONFIG_RELATIVE=configurations/hanzi_stroke_temporal_composition_canonical_stroke_refinement_v1.json
OUTPUT_DIR="$REPO/runs/hanzi_stroke_temporal_composition/canonical_stroke_refinement/dev42"
LOG_TMP="${ARCHIVE_BASE}.execution.log.tmp"
ARCHIVE="${ARCHIVE_BASE}.tar.gz"
ARCHIVE_SHA="${ARCHIVE}.sha256"

test -d "$REPO/.git"
test -x "$PYTHON"
test -f "$REPO/$CONFIG_RELATIVE"
test -f "$SOURCE_STAGE1/provenance.json"
test -f "$SOURCE_STAGE1/SHA256SUMS"
test "$(git -C "$REPO" branch --show-current)" = "codex/hanzi-stroke-temporal-composition"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO/mRNNTorch" rev-parse HEAD)" = "$SUBMODULE_HEAD"
test -z "$(git -C "$REPO" status --short)"
test -z "$(git -C "$REPO/mRNNTorch" status --short)"
for path in "$OUTPUT_DIR" "$LOG_TMP" "$ARCHIVE" "$ARCHIVE_SHA"; do
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
  (
    cd "$SOURCE_STAGE1"
    sha256sum -c SHA256SUMS
  )
  "$PYTHON" - <<'PY'
import importlib.metadata
import torch

assert torch.__version__ == "2.6.0+cpu", torch.__version__
assert torch.cuda.is_available() is False
assert importlib.metadata.version("motornet") == "0.2.0"
print("SERVER_RUNTIME_PREFLIGHT=1")
PY
  "$PYTHON" -m unittest \
    tests.test_hanzi_geometry \
    tests.test_hanzi_environment \
    tests.test_hanzi_training_protocol \
    tests.test_hanzi_fit_diagnostic \
    tests.test_hanzi_boundary_intervention \
    tests.test_hanzi_fixed_duration_protocol \
    tests.test_hanzi_fixed_duration_diagnostic \
    tests.test_phase_normalized_loss \
    tests.test_hanzi_canonical_protocol \
    tests.test_hanzi_canonical_overfit \
    tests.test_hanzi_canonical_refinement
  echo "CANONICAL_EIGHT_STROKE_REFINEMENT_STARTED=1"
  echo "MOVE_REFINEMENT_STARTED=0"
  "$PYTHON" -m hanzi_writing.canonical_refinement \
    --config "$CONFIG_RELATIVE" \
    --prepare \
    --source "$SOURCE_STAGE1"
  TASKS=(heng shu pie na dian ti hengzhe shugou)
  WORKER_LOG_DIR="$OUTPUT_DIR/worker_logs"
  mkdir "$WORKER_LOG_DIR"
  WORKER_PIDS=()
  terminate_workers() {
    for pid in "${WORKER_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap terminate_workers INT TERM EXIT
  for task in "${TASKS[@]}"; do
    "$PYTHON" -m hanzi_writing.canonical_refinement \
      --config "$CONFIG_RELATIVE" \
      --task "$task" \
      --source "$SOURCE_STAGE1" \
      > "$WORKER_LOG_DIR/$task.log" 2>&1 &
    WORKER_PIDS+=("$!")
    echo "REFINEMENT_TASK_LAUNCHED=$task PID=$!"
  done
  echo "REFINEMENT_PARALLEL_WORKER_COUNT=${#WORKER_PIDS[@]}"
  WORKER_FAILURE=0
  for index in "${!WORKER_PIDS[@]}"; do
    task="${TASKS[$index]}"
    pid="${WORKER_PIDS[$index]}"
    if wait "$pid"; then
      echo "REFINEMENT_TASK_COMPLETE=$task PID=$pid"
    else
      status="$?"
      echo "REFINEMENT_TASK_FAILED=$task PID=$pid EXIT=$status" >&2
      WORKER_FAILURE=1
    fi
  done
  trap - INT TERM EXIT
  test "$WORKER_FAILURE" -eq 0
  "$PYTHON" -m hanzi_writing.canonical_refinement \
    --config "$CONFIG_RELATIVE" \
    --finalize \
    --source "$SOURCE_STAGE1"
) 2>&1 | tee "$LOG_TMP"
RUN_CODES=("${PIPESTATUS[@]}")
set -e
PROGRAM_EXIT="${RUN_CODES[0]}"
TEE_EXIT="${RUN_CODES[1]}"
if [[ "$PROGRAM_EXIT" -ne 0 || "$TEE_EXIT" -ne 0 ]]; then
  echo "Canonical eight-stroke refinement stopped; see $LOG_TMP and worker logs." >&2
  exit 1
fi

TEST_COUNT="$(sed -nE 's/^Ran ([0-9]+) tests? in .*/\1/p' "$LOG_TMP" | tail -n 1)"
test "$TEST_COUNT" = 69
if grep -Eq '^OK \(.*skipped=[1-9][0-9]*.*\)$' "$LOG_TMP"; then
  echo "ABORT: skipped tests are forbidden" >&2
  exit 1
fi
test -d "$OUTPUT_DIR"
mv "$LOG_TMP" "$OUTPUT_DIR/execution.log"
printf 'PROGRAM_EXIT=%s\nTEE_EXIT=%s\n' "$PROGRAM_EXIT" "$TEE_EXIT" \
  > "$OUTPUT_DIR/exit_code.txt"

OUTPUT_DIR="$OUTPUT_DIR" SOURCE_STAGE1="$SOURCE_STAGE1" TEST_COUNT="$TEST_COUNT" \
PROGRAM_EXIT="$PROGRAM_EXIT" TEE_EXIT="$TEE_EXIT" "$PYTHON" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])

def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )

manifest = load_json(output / "source_selection.json")
assert manifest["move_included"] is False
assert len(manifest["selections"]) == 8
assert {value["task"] for value in manifest["selections"]} == {
    "heng", "shu", "pie", "na", "dian", "ti", "hengzhe", "shugou"
}
for selection in manifest["selections"]:
    losses = {
        value["checkpoint"]: value["validation_loss"]
        for value in selection["candidates"]
    }
    expected = "best" if losses["best"] <= losses["review"] else "review"
    assert selection["selected_checkpoint"] == expected

models = output / "models"
expected_files = {
    "best_checkpoint.pt",
    "continuation_checkpoint.pt",
    "review_checkpoint.pt",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
}
assert {path.name for path in models.iterdir()} == {
    "heng", "shu", "pie", "na", "dian", "ti", "hengzhe", "shugou"
}
for directory in models.iterdir():
    assert {path.name for path in directory.iterdir()} == expected_files
    assert sum(1 for line in (directory / "training_metrics.jsonl").open()) == 20
    rows = [json.loads(line) for line in (directory / "validation_metrics.jsonl").open()]
    assert len(rows) == 21
    assert rows[-1]["validation_kind"] == "review_read_only"
    assert rows[-1]["update"] == 1999
    assert all(rows[-1]["read_only_checks"].values())

with (output / "stroke_refinement_metrics.csv").open(encoding="utf-8", newline="") as handle:
    metric_rows = list(csv.DictReader(handle))
assert len(metric_rows) == 24
assert {row["checkpoint"] for row in metric_rows} == {"source", "best", "review"}
for row in metric_rows:
    for key, value in row.items():
        if value and key.endswith(("_m", "_deg", "_ratio", "_loss")):
            assert math.isfinite(float(value)), (key, value)

assert len(list((output / "plots").glob("*.png"))) == 9
provenance_path = output / "provenance.json"
provenance = load_json(provenance_path)
assert provenance["completed"] is True
assert provenance["learning_rate"] == 0.0001
assert provenance["additional_updates"] == 2000
assert provenance["validation_interval"] == 100
assert provenance["execution_mode"] == "eight_independent_parallel_processes"
assert provenance["integrity"]["stroke_tasks_completed"] == 8
assert provenance["integrity"]["metrics_rows"] == 24
assert provenance["integrity"]["trajectory_rows"] == 3924
assert provenance["integrity"]["plots"] == 9
assert len(provenance["checkpoint_sha256"]) == 24
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
with provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(provenance, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

(
  cd "$OUTPUT_DIR"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
tar -C "$(dirname "$OUTPUT_DIR")" -czf "$ARCHIVE" "$(basename "$OUTPUT_DIR")"
sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"

echo "CANONICAL_EIGHT_STROKE_REFINEMENT_COMPLETE=1"
echo "MOVE_REFINEMENT_STARTED=0"
echo "SHARED_9TASK_STARTED=0"
echo "FORMAL_75K_TRAINING_STARTED=0"
echo "COMPLETE_CHARACTER_ROLLOUT_STARTED=0"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ARCHIVE=$ARCHIVE"
