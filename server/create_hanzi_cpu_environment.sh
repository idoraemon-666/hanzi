#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash server/create_hanzi_cpu_environment.sh REPO" >&2
  exit 2
fi

REPO="$(realpath "$1")"
ENV_PATH=/root/autodl-tmp/conda/envs/hanzi-stroke-temporal-composition-cpu
PYTHON="$ENV_PATH/bin/python"
COMPLETE_MARKER="$ENV_PATH/.hanzi_cpu_environment_complete"
TUNA_CONDA_MAIN=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
TUNA_PYPI=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ALIYUN_TORCH_WHEEL=https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl
COMMON_REQUIREMENTS="$REPO/server/requirements-linux-common.txt"

test -d "$REPO/.git"
test -f "$COMMON_REQUIREMENTS"
if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "ENVIRONMENT_ALREADY_READY=$ENV_PATH"
  exit 0
fi
if [[ -e "$ENV_PATH" ]]; then
  test -x "$PYTHON"
  echo "RESUMING_INCOMPLETE_ENVIRONMENT=$ENV_PATH"
else
  conda create -y -p "$ENV_PATH" --override-channels \
    -c "$TUNA_CONDA_MAIN" python=3.10.20 pip
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON" -m pip install \
  --index-url "$TUNA_PYPI" pip==26.1.2 setuptools==83.0.0 wheel==0.47.0
PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON" -m pip install \
  --no-deps --index-url "$TUNA_PYPI" -r "$COMMON_REQUIREMENTS"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON" -m pip install \
  --no-deps "$ALIYUN_TORCH_WHEEL"

"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import motornet
import numpy
import torch

assert torch.__version__ == "2.6.0+cpu", torch.__version__
assert not torch.cuda.is_available()
print("torch", torch.__version__)
print("motornet", motornet.__version__)
print("numpy", numpy.__version__)
print("CPU_BASELINE_VERIFIED=1")
PY
{
  printf 'project=hanzi_stroke_temporal_composition\n'
  printf 'device=cpu\n'
  printf 'python=3.10.20\n'
  printf 'torch=2.6.0+cpu\n'
  printf 'conda_source=%s\n' "$TUNA_CONDA_MAIN"
  printf 'pypi_source=%s\n' "$TUNA_PYPI"
  printf 'torch_source=%s\n' "$ALIYUN_TORCH_WHEEL"
} > "$COMPLETE_MARKER"
echo "ENVIRONMENT_READY=$ENV_PATH"
