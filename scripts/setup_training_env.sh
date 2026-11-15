#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
UV="${UV_BIN:-$PWD/.local-tools/bin/uv}"
PYPI_INDEX="${DECO_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [ ! -x "$UV" ]; then
    python3.10 -m pip install --target .local-tools --no-cache-dir \
        --index-url "$PYPI_INDEX" uv==0.13.7
fi
if [ ! -x .venv/bin/python ]; then
    "$UV" venv --python python3.10 .venv
fi

# Install the explicit CUDA wheels before resolving the remaining dependencies.
"$UV" pip install --python .venv/bin/python --no-deps \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124'
TRAINING_REQUIREMENTS=requirements-training.in
if [ -f requirements-training.lock.txt ]; then
    TRAINING_REQUIREMENTS=requirements-training.lock.txt
fi
"$UV" pip install --python .venv/bin/python \
    --index-url "$PYPI_INDEX" -r "$TRAINING_REQUIREMENTS"
"$UV" pip check --python .venv/bin/python

# CUDA is required: a CPU-only run must not produce a successful training report.
.venv/bin/python scripts/verify_training_env.py --download-weights
