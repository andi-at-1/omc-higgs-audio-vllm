#!/bin/bash
# Build script for omc-higgs-audio-vllm with ccache

set -e

cd /root/omc-higgs-audio-vllm

# Activate venv
source .venv/bin/activate

# Setup ccache for faster rebuilds
export CCACHE_DIR=/root/.ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# RTX 3090 = sm_86, RTX 5090 = sm_120
export TORCH_CUDA_ARCH_LIST="8.6;12.0"

# Parallel jobs (64 cores available)
export MAX_JOBS=60

echo "=== Building vLLM with CUDA arch: $TORCH_CUDA_ARCH_LIST ==="
echo "=== Using ccache dir: $CCACHE_DIR ==="

pip install -e . 2>&1 | tee install.log

echo ""
echo "=== Build complete! ==="
echo "Check install.log for details"
