#!/bin/bash
# Start Higgs Audio vLLM server on GPU 1

cd /root/omc-higgs-audio-vllm
source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

exec python -m vllm.entrypoints.bosonai.api_server \
    --model "bosonai/higgs-audio-v2-generation-3B-base" \
    --audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
    --port 8778 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --ras-window-length 7 \
    --ras-max-num-repeat 2 \
    "$@"
