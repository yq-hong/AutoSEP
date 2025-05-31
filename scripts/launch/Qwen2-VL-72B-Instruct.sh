set -e
set -x
#!/bin/bash

export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"
export HF_DATASETS_CACHE="/data/huggingface/datasets"

MODEL_REPO=Qwen/Qwen2-VL-72B-Instruct

GPU=$1
PORT=30000

IFS=',' read -ra GPU_ARRAY <<< "$GPU"

tensor_parallel_size=${#GPU_ARRAY[@]}
echo $tensor_parallel_size

mem_fraction_static=0.8

CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
  --model-path $MODEL_REPO \
  --port $PORT \
  --tp-size $tensor_parallel_size \
  --trust-remote-code \
  --mem-fraction-static $mem_fraction_static \
  --chat-template qwen2-vl \
  --disable-radix-cache \
  --chunked-prefill-size -1

# bash scripts/launch/Qwen2-VL-72B-Instruct.sh 4,5,6,7
