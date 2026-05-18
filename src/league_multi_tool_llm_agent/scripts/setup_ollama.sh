#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=999
export OLLAMA_FLASH_ATTN=1

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_HOST=127.0.0.1:11434

sudo apt-get update -qq

sudo apt-get install lshw

sudo apt-get install zstd

# sudo apt install -y pciutils

curl -fsSL https://ollama.com/install.sh | sh

# sleep 8

echo "Checking GPU..."
nvidia-smi
