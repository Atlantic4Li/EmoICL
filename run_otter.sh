#!/bin/bash
# Otter-LLaMA: whole strategy, three datasets
# Usage: bash run_otter.sh [GPU_ID]

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for DS in EmotionROI EmoSet ArtPhoto; do
  echo "=== $DS | otter-llama ==="
  python inference.py --dataset "$DS" --engine otter-llama --n_shot 1 2 4 --balance_threshold 0.5 --exp_scale 10000
done

echo "Done."
