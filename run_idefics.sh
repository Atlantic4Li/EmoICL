#!/bin/bash
# Idefics-9B-Instruct: whole strategy, three datasets
# Usage: bash run_idefics.sh [GPU_ID]

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for DS in EmotionROI EmoSet ArtPhoto; do
  echo "=== $DS | idefics-9b-instruct ==="
  python inference.py --dataset "$DS" --engine idefics-9b-instruct --n_shot 1 2 4 --balance_threshold 0.5 --exp_scale 10000
done

echo "Done."
