#!/bin/bash
# LLaVA-OneVision: whole strategy, EmotionROI / EmoSet / ArtPhoto
# Usage: bash run_llava.sh [GPU_ID]

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for DS in EmotionROI EmoSet ArtPhoto; do
  echo "=== $DS | llava-onevision-0.5b ==="
  python inference.py --dataset "$DS" --engine llava-onevision-0.5b --n_shot 1 2 4 --balance_threshold 0.5 --exp_scale 10000
  echo "=== $DS | llava-onevision-7b ==="
  python inference.py --dataset "$DS" --engine llava-onevision-7b --n_shot 1 2 4 --balance_threshold 0.5 --exp_scale 10000
done

echo "Done."
