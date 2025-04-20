#!/bin/bash

# Batch run for qwen-vl on three datasets
for dataset in "ArtPhoto" "Intentonomy" "EmotionROI" "EmoSet"; do
    python inference.py --dataset "$dataset"
done
