#!/bin/bash
# 用于训练和使用视觉-情感特征对齐CLIP模型的简化脚本

# 基本配置
DATASET="Intentonomy"  # 可选: Intentonomy, EmoSet, ArtPhoto, EmotionROI
DATA_ROOT="/data1/yq_log/IntenICL/DataSet"
CHECKPOINT_DIR="/data1/yq_log/IntenICL/ICL_log/intCLIP/0_8_2"

# 创建检查点目录
mkdir -p "$CHECKPOINT_DIR"

# 获取命令行第一个参数作为模式
MODE=$1

if [ "$MODE" = "train" ]; then
  # 训练模式
  echo "开始训练模型 (数据集: $DATASET)..."
  python myCLIP.py --mode train \
                   --dataset "$DATASET" \
                   --data_root "$DATA_ROOT" \
                   --batch_size 32 \
                   --lr 2e-4 \
                   --epochs 60 \
                   --lora_rank 16 \
                   --checkpoint_dir "$CHECKPOINT_DIR"

elif [ "$MODE" = "inference" ]; then
  # 推理模式
  IMAGE_PATH=$2
  MODEL_PATH=$3
  
  # 检查必要参数
  if [ -z "$IMAGE_PATH" ] || [ -z "$MODEL_PATH" ]; then
    echo "错误: 推理模式需要指定图像路径和模型路径"
    echo "用法: $0 inference 图像路径 模型路径 [use_classifier]"
    exit 1
  fi
  
  # 检查是否使用分类头
  if [ "$4" = "use_classifier" ]; then
    echo "使用分类头进行推理..."
    CLASSIFIER_FLAG="--use_classifier"
  else
    echo "使用特征相似度进行推理 (推荐)..."
    CLASSIFIER_FLAG=""
  fi
  
  # 执行推理
  python myCLIP.py --mode inference \
                   --dataset "$DATASET" \
                   --image_path "$IMAGE_PATH" \
                   --model_path "$MODEL_PATH" \
                   $CLASSIFIER_FLAG

else
  # 显示帮助信息
  echo "CLIP视觉-情感特征对齐模型脚本"
  echo ""
  echo "用法:"
  echo "  $0 train                       训练模型"
  echo "  $0 inference 图像路径 模型路径 [use_classifier]  使用模型进行推理"
  echo ""
  echo "当前设置:"
  echo "  数据集: $DATASET"
  echo "  数据根目录: $DATA_ROOT"
  echo "  检查点目录: $CHECKPOINT_DIR"
  echo ""
  echo "要修改设置，请直接编辑脚本开头的基本配置部分。"
fi 