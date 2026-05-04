import os

import json
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import sys
sys.path.append('..')

# 配置基础路径
BASE_DIR = "/data1/yq_log/IntenICL"
BASE_IMG_DIR = os.path.join(BASE_DIR, "DataSet")
BLIP_MODEL_DIR = os.path.join(BASE_DIR, "huggingface/Salesforce/blip2-flan-t5-xl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_blip_model(device="cuda"):
    """加载BLIP2模型（Hugging Face版本）"""
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP_MODEL_DIR)
    processor = Blip2Processor.from_pretrained(BLIP_MODEL_DIR)
    model = model.to(device)
    return model, processor


def get_query_json_path(dataset_name):
    """DataSet/{dataset}/query.json"""
    dataset_dir = os.path.join(BASE_IMG_DIR, dataset_name)
    return dataset_dir, os.path.join(dataset_dir, "query.json")


def _image_rel_path(item):
    """从 json 条目中取出相对 DataSet 根目录的图像路径（与 create_vt_sim 一致）。"""
    img = item.get("image")
    if img is None:
        return None
    if isinstance(img, list):
        return img[0] if img else None
    if isinstance(img, str):
        return img
    return None


def resolve_image_path(item):
    rel = _image_rel_path(item)
    if not rel:
        return None
    return os.path.join(BASE_IMG_DIR, rel)


def generate_description(image_path, model, processor, dataset_type):
    """为单张图片生成描述"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    if dataset_type in ['EmotionROI', 'ArtPhoto', 'EmoSet']:
        prompt = "Describe this image with a focus on the emotional content, mood, and atmosphere."
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)
        return description
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def process_dataset(dataset_name):
    """为 query.json 中每条样本写入 description 字段（供 create_vt_sim 等使用）。"""
    print(f"\n处理数据集: {dataset_name}")

    dataset_dir, json_path = get_query_json_path(dataset_name)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"query.json 不存在: {json_path}")

    print("加载BLIP模型...")
    model, vis_processors = load_blip_model(DEVICE)

    print(f"读取 {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_images = len(data)
    processed_images = 0
    failed_images = 0

    print(f"开始为 {total_images} 条 query 生成描述...")

    for item in tqdm(data, desc=f"query {dataset_name}"):
        image_path = resolve_image_path(item)
        if not image_path or not os.path.exists(image_path):
            print(f"\n警告: 图片路径无效或不存在: {image_path}")
            failed_images += 1
            continue

        description = generate_description(image_path, model, vis_processors, dataset_name)
        if description:
            item['description'] = description
            processed_images += 1
        else:
            failed_images += 1

    print("\n保存更新后的 query.json...")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n处理完成")
    print(f"总条数: {total_images}")
    print(f"成功写入 description: {processed_images}")
    print(f"失败: {failed_images}")
    if total_images:
        print(f"成功率: {(processed_images / total_images) * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='为 query.json 生成图像描述（BLIP2）')
    parser.add_argument(
        '--datasets', nargs='+',
        default=['EmoSet'],
        choices=['EmoSet', 'ArtPhoto', 'EmotionROI'],
        help='要处理的数据集列表',
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"\n处理数据集 {dataset} 时出错: {str(e)}")
            continue
