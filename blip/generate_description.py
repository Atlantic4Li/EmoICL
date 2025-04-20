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

def get_dataset_paths(dataset_name):
    """根据数据集名称生成相关路径"""
    dataset_dir = os.path.join(BASE_IMG_DIR, dataset_name)
    json_path = os.path.join(dataset_dir, "support.json")
    return BASE_IMG_DIR, json_path

def generate_description(image_path, model, processor, dataset_type):
    """为单张图片生成描述"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # 根据数据集类型选择合适的提示词
    if dataset_type in ['EmotionROI', 'ArtPhoto', 'EmoSet']:
        prompt = "Describe this image with a focus on the emotional content, mood, and atmosphere."
    elif dataset_type == 'Intentonomy':
        prompt = "Describe this image focusing on what purpose it serves and the intention behind it."
    else:
        prompt = "Describe this image with detailed focus on the main object and its attributes."

    try:
        # 加载和预处理图像
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
        
        # 生成描述
        outputs = model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)
        return description
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_dataset(dataset_name):
    """处理单个数据集"""
    print(f"\n处理数据集: {dataset_name}")
    
    # 获取数据集相关路径
    dataset_dir, json_path = get_dataset_paths(dataset_name)
    
    # 验证路径
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"support.json不存在: {json_path}")
    
    # 加载模型
    print("加载BLIP模型...")
    model, vis_processors = load_blip_model(DEVICE)
    
    # 读取JSON文件
    print("读取support.json...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 统计信息
    total_images = len(data)
    processed_images = 0
    failed_images = 0
    
    print(f"开始处理{total_images}张图片...")
    
    # 为每张图片生成描述
    for item in tqdm(data, desc=f"处理{dataset_name}数据集"):
        image_path = os.path.join(dataset_dir, item.get('image', '')[0])
        if not os.path.exists(image_path):
            print(f"\n警告: 图片不存在 {image_path}")
            failed_images += 1
            continue
            
        description = generate_description(image_path, model, vis_processors, dataset_name)
        if description:
            item['description'] = description
            processed_images += 1
        else:
            failed_images += 1
    
    # 保存更新后的JSON文件
    print("\n保存更新后的JSON文件...")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # 打印处理结果
    print(f"\n处理完成！")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"处理失败: {failed_images}")
    print(f"成功率: {(processed_images/total_images)*100:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='为数据集生成图像描述')
    parser.add_argument('--datasets', nargs='+', 
                      default=['Intentonomy', 'EmoSet', 'EmotionROI', 'ArtPhoto'],
                      choices=['Intentonomy', 'EmoSet', 'ArtPhoto', 'EmotionROI'],
                      help='要处理的数据集列表')
    args = parser.parse_args()
    
    # 处理每个数据集
    for dataset in args.datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"\n处理数据集 {dataset} 时出错: {str(e)}")
            continue 