import sys
import argparse
import os
import csv
import torch
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    

sys.path.insert(0, "/home/lixiyang/ICL/LLaVA")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from sklearn.metrics import classification_report


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 类别映射字典
CATEGORY_MAP = {
    0: "Attractive",
    1: "BeatCompete",
    2: "Communicate",
    3: "CreativeUnique",
    4: "CuriousAdventurousExcitingLife",
    5: "EasyLife",
    6: "EnjoyLife",
    7: "FineDesignLearnArt-Arch",
    8: "FineDesignLearnArt-Art",
    9: "FineDesignLearnArt-Culture",
    10: "GoodParentEmoCloseChild",
    11: "Happy",
    12: "HardWorking",
    13: "Harmony",
    14: "Health",
    15: "InLove",
    16: "InLoveAnimal",
    17: "InspirOthrs",
    18: "ManagableMakePlan",
    19: "NatBeauty",
    20: "PassionAbSmthing",
    21: "Playful",
    22: "ShareFeelings",
    23: "SocialLifeFriendship",
    24: "SuccInOccupHavGdJob",
    25: "TchOthrs",
    26: "ThngsInOrdr",
    27: "WorkILike"
}

def load_category_mapping(csv_path):
    """加载图像ID到类别的映射"""
    id_to_category = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_id']
            for ext in ['.jpg', '.png', '.jpeg']:
                if image_id.endswith(ext):
                    image_id = image_id[:-len(ext)]
                    break
            id_to_category[image_id] = int(row['category'])
    return id_to_category

def create_version_dir(base_path="/data1/lxy_log/icl_log/IntentICL/results"):
    """创建版本目录并返回路径"""
    existing_versions = []
    for dir_name in os.listdir(base_path):
        if dir_name.startswith("version_") and os.path.isdir(os.path.join(base_path, dir_name)):
            try:
                version_num = int(dir_name.split("_")[1])
                existing_versions.append(version_num)
            except (IndexError, ValueError):
                pass
    
    new_version = max(existing_versions) + 1 if existing_versions else 1
    version_dir = os.path.join(base_path, f"version_{new_version}")
    os.makedirs(version_dir, exist_ok=True)
    return version_dir

def generate_explanations(args):
    # 初始化模型
    model_path = "/data0/lxy_data/huggingface/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base,
        model_name
    )
    
    # 加载数据
    id_to_category = load_category_mapping(args.csv_path)
    
    # 准备结果存储
    results = []
    version_dir = create_version_dir()
    
    # 创建进度条
    pbar = tqdm(total=len(id_to_category), desc="Generating Explanations")
    
    for image_id, true_label in id_to_category.items():
        # 获取真实标签文本
        true_category = CATEGORY_MAP[true_label]
        
        # 构建图像路径
        img_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(args.image_dir, f"{image_id}.png")
            
        # 加载并预处理图像
        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # 构建解释型prompt
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        prompt = (
            f"Question: Given this image belongs to the category: '{true_category}', "
            "analyze the visual elements and explain why it fits this category. "
            "Consider these aspects:\n"
            "1. Key objects and their attributes\n"
            "2. Composition and visual patterns\n"
            "3. Contextual cues and implied meaning\n"
            "4. Emotional tone and atmosphere"
        )
        
        # 构建模型输入
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt, tokenizer, 
            IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # 生成解释
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=1024,  # 增加token长度
                use_cache=True
            )

        # 解析输出
        explanation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 存储结果
        results.append({
            "image_id": image_id,
            "true_category": true_category,
            "explanation": explanation
        })
        
        pbar.update(1)
    
    pbar.close()

    # 保存结果
    csv_path = os.path.join(version_dir, "explanations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "true_category", "explanation"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Generated {len(results)} explanations in {version_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="/data0/lxy_data/Intentonomy/images")
    parser.add_argument("--csv-path", type=str, default="/data0/lxy_data/Intentonomy/demo.csv")
    parser.add_argument("--model-base", type=str, default=None)
    args = parser.parse_args()
    
    generate_explanations(args)



