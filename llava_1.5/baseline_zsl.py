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

def predict_intent(args):
    # 初始化模型
    model_path = "/data0/lxy_data/huggingface/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base,
        model_name
    )
    
    # 加载测试数据
    id_to_category = load_category_mapping(args.csv_path)
    
    # 准备结果存储
    all_preds = []
    all_labels = []
    csv_rows = []
    
    # 创建版本目录
    version_dir = create_version_dir()
    
    # 创建进度条
    pbar = tqdm(total=len(id_to_category), desc="Processing Images")
    
    for image_id, true_label in id_to_category.items():
        # 构建图像路径
        img_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(args.image_dir, f"{image_id}.png")
            
        # 加载并预处理图像
        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # 构建prompt
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        prompt = (
            "Analyze this image and determine its intent category from the following options:\n"
            + ", ".join(CATEGORY_MAP.values()) + 
            "\nOnly respond with the exact category name."
        )
        
        # 构建输入
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt, tokenizer, 
            IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # 生成响应
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=0.7,
                top_p=1.0,
                max_new_tokens=64,
                use_cache=True
            )

        # 解析输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 匹配预测类别
        pred_category = None
        for name in CATEGORY_MAP.values():
            if name.lower() in outputs.lower():
                pred_category = name
                break
        
        # 记录CSV数据
        csv_rows.append({
            "image_id": image_id,
            "output": outputs,
            "ver_category": pred_category if pred_category else "",
            "true_category": CATEGORY_MAP[true_label]
        })
        
        # 记录评估数据
        if pred_category:
            pred_label = [k for k, v in CATEGORY_MAP.items() if v == pred_category][0]
        else:
            pred_label = -1
            
        all_preds.append(pred_label)
        all_labels.append(true_label)
        
        pbar.update(1)
    
    pbar.close()
    
    # 保存结果
    csv_path = os.path.join(version_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "output", "ver_category", "true_category"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # 计算指标
    valid_indices = [i for i, pred in enumerate(all_preds) if pred != -1]
    valid_preds = [all_preds[i] for i in valid_indices]
    valid_labels = [all_labels[i] for i in valid_indices]

    if valid_preds:
        acc = accuracy_score(valid_labels, valid_preds)
        macro_f1 = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
        macro_precision = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
        macro_recall = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
 
        micro_f1 = f1_score(valid_labels, valid_preds, average='micro', zero_division=0)
        micro_precision = precision_score(valid_labels, valid_preds, average='micro', zero_division=0)
        micro_recall = recall_score(valid_labels, valid_preds, average='micro', zero_division=0)

        cls_report = classification_report(
            valid_labels, 
            valid_preds, 
            target_names=CATEGORY_MAP.values(),
            output_dict=True,
            zero_division=0
        )
        
    else:
        acc = macro_precision = macro_recall = macro_f1 = micro_precision = micro_recall = micro_f1 = 0.0
        cls_report = {}
    
    coverage = len(valid_preds) / len(all_preds) if all_preds else 0.0
    
    json_path = os.path.join(version_dir, "results.json")
    
    with open(json_path, "w") as f:
        json.dump({
            "coverage": coverage,
            "accuracy": acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "per_class_metrics": cls_report
        }, f, indent=2)
    
    print(f"\nResults saved to: {version_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, 
                       default="/data0/lxy_data/Intentonomy/images")
    parser.add_argument("--csv-path", type=str,
                       default="/data0/lxy_data/Intentonomy/test.csv")
    parser.add_argument("--model-base", type=str, default=None)
    args = parser.parse_args()
    
    predict_intent(args)