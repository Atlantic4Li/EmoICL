#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取查询集和支持集的文本特征

本脚本用于从查询集和支持集的文本描述中提取特征向量，
并将这些特征向量保存为pkl文件，以便后续相似度计算和检索使用。
"""

import os
import json
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import clip  # 使用OpenAI的原始CLIP库代替transformers

# 配置参数
BASE_DIR = "/data1/yq_log/IntenICL"
BASE_IMG_DIR = os.path.join(BASE_DIR, "DataSet")
CLIP_MODEL = "ViT-B/32"  # CLIP模型名称
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128  # 批处理大小

def load_metadata(json_path):
    """加载元数据并提取文本描述
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        字典，格式为 {img_id: {"description": text_description}}
    """
    print(f"加载元数据：{json_path}")
    with open(json_path) as f:
        raw_data = json.load(f)
    
    meta_dict = {}
    description_count = 0
    
    for item in raw_data:
        img_id = str(item["img_id"])
        # 尝试获取不同可能的文本描述字段
        description = item.get("description", "")
        if not description and "caption" in item:
            description = item.get("caption", "")
        if not description and "text" in item:
            description = item.get("text", "")
        if not description and "answer" in item:
            description = item.get("answer", "")
        
        if description:
            description_count += 1
            
        # 存储元数据
        meta_dict[img_id] = {
            "description": description,
            "label": item.get("answer", item.get("label", "")),
            "category": item.get("category", "")
        }
    
    print(f"共加载 {len(meta_dict)} 条元数据，其中包含文本描述的有 {description_count} 条")
    return meta_dict

def extract_text_features_batch(descriptions, model, preprocess):
    """批量提取文本特征
    
    Args:
        descriptions: 字典，格式为 {id: text_description}
        model: CLIP模型
        preprocess: CLIP预处理函数（这里不会使用，因为我们只处理文本）
        
    Returns:
        特征字典，格式为 {id: feature_tensor}
    """
    model.eval()
    text_features_dict = {}
    items = list(descriptions.items())
    
    # 按批次处理
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="提取文本特征"):
        batch = items[i:i+BATCH_SIZE]
        batch_ids = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]
        
        with torch.no_grad():
            # 处理空文本
            valid_indices = [i for i, text in enumerate(batch_texts) if text.strip()]
            if not valid_indices:
                continue
                
            valid_texts = [batch_texts[i] for i in valid_indices]
            valid_ids = [batch_ids[i] for i in valid_indices]
            
            # 批量编码文本 (原生CLIP API)
            text_tokens = clip.tokenize(valid_texts).to(DEVICE)
            text_features = model.encode_text(text_tokens)
            
            # 归一化
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # 保存特征
            for idx, img_id in enumerate(valid_ids):
                text_features_dict[img_id] = text_features[idx].cpu()
    
    return text_features_dict

def generate_fallback_descriptions(meta_dict):
    """为没有描述的数据生成替代文本
    
    使用类别信息或标签生成简单描述
    
    Args:
        meta_dict: 元数据字典
        
    Returns:
        带有描述的字典 {id: description}
    """
    descriptions = {}
    no_desc_count = 0
    
    for img_id, meta in meta_dict.items():
        desc = meta.get("description", "").strip()
        
        # 如果没有描述，尝试使用类别或标签信息
        if not desc:
            no_desc_count += 1
            category = meta.get("category", "").strip()
            label = meta.get("label", "").strip()
            
            if category:
                desc = f"This is an image about {category}."
            elif label:
                desc = f"This image shows {label}."
            else:
                desc = "This is an image."
        
        descriptions[img_id] = desc
    
    print(f"共生成 {no_desc_count} 条替代描述")
    return descriptions

def process_dataset(dataset_name):
    """处理单个数据集
    
    Args:
        dataset_name: 数据集名称
    """
    print(f"\n========== 处理数据集: {dataset_name} ==========")
    
    # 设置路径
    dataset_dir = os.path.join(BASE_IMG_DIR, dataset_name)
    query_json_path = os.path.join(dataset_dir, "query.json")
    support_json_path = os.path.join(dataset_dir, "support.json")
    
    # 检查文件是否存在
    for file_path in [query_json_path, support_json_path]:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return
    
    # 加载原生CLIP模型
    print("加载CLIP模型...")
    model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    
    # 处理查询集
    print("\n提取查询集文本特征...")
    query_meta = load_metadata(query_json_path)
    query_descriptions = generate_fallback_descriptions(query_meta)
    query_text_features = extract_text_features_batch(query_descriptions, model, preprocess)
    
    # 保存查询集文本特征
    query_features_path = os.path.join(dataset_dir, "query_text_features.pkl")
    print(f"保存查询集文本特征到: {query_features_path}")
    with open(query_features_path, "wb") as f:
        pickle.dump(query_text_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 处理支持集
    print("\n提取支持集文本特征...")
    support_meta = load_metadata(support_json_path)
    support_descriptions = generate_fallback_descriptions(support_meta)
    support_text_features = extract_text_features_batch(support_descriptions, model, preprocess)
    
    # 保存支持集文本特征
    support_features_path = os.path.join(dataset_dir, "support_text_features.pkl")
    print(f"保存支持集文本特征到: {support_features_path}")
    with open(support_features_path, "wb") as f:
        pickle.dump(support_text_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 计算文本-文本相似度（可选）
    compute_text_similarity = True
    if compute_text_similarity:
        print("\n计算文本-文本相似度...")
        text_text_similarity = compute_similarity(
            query_text_features, 
            support_text_features, 
            os.path.join(dataset_dir, "text_text_similarity.pkl")
        )
    
    # 打印统计信息
    print("\n处理结果摘要：")
    print(f"数据集: {dataset_name}")
    print(f"查询文本特征数: {len(query_text_features)}")
    print(f"支持文本特征数: {len(support_text_features)}")
    if compute_text_similarity:
        print(f"文本相似度记录数: {len(text_text_similarity)}")
    print(f"存储位置: {dataset_dir}")

def compute_similarity(query_features, support_features, output_path):
    """计算两组特征之间的相似度矩阵
    
    Args:
        query_features: 查询特征字典 {id: feature}
        support_features: 支持特征字典 {id: feature}
        output_path: 输出文件路径
        
    Returns:
        相似度字典 {query_id: {support_id: similarity}}
    """
    query_ids = list(query_features.keys())
    support_ids = list(support_features.keys())
    
    # 转换为张量矩阵
    Q = torch.stack([query_features[q_id] for q_id in query_ids]).to(DEVICE)
    S = torch.stack([support_features[s_id] for s_id in support_ids]).to(DEVICE)
    
    # 计算相似度
    similarity_dict = {}
    with torch.no_grad():
        # 计算文本-文本相似度
        sim_matrix = torch.mm(Q, S.T).cpu()
        
        # 构建字典
        for i, q_id in tqdm(enumerate(query_ids), desc="构建相似度字典"):
            similarity_dict[q_id] = {
                s_id: sim_matrix[i][j].item() 
                for j, s_id in enumerate(support_ids)
            }
    
    # 保存相似度
    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(similarity_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return similarity_dict

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='提取查询集和支持集的文本特征')
    parser.add_argument('--datasets', nargs='+', 
                      default=['EmoSet'],
                      choices=['Intentonomy', 'EmoSet', 'ArtPhoto', 'EmotionROI'],
                      help='要处理的数据集列表')
    args = parser.parse_args()
    
    print(f"设备: {DEVICE}")
    print(f"批处理大小: {BATCH_SIZE}")
    
    # 处理每个数据集
    for dataset in args.datasets:
        try:
            process_dataset(dataset)
            print(f"\n数据集 {dataset} 处理完成")
        except Exception as e:
            import traceback
            print(f"\n处理数据集 {dataset} 时出错:")
            print(traceback.format_exc())
            continue
    
    print("\n所有数据集处理完成!")

if __name__ == "__main__":
    main()
