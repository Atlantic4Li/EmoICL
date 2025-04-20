import os
import json
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# 配置参数
BASE_DIR = "/data1/yq_log/IntenICL"
BASE_IMG_DIR = os.path.join(BASE_DIR, "DataSet")
CLIP_MODEL_DIR = os.path.join(BASE_DIR, "huggingface/openai/clip-vit-base-patch32")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # 批处理大小

def load_metadata(json_path):
    """加载元数据并验证完整性"""
    with open(json_path) as f:
        raw_data = json.load(f)
    
    meta_dict = {}
    for item in raw_data:
        img_id = str(item["img_id"])
        if "description" in item:
            meta_dict[img_id] = {
                "description": item.get("description", "")
            }
        else:
            # 如果没有描述，则跳过
            continue
    
    return meta_dict

def extract_text_features(descriptions, model, processor):
    """批量提取文本特征"""
    model.eval()
    text_features_dict = {}
    items = list(descriptions.items())
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items), BATCH_SIZE), desc="提取文本特征"):
            batch = items[i:i+BATCH_SIZE]
            texts = [desc for _, desc in batch]
            ids = [id for id, _ in batch]
            
            # 处理文本
            inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP的最大文本长度
            )
            # 移动到GPU
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # 提取特征
            text_features = model.get_text_features(**inputs)
            # 归一化
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # 保存特征
            for idx, img_id in enumerate(ids):
                text_features_dict[img_id] = text_features[idx].cpu()
    
    return text_features_dict

def compute_text_similarity(query_text_features, support_text_features, save_dir):
    """计算查询集文本与支持集文本之间的相似度"""
    query_ids = list(query_text_features.keys())
    support_ids = list(support_text_features.keys())
    
    print(f"有效查询样本数: {len(query_ids)}")
    print(f"有效支持样本数: {len(support_ids)}")
    
    # 准备特征矩阵
    Q = torch.stack([query_text_features[q_id] for q_id in query_ids]).to(DEVICE)
    S = torch.stack([support_text_features[s_id] for s_id in support_ids]).to(DEVICE)
    
    # 检查并打印特征类型
    print(f"查询特征类型: {Q.dtype}")
    print(f"支持特征类型: {S.dtype}")
    
    # 计算相似度
    similarity_dict = {}
    print("计算相似度矩阵...")
    with torch.no_grad():
        # 计算文本-文本相似度
        sim_matrix = torch.mm(Q, S.T).cpu()
        
        # 构建相似度字典
        for i, q_id in tqdm(enumerate(query_ids), desc="构建相似度字典"):
            similarity_dict[q_id] = {
                s_id: sim_matrix[i][j].item() 
                for j, s_id in enumerate(support_ids)
            }
    
    # 保存相似度
    sim_path = os.path.join(save_dir, "clip_text_text_similarity.pkl")
    print(f"保存相似度矩阵到: {sim_path}")
    with open(sim_path, "wb") as f:
        pickle.dump(similarity_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return similarity_dict

def process_dataset(dataset_name):
    """处理单个数据集"""
    print(f"\n处理数据集: {dataset_name}")
    
    # 设置路径
    dataset_dir = os.path.join(BASE_IMG_DIR, dataset_name)
    query_json_path = os.path.join(dataset_dir, "query.json")
    support_json_path = os.path.join(dataset_dir, "support.json")
    
    # 加载CLIP模型
    print("加载CLIP模型...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_DIR).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_DIR)
    
    # 加载元数据
    query_meta = load_metadata(query_json_path)
    support_meta = load_metadata(support_json_path)
    
    # 提取查询集文本特征
    print("提取查询集文本特征...")
    query_descriptions = {
        img_id: meta["description"] 
        for img_id, meta in query_meta.items()
        if "description" in meta and meta["description"].strip()
    }
    query_text_features = extract_text_features(query_descriptions, model, processor)
    
    # 提取支持集文本特征
    print("提取支持集文本特征...")
    support_descriptions = {
        img_id: meta["description"]
        for img_id, meta in support_meta.items()
        if "description" in meta and meta["description"].strip()
    }
    support_text_features = extract_text_features(support_descriptions, model, processor)
    
    # 保存文本特征
    query_features_path = os.path.join(dataset_dir, "clip_query_text_features.pkl")
    support_features_path = os.path.join(dataset_dir, "clip_support_text_features.pkl")
    
    print(f"保存查询集文本特征到: {query_features_path}")
    with open(query_features_path, "wb") as f:
        pickle.dump(query_text_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"保存支持集文本特征到: {support_features_path}")
    with open(support_features_path, "wb") as f:
        pickle.dump(support_text_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 计算相似度
    print("计算文本-文本相似度...")
    sim_data = compute_text_similarity(
        query_text_features,
        support_text_features,
        dataset_dir
    )
    
    # 打印统计信息
    print("\n处理结果摘要：")
    print(f"查询文本特征数: {len(query_text_features)}")
    print(f"支持文本特征数: {len(support_text_features)}")
    print(f"相似度记录数: {len(sim_data)}")
    print(f"单个查询样本关联数: {len(next(iter(sim_data.values())))}")
    print(f"存储位置: {dataset_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='使用CLIP计算查询文本与支持文本的相似度')
    parser.add_argument('--datasets', nargs='+', 
                      default=['Intentonomy', 'EmoSet', 'ArtPhoto', 'EmotionROI'],
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