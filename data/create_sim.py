import os
import json
import pickle
import torch
import clip
from PIL import Image
from tqdm import tqdm

# 配置参数
BASE_IMG_DIR = "/data0/lxy_data"
OUTPUT_DIR = "/data0/lxy_data/Intentonomy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128  # 根据GPU显存调整

def load_metadata(json_path):
    """加载元数据并验证完整性"""
    with open(json_path) as f:
        raw_data = json.load(f)
    
    meta_dict = {}
    for item in raw_data:
        img_id = str(item["img_id"])  # 强制转换为字符串
        rel_path = item["image"][0]   # 获取相对路径
        abs_path = os.path.join(BASE_IMG_DIR, rel_path)
        
        if img_id in meta_dict:
            raise ValueError(f"重复的img_id: {img_id}")
        meta_dict[img_id] = abs_path
    
    return meta_dict  # 返回 {img_id: abs_path}

def extract_features(meta_dict, save_name):
    """生成{img_id: tensor}的特征字典"""
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    
    features_dict = {}
    failed_ids = {}
    items = list(meta_dict.items())
    
    with torch.no_grad():
        # 分批次处理
        for i in tqdm(range(0, len(items), BATCH_SIZE), 
                     desc=f"特征提取 {save_name}"):
            batch = items[i:i+BATCH_SIZE]
            batch_tensors = []
            valid_ids = []
            
            # 预处理阶段
            for img_id, path in batch:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                batch_tensors.append(tensor)
                valid_ids.append(img_id)

            
            if batch_tensors:
                # 批量特征提取
                batch_tensor = torch.cat(batch_tensors)
                batch_features = model.encode_image(batch_tensor)
                
                # 填充字典
                for idx, img_id in enumerate(valid_ids):
                    features_dict[img_id] = batch_features[idx].cpu()
    
    # 保存数据
    with open(os.path.join(OUTPUT_DIR, f"{save_name}_features.pkl"), "wb") as f:
        pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return features_dict

def compute_similarity(query_dict, support_dict):
    """生成嵌套字典的相似度结构"""
    # 准备数据
    query_ids = list(query_dict.keys())
    support_ids = list(support_dict.keys())
    
    # 转换为张量矩阵
    Q = torch.stack([query_dict[q_id] for q_id in query_ids]).to(DEVICE)
    S = torch.stack([support_dict[s_id] for s_id in support_ids]).to(DEVICE)
    
    # 矩阵乘法计算相似度
    with torch.no_grad():
        sim_matrix = torch.mm(Q, S.T).cpu()
    
    # 构建嵌套字典
    similarity_dict = {}
    for i, q_id in tqdm(enumerate(query_ids), desc="构建相似度字典"):
        similarity_dict[q_id] = {
            s_id: sim_matrix[i][j].item() 
            for j, s_id in enumerate(support_ids)
        }
    
    # 保存结果
    with open(os.path.join(OUTPUT_DIR, "cross_similarity.pkl"), "wb") as f:
        pickle.dump(similarity_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return similarity_dict

if __name__ == "__main__":
    # 加载元数据
    query_meta = load_metadata("/data0/lxy_data/Intentonomy/query.json")
    support_meta = load_metadata("/data0/lxy_data/Intentonomy/support.json")
    
    # 特征提取
    query_features = extract_features(query_meta, "query")
    support_features = extract_features(support_meta, "support")
    
    # 计算相似度
    sim_data = compute_similarity(query_features, support_features)
    
    # 打印统计信息
    print("\n处理结果摘要：")
    print(f"有效查询样本: {len(query_features)}")
    print(f"有效支持样本: {len(support_features)}")
    print(f"相似度记录数: {len(sim_data)}")
    print(f"单个查询样本关联数: {len(next(iter(sim_data.values())))}")
    print(f"存储位置: {OUTPUT_DIR}")