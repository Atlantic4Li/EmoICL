import os
import json
import pickle
import torch
import clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# 配置参数
BASE_IMG_DIR = "/data1/yq_log/IntenICL/DataSet"
OUTPUT_DIR = "/data1/yq_log/IntenICL/DataSet/Oxford-IIIT_Pet"
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
    """生成{img_id: tensor}的特征字典（已添加特征归一化）"""
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
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                    batch_tensors.append(tensor)
                    valid_ids.append(img_id)
                except Exception as e:
                    failed_ids[img_id] = str(e)
            
            if batch_tensors:
                # 批量特征提取
                batch_tensor = torch.cat(batch_tensors)
                batch_features = model.encode_image(batch_tensor)
                # 关键修改：添加L2归一化
                batch_features = F.normalize(batch_features, p=2, dim=-1)
                
                # 填充字典
                for idx, img_id in enumerate(valid_ids):
                    features_dict[img_id] = batch_features[idx].cpu()
    
    # 保存数据
    with open(os.path.join(OUTPUT_DIR, f"{save_name}_features.pkl"), "wb") as f:
        pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return features_dict

def compute_similarity(query_dict, support_dict):
    """生成嵌套字典的相似度结构（特征已归一化，直接点积即余弦相似度）"""
    # 准备数据
    query_ids = list(query_dict.keys())
    support_ids = list(support_dict.keys())
    
    # 转换为张量矩阵
    Q = torch.stack([query_dict[q_id] for q_id in query_ids]).to(DEVICE)
    S = torch.stack([support_dict[s_id] for s_id in support_ids]).to(DEVICE)
    
    # 矩阵乘法计算余弦相似度
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
    query_meta = load_metadata("/data1/yq_log/IntenICL/DataSet/Oxford-IIIT_Pet/query.json")
    support_meta = load_metadata("/data1/yq_log/IntenICL/DataSet/Oxford-IIIT_Pet/support.json")
    
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


# import os
# import torch
# import pickle
# import numpy as np

# # 文件路径
# FEATURE_DIR = "/data0/lxy_data/Intentonomy"  # 替换为你的特征保存路径
# QUERY_FEATURES = os.path.join(FEATURE_DIR, "query_features.pkl")
# SUPPORT_FEATURES = os.path.join(FEATURE_DIR, "support_features.pkl")

# def load_features(filepath):
#     """加载特征文件"""
#     with open(filepath, "rb") as f:
#         features = pickle.load(f)
#     return features

# def check_normalization(features):
#     """检查特征是否归一化"""
#     all_normalized = True
#     norms = []
    
#     for img_id, vec in features.items():
#         # 将特征转换为张量并确保为 float32
#         tensor = torch.tensor(vec).to(dtype=torch.float32)

#         # 计算 L2 范数
#         norm = torch.norm(tensor)
#         norms.append(norm.item())
        
#         # 确保数据类型一致后再比较
#         if not torch.isclose(norm.float(), torch.tensor(1.0, dtype=torch.float32), atol=1e-4):
#             all_normalized = False
#             print(f"⚠️ 特征未归一化: {img_id}, 范数 = {norm:.6f}")

#     # 统计信息
#     print("\n✅ 特征归一化检查完成")
#     print(f"样本数量: {len(norms)}")
#     print(f"平均范数: {np.mean(norms):.6f}")
#     print(f"范数范围: [{min(norms):.6f}, {max(norms):.6f}]")

#     if all_normalized:
#         print("🎯 所有特征均已归一化！")
#     else:
#         print("⚠️ 存在未归一化特征！")

# # 加载特征
# query_features = load_features(QUERY_FEATURES)
# support_features = load_features(SUPPORT_FEATURES)

# # 检查特征归一化
# print("\n🔍 检查查询集特征归一化...")
# check_normalization(query_features)

# print("\n🔍 检查支持集特征归一化...")
# check_normalization(support_features)
