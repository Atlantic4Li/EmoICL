import os
import json
import pickle
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# 配置参数
BASE_DIR = "/data1/yq_log/IntenICL"
BASE_IMG_DIR = os.path.join(BASE_DIR, "DataSet")
CLIP_MODEL_DIR = os.path.join(BASE_DIR, "huggingface/openai/clip-vit-base-patch32")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128  # 批处理大小

def load_metadata(json_path):
    """加载元数据并验证完整性"""
    with open(json_path) as f:
        raw_data = json.load(f)
    
    meta_dict = {}
    for item in raw_data:
        img_id = str(item["img_id"])
        if "description" in item:
            meta_dict[img_id] = {
                "description": item.get("description", ""),
                "image_path": os.path.join(BASE_IMG_DIR, item["image"][0])
            }
        else:
            meta_dict[img_id] = {
                "image_path": os.path.join(BASE_IMG_DIR, item["image"][0])
            }
    
    return meta_dict

def extract_text_features(descriptions, model, processor):
    """提取文本特征"""
    model.eval()
    text_features_dict = {}
    
    with torch.no_grad():
        for img_id, desc in tqdm(descriptions.items(), desc="提取文本特征"):
            # 处理文本
            inputs = processor(
                text=desc,
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
            text_features_dict[img_id] = text_features[0].cpu()
    
    return text_features_dict

def extract_image_features(image_paths, model, processor):
    """批量提取图像特征"""
    model.eval()
    image_features_dict = {}
    items = list(image_paths.items())
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items), BATCH_SIZE), desc="提取图像特征"):
            batch = items[i:i+BATCH_SIZE]
            batch_images = []
            valid_ids = []
            
            # 预处理阶段
            for img_id, path in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    processed = processor(
                        images=img,
                        return_tensors="pt",
                        do_rescale=False
                    )
                    batch_images.append(processed['pixel_values'])
                    valid_ids.append(img_id)
                except Exception as e:
                    print(f"处理图像失败 {img_id}: {str(e)}")
                    continue
            
            if batch_images:
                # 将批次数据移到GPU
                batch_tensor = torch.cat(batch_images).to(DEVICE)
                
                # 提取特征
                image_features = model.get_image_features(batch_tensor)
                # 归一化
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                # 保存特征
                for idx, img_id in enumerate(valid_ids):
                    image_features_dict[img_id] = image_features[idx].cpu()
    
    return image_features_dict

def compute_cross_similarity(query_text_features, support_image_features, save_dir):
    """计算查询集文本与支持集图像之间的跨模态相似度"""
    query_ids = list(query_text_features.keys())
    support_ids = list(support_image_features.keys())
    
    print(f"有效查询样本数: {len(query_ids)}")
    print(f"有效支持样本数: {len(support_ids)}")
    
    # 准备特征矩阵
    Q = torch.stack([query_text_features[q_id] for q_id in query_ids]).to(DEVICE)
    S = torch.stack([support_image_features[s_id] for s_id in support_ids]).to(DEVICE)
    
    # 检查并打印特征类型
    print(f"查询特征类型: {Q.dtype}")
    print(f"支持特征类型: {S.dtype}")
    
    # 计算相似度
    similarity_dict = {}
    print("计算相似度矩阵...")
    with torch.no_grad():
        # 计算文本-图像相似度
        sim_matrix = torch.mm(Q, S.T).cpu()
        
        # 构建相似度字典
        for i, q_id in tqdm(enumerate(query_ids), desc="构建相似度字典"):
            similarity_dict[q_id] = {
                s_id: sim_matrix[i][j].item() 
                for j, s_id in enumerate(support_ids)
            }
    
    # 保存相似度
    sim_path = os.path.join(save_dir, "clip_text_image_similarity.pkl")
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
    }
    query_text_features = extract_text_features(query_descriptions, model, processor)
    
    # 保存查询集文本特征
    text_features_path = os.path.join(dataset_dir, "query_text_features.pkl")
    print(f"保存查询集文本特征到: {text_features_path}")
    with open(text_features_path, "wb") as f:
        pickle.dump(query_text_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 提取支持集图像特征
    print("提取支持集图像特征...")
    support_image_paths = {
        img_id: meta["image_path"]
        for img_id, meta in support_meta.items()
    }
    support_image_features = extract_image_features(support_image_paths, model, processor)
    
    # 计算相似度
    print("计算跨模态相似度...")
    sim_data = compute_cross_similarity(
        query_text_features,
        support_image_features,
        dataset_dir
    )
    
    # 打印统计信息
    print("\n处理结果摘要：")
    print(f"查询文本特征数: {len(query_text_features)}")
    print(f"支持图像特征数: {len(support_image_features)}")
    print(f"相似度记录数: {len(sim_data)}")
    print(f"单个查询样本关联数: {len(next(iter(sim_data.values())))}")
    print(f"存储位置: {dataset_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='计算查询文本与支持图像的跨模态相似度')
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