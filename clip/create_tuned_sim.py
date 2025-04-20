import os
import json
import pickle
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor
import sys
sys.path.append('..')
from clip.myCLIP import CLIPImageTuner

# 配置参数
BASE_DIR = "/data1/yq_log/IntenICL"
BASE_IMG_DIR = os.path.join(BASE_DIR, "DataSet")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128  # 根据GPU显存调整

def load_metadata(json_path):
    """加载元数据并验证完整性"""
    with open(json_path) as f:
        raw_data = json.load(f)
    
    meta_dict = {}
    for item in raw_data:
        img_id = str(item["img_id"])
        rel_path = item["image"][0]
        abs_path = os.path.join(BASE_IMG_DIR, rel_path)
        
        if img_id in meta_dict:
            raise ValueError(f"重复的img_id: {img_id}")
        meta_dict[img_id] = abs_path
    
    return meta_dict

def extract_features(meta_dict, model, processor, save_dir, save_name):
    """使用微调后的CLIP提取特征"""
    model.eval()
    features_dict = {}
    failed_ids = {}
    items = list(meta_dict.items())
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items), BATCH_SIZE), 
                     desc=f"特征提取 {save_name}"):
            batch = items[i:i+BATCH_SIZE]
            batch_tensors = []
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
                    batch_tensors.append(processed['pixel_values'])
                    valid_ids.append(img_id)
                except Exception as e:
                    failed_ids[img_id] = str(e)
            
            if batch_tensors:
                batch_tensor = torch.cat(batch_tensors).to(DEVICE)
                # 使用微调后的模型提取特征
                # 使用正确的方法从myCLIP.py中的CLIPImageTuner类提取特征
                image_features = model.clip.get_image_features(batch_tensor)
                aligned_features = model.vision_to_semantic(image_features)
                batch_features = F.normalize(aligned_features, p=2, dim=1)
                
                for idx, img_id in enumerate(valid_ids):
                    features_dict[img_id] = batch_features[idx].cpu()
    
    # 保存特征
    feature_path = os.path.join(save_dir, f"ft_{save_name}_features.pkl")
    with open(feature_path, "wb") as f:
        pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 保存失败记录
    if failed_ids:
        print(f"\n处理失败的图像数: {len(failed_ids)}")
        failed_path = os.path.join(save_dir, f"ft_{save_name}_failed.json")
        with open(failed_path, "w") as f:
            json.dump(failed_ids, f, indent=4)
    
    return features_dict

def compute_similarity(query_dict, support_dict, save_dir):
    """计算余弦相似度"""
    query_ids = list(query_dict.keys())
    support_ids = list(support_dict.keys())
    
    Q = torch.stack([query_dict[q_id] for q_id in query_ids]).to(DEVICE)
    S = torch.stack([support_dict[s_id] for s_id in support_ids]).to(DEVICE)
    
    with torch.no_grad():
        sim_matrix = torch.mm(Q, S.T).cpu()
    
    similarity_dict = {}
    for i, q_id in tqdm(enumerate(query_ids), desc="构建相似度字典"):
        similarity_dict[q_id] = {
            s_id: sim_matrix[i][j].item() 
            for j, s_id in enumerate(support_ids)
        }
    
    # 保存相似度
    sim_path = os.path.join(save_dir, "ft_cross_similarity.pkl")
    with open(sim_path, "wb") as f:
        pickle.dump(similarity_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return similarity_dict

def process_dataset(dataset_name, model_path, num_classes):
    """处理单个数据集"""
    print(f"\n处理数据集: {dataset_name}")
    
    # 根据数据集名称生成输出路径
    OUTPUT_DIR = os.path.join(BASE_IMG_DIR, dataset_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    checkpoint = torch.load(model_path)
    model = CLIPImageTuner(num_classes=num_classes, lora_rank=16).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 初始化处理器
    processor = CLIPProcessor.from_pretrained(os.path.join(BASE_DIR, "huggingface/openai/clip-vit-base-patch32"))
    
    # 加载数据
    query_meta = load_metadata(os.path.join(OUTPUT_DIR, "query.json"))
    support_meta = load_metadata(os.path.join(OUTPUT_DIR, "support.json"))
    
    # 提取特征
    query_features = extract_features(query_meta, model, processor, OUTPUT_DIR, "query")
    support_features = extract_features(support_meta, model, processor, OUTPUT_DIR, "support")
    
    # 计算相似度
    sim_data = compute_similarity(query_features, support_features, OUTPUT_DIR)
    
    # 打印统计信息
    print("\n处理结果摘要：")
    print(f"有效查询样本: {len(query_features)}")
    print(f"有效支持样本: {len(support_features)}")
    print(f"相似度记录数: {len(sim_data)}")
    print(f"单个查询样本关联数: {len(next(iter(sim_data.values())))}")
    print(f"存储位置: {OUTPUT_DIR}")

def get_model_path(dataset_name):
    """根据数据集名称生成对应的模型路径"""
    return os.path.join(BASE_DIR, 'ICL_log/intCLIP/2_6_2', f'best_model_{dataset_name}.pth')

def get_num_classes(dataset_name):
    """根据数据集名称返回对应的类别数"""
    num_classes_dict = {
        'Intentonomy': 28,
        'EmoSet': 8,
        'ArtPhoto': 8,
        'EmotionROI': 6
    }
    return num_classes_dict.get(dataset_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['ArtPhoto', 'Intentonomy', 'EmoSet', 'EmotionROI'],
                      choices=['Intentonomy', 'EmoSet', 'ArtPhoto', 'EmotionROI'],
                      help='要处理的数据集列表')
    args = parser.parse_args()
    
    # 根据数据集名称自动生成对应的模型路径和类别数
    model_paths = [get_model_path(dataset) for dataset in args.datasets]
    num_classes = [get_num_classes(dataset) for dataset in args.datasets]
    
    # 检查所有模型文件是否存在
    for dataset, model_path in zip(args.datasets, model_paths):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到数据集 {dataset} 对应的模型文件: {model_path}")
    
    # 处理每个数据集
    for dataset, model_path, n_classes in zip(args.datasets, model_paths, num_classes):
        print(f"\n正在处理数据集: {dataset}")
        print(f"模型路径: {model_path}")
        print(f"类别数: {n_classes}")
        process_dataset(dataset, model_path, n_classes) 