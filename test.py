# Attractive,BeatCompete,Communicate,CreativeUnique,CuriousAdventurousExcitingLife,EasyLife,EnjoyLife,FineDesignLearnArt-Arch,FineDesignLearnArt-Art,FineDesignLearnArt-Culture,GoodParentEmoCloseChild,Happy,HardWorking,Harmony,Health,InLove,InLoveAnimal,InspirOthrs,ManagableMakePlan,NatBeauty,PassionAbSmthing,Playful,ShareFeelings,SocialLifeFriendship,SuccInOccupHavGdJob,TchOthrs,ThngsInOrdr,WorkILike


# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image

# # 设备选择
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 加载本地 CLIP 模型
# model_path = "/data0/lxy_data/huggingface/clip-vit-base-patch32"
# model = CLIPModel.from_pretrained(model_path).to(device)
# processor = CLIPProcessor.from_pretrained(model_path)

# # 读取并预处理图像
# image_path = "/home/lixiyang/ICL/IntentICL/OIP.jpg"  # 替换为你的本地图片路径
# image = Image.open(image_path)

# # 定义文本描述
# texts = ["sad", "A happy puppy", "a picture of a person"]

# # 预处理输入
# inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

# # 计算图像和文本的特征
# with torch.no_grad():
#     outputs = model(**inputs)
#     image_features = outputs.image_embeds
#     text_features = outputs.text_embeds

# # 归一化向量
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)

# # 计算相似度
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# # 输出匹配结果
# best_match_idx = similarity.argmax().item()
# print(f"The most relevant description: {texts[best_match_idx]}")


# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import torch

# # 设置本地模型路径
# model_path = "/data0/lxy_data/huggingface/blip-image-captioning-base"

# # 加载本地 BLIP 处理器和模型
# processor = BlipProcessor.from_pretrained(model_path)
# model = BlipForConditionalGeneration.from_pretrained(model_path)

# # 读取图片
# image_path = "/data0/lxy_data/Intentonomy/images/0ad40cec6dc0ae553e2b8bcfb6241b02.jpg"  # 替换为你的图片路径
# image = Image.open(image_path).convert('RGB')

# # 处理图片
# inputs = processor(image, return_tensors="pt")

# # 生成图片描述
# with torch.no_grad():
#     out = model.generate(**inputs, max_new_tokens=100)

# caption = processor.batch_decode(out, skip_special_tokens=True)[0]
# print("生成的图片描述:", caption)



# /data0/lxy_data/huggingface/blip-vqa-base

# from transformers import BlipProcessor, BlipForQuestionAnswering
# from PIL import Image

# # 加载本地模型
# model_path = '/data0/lxy_data/huggingface/blip-vqa-base'
# processor = BlipProcessor.from_pretrained(model_path)
# model = BlipForQuestionAnswering.from_pretrained(model_path)

# # 加载图像
# image_path = '/data0/lxy_data/Intentonomy/images/0a3e20fb82db4cb5b32c349aaf0e7f0b.jpg'  # 替换为您的图像路径
# image = Image.open(image_path).convert('RGB')

# # 提问
# question = "Why is this image classified as attractive? "  # 替换为您的问题

# # 处理输入
# inputs = processor(image, question, return_tensors="pt")

# # 生成回答
# out = model.generate(**inputs, max_new_tokens=100)
# answer = processor.batch_decode(out, skip_special_tokens=True)[0]

# print("模型回答:", answer)

# import json
# import csv
# from pathlib import Path

# def process_test_val(json_path):
#     """处理测试集和验证集"""
#     with open(json_path) as f:
#         data = json.load(f)
    
#     results = []
#     for ann in data['annotations']:
#         if len(ann['category_ids']) == 1:
#             results.append((ann['image_id'], ann['category_ids'][0]))
#     return results

# def process_train(json_path):
#     """处理训练集"""
#     with open(json_path) as f:
#         data = json.load(f)
    
#     results = []
#     for ann in data['annotations']:
#         probs = ann.get('category_ids_softprob', [])
#         if not probs:
#             continue
            
#         max_prob = max(probs)
#         if max_prob > 0.5 and probs.count(max_prob) == 1:
#             category = probs.index(max_prob)
#             results.append((ann['image_id'], category))
#     return results

# def main():
#     base_path = Path("/data0/lxy_data/Intentonomy/annotations")
#     output_path = base_path.parent / "dataset_categories.csv"

#     # 处理所有数据集
#     all_data = []
#     all_data.extend(process_test_val(base_path / "intentonomy_test2020.json"))
#     all_data.extend(process_test_val(base_path / "intentonomy_val2020.json"))
#     all_data.extend(process_train(base_path / "intentonomy_train2020.json"))

#     # 写入CSV文件
#     with open(output_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["id", "image_id", "category"])
        
#         # 生成自增ID
#         for idx, (img_id, category) in enumerate(all_data, 1):
#             writer.writerow([idx, img_id, category])

#     print(f"处理完成！共生成 {len(all_data)} 条记录")
#     print(f"结果已保存至：{output_path}")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# from pathlib import Path

# # 读取数据
# csv_path = Path("/data0/lxy_data/Intentonomy/dataset_categories.csv")
# df = pd.read_csv(csv_path)

# # 统计类别分布
# class_dist = df['category'].value_counts().reset_index()
# class_dist.columns = ['category', 'count']

# # 打印统计结果
# print("类别分布统计：")
# print(class_dist)
# print("\n统计摘要：")
# print(f"总类别数：{len(class_dist)}")
# print(f"最多样本类别：{class_dist.iloc[0]['category']}（{class_dist.iloc[0]['count']}个）")
# print(f"最少样本类别：{class_dist.iloc[-1]['category']}（{class_dist.iloc[-1]['count']}个）")

# # 保存统计结果
# output_path = csv_path.parent / "class_distribution.csv"
# class_dist.to_csv(output_path, index=False)
# print(f"\n统计结果已保存至：{output_path}")


# import pandas as pd
# import numpy as np
# from pathlib import Path

# # 设置随机种子保证可重复性
# SEED = 42
# np.random.seed(SEED)

# def stratified_split(df):
#     """执行分层划分"""
#     train_dfs = []
#     test_dfs = []
    
#     # 按类别分组处理
#     for category, group in df.groupby('category'):
#         indices = group.index.tolist()
#         np.random.shuffle(indices)
        
#         split_point = len(indices) // 2
        
#         test_dfs.append(df.loc[indices[:split_point]])
#         train_dfs.append(df.loc[indices[split_point:]])
    
#     return pd.concat(train_dfs), pd.concat(test_dfs)

# # 读取数据
# csv_path = Path("/data0/lxy_data/Intentonomy/dataset_categories.csv")
# df = pd.read_csv(csv_path)

# # 执行分层划分
# train_df, test_df = stratified_split(df)

# # 生成新ID并保存（关键修正部分）
# def save_dataset(df, path):
#     df = df.reset_index(drop=True)  # 清除原有索引
#     df = df.reset_index()           # 创建新索引
#     df = df.rename(columns={'index':'id'})
#     df['id'] += 1                   # ID从1开始
#     df[['id', 'image_id', 'category']].to_csv(path, index=False)

# # 保存结果
# output_dir = csv_path.parent
# save_dataset(train_df, output_dir / "train_dataset.csv")
# save_dataset(test_df, output_dir / "test_dataset.csv")

# # 验证文件格式
# print("训练集前两行示例：")
# print(pd.read_csv(output_dir / "train_dataset.csv").head(2))
# print("\n测试集前两行示例：")
# print(pd.read_csv(output_dir / "test_dataset.csv").head(2))

# import pandas as pd
# from pathlib import Path

# def fix_csv_columns(file_path):
#     """修复CSV文件列问题"""
#     # 读取CSV文件（自动检测列名）
#     df = pd.read_csv(file_path)
    
#     # 删除第二列（索引1的列）
#     df = df.drop(df.columns[1], axis=1)
    
#     # 重命名第一列为id
#     df = df.rename(columns={df.columns[0]: 'id'})
    
#     # 保存修复后的文件（覆盖原文件）
#     df.to_csv(file_path, index=False)
#     print(f"已修复 {file_path} | 当前列：{list(df.columns)}")

# # 设置文件路径
# base_dir = Path("/data0/lxy_data/Intentonomy")
# files_to_fix = [
#     base_dir / "train_dataset.csv",
#     base_dir / "test_dataset.csv"
# ]

# # 执行修复
# for file_path in files_to_fix:
#     if file_path.exists():
#         fix_csv_columns(file_path)
#     else:
#         print(f"文件不存在：{file_path}")

# import csv
# import json

# "Being physically or personally attractive", "Competing to outperform others", "Communicating with others", "Expressing creativity and uniqueness",
#     "Living a curious, adventurous, and exciting life", "Having a easy life", "Enjoying life", "Appreciating architecture", "Appreciating artistic expression", 
#     "Appreciating cultural heritage", "Being a good parent with emotional closeness to children", "Experiencing happiness", "Working diligently", 
#     "Maintaining balance and harmony", "Keeping healthy", "Being in love", "Having affection for animals", "Inspiring others", "Effectively managing tasks and making plans", 
#     "Appreciating natural beauty", "Having a passion for something", "Being playful", "Sharing feelings", "Maintaining social life and friendships", 
#     "Achieving occupational success and job satisfaction", "Teaching others", "Keeping things orderly", "Engaging in personally fulfilling work"

# 需要您提供实际的category映射关系
# category_map = {
#     0: "Being physically or personally attractive",
#     1: "Competing to outperform others",
#     2: "Communicating with others",
#     3: "Expressing creativity and uniqueness",
#     4: "Living a curious, adventurous, and exciting life",
#     5: "Having a easy life",
#     6: "Enjoying life",
#     7: "Appreciating architecture",
#     8: "Appreciating artistic expression",
#     9: "Appreciating cultural heritage",
#     10: "Being a good parent with emotional closeness to children",
#     11: "Experiencing happiness",
#     12: "Working diligently",
#     13: "Maintaining balance and harmony",
#     14: "Keeping healthy",
#     15: "Being in love",
#     16: "Having affection for animals",
#     17: "Inspiring others",
#     18: "Effectively managing tasks and making plans",
#     19: "Appreciating natural beauty",
#     20: "Having a passion for something",
#     21: "Being playful",
#     22: "Sharing feelings",
#     23: "Maintaining social life and friendships",
#     24: "Achieving occupational success and job satisfaction",
#     25: "Teaching others",
#     26: "Keeping things orderly",
#     27: "Engaging in personally fulfilling work"
# }

# def csv_to_json(csv_path, json_path):
#     result = []
    
#     with open(csv_path, 'r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
        
#         for row in csv_reader:
#             image_id = row['image_id']
#             category = int(row['category'])
            
#             # 构建JSON条目
#             entry = {
#                 "id": f"Intentonomy_images_{image_id}",
#                 "image": [f"Intentonomy/images/{image_id}.jpg"],
#                 "question": "What is the intention of this image?",
#                 "answer": "The intention of this image is " + category_map.get(category, "unknown_intention"),
#                 "img_id": image_id,
#                 "category": category
#             }
#             result.append(entry)
    
#     with open(json_path, 'w') as json_file:
#         json.dump(result, json_file, indent=2)

# # 使用示例
# csv_to_json('/data0/lxy_data/Intentonomy/query.csv', '/data0/lxy_data/Intentonomy/query_1.json')

# ====== 查看pkl文件内容工具 ======
import pickle

def view_pkl(path, max_items=5):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {type(data)}")
    if isinstance(data, dict):
        print(f"dict keys (show {max_items}):", list(data.keys())[:max_items])
        for k in list(data.keys())[:max_items]:
            print(f"  {k}: {type(data[k])}, sample: {str(data[k])[:200]}")
    elif isinstance(data, list):
        print(f"list length: {len(data)}")
        for i, v in enumerate(data[:max_items]):
            print(f"  [{i}]: {type(v)}, sample: {str(v)[:200]}")
    else:
        print("data sample:", str(data)[:500])

# 用法示例：
data1 = view_pkl('/data1/yq_log/IntenICL/DataSet/EmoSet/similarity/clip_text_image_similarity.pkl')
data2 = view_pkl('/data1/yq_log/IntenICL/DataSet/EmoSet/similarity/image_image_similarity.pkl')
data3 = view_pkl('/data1/yq_log/IntenICL/DataSet/EmoSet/similarity/text_text_similarity.pkl')
data4 = 1