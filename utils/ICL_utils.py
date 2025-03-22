import random
import copy
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import logging


def select_demonstration(support_meta, n_shot, dataset, strategy, query=None, similarity_data=None, support_features=None, query_feature=None):
    if dataset == 'Intentonomy':
        if strategy == 'random':
            n_shot_support_raw = random.sample(support_meta, n_shot)
            n_shot_support = copy.deepcopy(n_shot_support_raw)
        elif strategy == 'similarity':
            n_shot_support = retrieve_similar_demos(query, support_meta, n_shot, similarity_data)
        elif strategy == 'diverse':
            n_shot_support = retrieve_diverse_demos(query, support_meta, n_shot, similarity_data, support_features)
        elif strategy == 'various':
            n_shot_support = retrieve_category_aware_demos(query, support_meta, n_shot, similarity_data, support_features, query_feature)
    else:
        n_shot_support = random.sample(support_meta, n_shot)

    return n_shot_support

def retrieve_similar_demos(query_item, support_meta, n_shot, similarity_data):
    """
    基于预计算相似度的Top-N检索
    
    参数：
        query_item: 查询样本的元数据字典（需包含img_id）
        support_meta: 支持集元数据列表
        n_shot: 需要返回的示例数量
        similarity_data: 预加载的相似度字典
    
    返回：
        list: 排序后的支持集样本列表（按相似度降序）
    """
    # 构建支持集索引
    support_dict = {item["img_id"]: item for item in support_meta}
    
    # 获取当前查询的相似度记录
    query_id = query_item["img_id"]
    if query_id not in similarity_data:
        raise KeyError(f"Query ID {query_id} 不存在于相似度数据中")
    
    # 获取排序后的支持集ID
    similarities = similarity_data[query_id]
    sorted_ids = sorted(similarities.keys(), 
                       key=lambda x: similarities[x], 
                       reverse=True)[:n_shot]
    
    # 按排序结果获取元数据
    selected = []
    for s_id in sorted_ids:
        if s_id in support_dict:
            selected.append(support_dict[s_id])
        else:
            print(f"警告: 支持集ID {s_id} 不存在于元数据中")
    
    return selected[:n_shot]



def retrieve_diverse_demos(query_item, support_meta, n_shot, similarity_data, support_features, pre_select_multiplier=2):
    """
    多样性增强的检索函数
    
    参数：
        query_item: 查询样本元数据字典
        support_meta: 支持集元数据列表
        n_shot: 最终需要返回的示例数量
        similarity_data: 预加载的相似度字典
        support_features: 支持集特征字典 {img_id: feature_vector}
        pre_select_multiplier: 预选样本倍数 (默认2n)
    
    返回：
        list: 聚类筛选后的支持集样本列表
    """
    # 构建支持集索引
    support_dict = {item["img_id"]: item for item in support_meta}
    
    # 获取查询ID
    query_id = query_item["img_id"]
    if query_id not in similarity_data:
        raise KeyError(f"Query ID {query_id} 不存在于相似度数据中")
    
    # 预选初始样本
    pre_select_num = n_shot * pre_select_multiplier
    similarities = similarity_data[query_id]
    
    # 获取预选ID列表
    pre_selected_ids = sorted(similarities.keys(), 
                             key=lambda x: similarities[x], 
                             reverse=True)[:pre_select_num]
    
    
    return cluster_selection(pre_selected_ids, support_dict, support_features, n_shot)

def cluster_selection(candidate_ids, support_dict, support_features, n_shot):
    """聚类筛选核心逻辑"""
    # 获取有效特征
    valid_features = []
    valid_ids = []
    for s_id in candidate_ids:
        if s_id in support_dict and s_id in support_features:
            valid_features.append(support_features[s_id].numpy())
            valid_ids.append(s_id)
    
    # 处理候选不足的情况
    if len(valid_ids) <= n_shot:
        return [support_dict[s_id] for s_id in valid_ids]
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_shot, n_init=10, random_state=0)
    feature_matrix = np.stack(valid_features)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    
    # 选择每个簇中心最近的样本
    selected = []
    for cluster_id in range(n_shot):
        cluster_mask = cluster_labels == cluster_id
        if not np.any(cluster_mask):
            continue
        cluster_features = feature_matrix[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
        best_idx = np.argmin(distances)
        selected.append(valid_ids[cluster_mask.nonzero()[0][best_idx]])
    
    return [support_dict[s_id] for s_id in selected if s_id in support_dict]



def retrieve_category_aware_demos(
    query_item,
    support_meta,
    n_shot,
    similarity_data,
    support_features,
    query_feature = None,
    balance_threshold=0.7,
    min_class_ratio=0.2,
    diversity_strategy="mmr",
    top_k_ratio=5  # 新增：相似性预筛选倍数（默认取5*n_shot）
):
    # 第一阶段：全局相似性筛选（取5*n_shot个最相似样本）
    # 第二阶段：按类别分组并过滤小类别（保留占比≥20%的类别）
    # 第三阶段：模式判断（主类模式/平衡模式）
    # 第四阶段：多样性选择（使用MMR/FPS等算法）

    if n_shot == 1:
        return handle_single_shot(
            query_item, 
            support_meta,
            similarity_data,
            balance_threshold,
            min_class_ratio,
            top_k_ratio
        )
    
    # 元数据预处理
    support_dict = {item["img_id"]: item for item in support_meta}
    feature_lookup = {k: v.numpy() for k, v in support_features.items()}

    # ------ 第一阶段：全局相似性筛选 ------
    # 获取查询与所有支持样本的相似度
    query_id = query_item["img_id"]
    all_similarities = similarity_data.get(query_id, {})
    
    # 按相似度排序并取Top-K（如5*n_shot）
    sorted_sims = sorted(
        all_similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_k = min(len(sorted_sims), top_k_ratio * n_shot)
    pre_filtered_ids = [s_id for s_id, _ in sorted_sims[:top_k]]

    # ------ 第二阶段：按类别分组及过滤 ------
    category_map = defaultdict(list)
    for item in support_meta:
        if item["img_id"] in pre_filtered_ids:
            category_map[item["category"]].append(item["img_id"])
    
    # 过滤小类别（基于预筛选后的分布）
    raw_counts = {k: len(v) for k, v in category_map.items()}
    total = sum(raw_counts.values())
    filtered_categories = {
        k: v for k, v in raw_counts.items()
        if v / total >= min_class_ratio
    }
    if not filtered_categories:
        filtered_categories = raw_counts

    # ------ 第三阶段：模式判断与配额分配 ------
    filtered_total = sum(filtered_categories.values())
    max_proportion = max(v / filtered_total for v in filtered_categories.values())

    if max_proportion > balance_threshold:
        # 主类模式
        main_class = max(filtered_categories, key=lambda k: filtered_categories[k])
        class_ids = category_map[main_class][:2 * n_shot]
    else:
        # 平衡模式：动态分配
        allocated = allocate_quota(filtered_categories, filtered_total, n_shot)
        class_ids = []
        for cls, quota in allocated.items():
            class_ids.extend(category_map[cls][:2 * quota])

    # ------ 第四阶段：多样性选择 ------
    final_selected = select_with_diversity(
        candidate_ids=class_ids,
        support_dict=support_dict,
        features=feature_lookup,
        query_feature=query_feature,
        n_shot=n_shot,
        strategy=diversity_strategy
    )
    return final_selected

def handle_single_shot(
    query_item, 
    support_meta,
    similarity_data,
    balance_threshold,
    min_class_ratio,
    top_k_ratio
):
    """单样本场景优化逻辑"""
    # 获取相似度排序
    query_id = query_item["img_id"]
    all_similarities = similarity_data.get(query_id, {})
    sorted_sims = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
    
    # 预筛选 Top-K 样本
    top_k = min(len(sorted_sims), top_k_ratio)
    pre_filtered_ids = [s_id for s_id, _ in sorted_sims[:top_k]]
    
    # 防御性检查：预筛选为空直接返回 []
    if not pre_filtered_ids:
        return []
    
    # 构建类别分布
    support_dict = {item["img_id"]: item for item in support_meta}
    category_counter = defaultdict(int)
    for s_id in pre_filtered_ids:
        if s_id in support_dict:
            category = support_dict[s_id]["category"]
            category_counter[category] += 1
    
    # 过滤小类别
    total = sum(category_counter.values())
    valid_categories = {}
    if total > 0:
        valid_categories = {
            k: v for k, v in category_counter.items()
            if v / total >= min_class_ratio
        }
    
    # 回退机制：无有效类别时使用原始分布
    if not valid_categories:
        # 直接返回预筛选中的最高相似度样本
        return [support_dict[pre_filtered_ids[0]]] if pre_filtered_ids else []
    
    # 判断主类模式
    total_valid = sum(valid_categories.values())
    max_proportion = max(v / total_valid for v in valid_categories.values())
    
    if max_proportion > balance_threshold:
        main_class = max(valid_categories, key=lambda k: valid_categories[k])
        # 遍历全局排序，但只考虑预筛选样本
        for s_id in pre_filtered_ids:
            if support_dict[s_id]["category"] == main_class:
                return [support_dict[s_id]]
        # 防御性返回：预筛选中的最高相似度
        return [support_dict[pre_filtered_ids[0]]]
    else:
        # 平衡模式：返回全局最高相似度
        return [support_dict[pre_filtered_ids[0]]]

def allocate_quota(category_counts, total, n_shot):
    """改进的配额分配算法"""
    # 计算有效类别
    valid_classes = [k for k, v in category_counts.items() if v > 0]
    num_classes = len(valid_classes)
    
    # 保证每个有效类至少1个
    base_alloc = {cls: 1 for cls in valid_classes}
    remaining = n_shot - num_classes
    
    # 处理名额不足的情况
    if remaining < 0:
        return {cls: 1 for cls in valid_classes[:n_shot]}
    
    # 按比例分配剩余名额
    proportions = {k: v/total for k, v in category_counts.items()}
    sorted_classes = sorted(valid_classes, key=lambda x: -category_counts[x])
    
    for cls in sorted_classes:
        add_num = int(round(proportions[cls] * remaining))
        add_num = min(add_num, remaining)
        base_alloc[cls] += add_num
        remaining -= add_num
        if remaining <= 0:
            break
    
    return {k: v for k, v in base_alloc.items() if v > 0}


def select_with_diversity(candidate_ids, support_dict, features, 
                        query_feature, n_shot, strategy):
    """查询感知的多样性选择"""
    # 特征提取与归一化
    valid_features = []
    valid_ids = []
    for s_id in candidate_ids:
        if s_id in features:
            feat = features[s_id].astype(np.float32)
            feat /= np.linalg.norm(feat)  # L2归一化
            valid_features.append(feat)
            valid_ids.append(s_id)
    
    if len(valid_ids) <= n_shot:
        return [support_dict[s_id] for s_id in valid_ids]
    
    feature_matrix = np.stack(valid_features)
    
    # 策略分派
    if strategy == "fps":
        indices = farthest_point_sampling(feature_matrix, query_feature, n_shot)
    elif strategy == "mmr":
        indices = mmr_selection(feature_matrix, query_feature, n_shot)
    elif strategy == "kmedoids":
        indices = kmedoids_selection(feature_matrix, n_shot)
    else:
        indices = np.arange(len(valid_ids))[:n_shot]
    
    return [support_dict[valid_ids[i]] for i in indices if valid_ids[i] in support_dict]

# ------------ 改进的多样性算法 ------------
def farthest_point_sampling(features, query_feature, n_shot):
    """查询引导的最远点采样"""
    # 初始点：与查询最相似的样本
    sim_to_query = np.dot(features, query_feature)
    selected = [np.argmax(sim_to_query)]
    
    # 迭代选择
    for _ in range(1, n_shot):
        dists = cdist(features[selected], features, metric='cosine')
        min_dists = np.min(dists, axis=0)
        next_point = np.argmax(min_dists)
        selected.append(next_point)
    
    return selected

def mmr_selection(features, query_feature, n_shot, lambda_param=0.6):
    """修正后的MMR算法"""
    query_sim = np.dot(features, query_feature)
    selected = []
    candidates = set(range(len(features)))
    
    while len(selected) < n_shot:
        scores = []
        for i in candidates:
            if not selected:
                score = query_sim[i]
            else:
                max_sim = np.max(np.dot(features[i], features[selected].T))
                score = lambda_param * query_sim[i] - (1 - lambda_param) * max_sim
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best = list(candidates)[best_idx]
        selected.append(best)
        candidates.remove(best)
    
    return selected

def kmedoids_selection(features, n_shot):
    """降维后聚类"""
    from sklearn_extra.cluster import KMedoids
    
    
    # 余弦距离聚类
    kmedoids = KMedoids(n_clusters=n_shot, metric='cosine', 
                       init='k-medoids++', max_iter=20)
    kmedoids.fit(features)
    
    return kmedoids.medoid_indices_.tolist()

def get_task_instruction(args):
    dataset = args.dataset
    description = args.task_description
    if description == 'nothing':
        instr = ''
        return instr
    
    if dataset == 'Intentonomy':
        if description == 'concise':
            instr = 'Select the intent category to which the image belongs from the options below: ["Being physically or personally attractive", "Competing to outperform others", "Communicating with others", "Expressing creativity and uniqueness", "Living a curious, adventurous, and exciting life", "Having a easy life", "Enjoying life", "Appreciating architecture", "Appreciating artistic expression",  "Appreciating cultural heritage", "Being a good parent with emotional closeness to children", "Experiencing happiness", "Working diligently",  "Maintaining balance and harmony", "Keeping healthy", "Being in love", "Having affection for animals", "Inspiring others", "Effectively managing tasks and making plans",  "Appreciating natural beauty", "Having a passion for something", "Being playful", "Sharing feelings", "Maintaining social life and friendships", "Achieving occupational success and job satisfaction", "Teaching others", "Keeping things orderly", "Engaging in personally fulfilling work"]. Your response must strictly follow this format: "The intention of this image is [intent category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    return instr

def format_answer(answer, dataset, query=None):
    if dataset in ['Intentonomy']:
        answer = str(answer)
    return answer
