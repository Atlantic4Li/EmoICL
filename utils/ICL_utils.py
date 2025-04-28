import random
import copy
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import math
import torch
from transformers import CLIPProcessor, CLIPModel


def select_demonstration(support_meta, n_shot, dataset, strategy, query=None, similarity_data=None, support_features=None, query_feature=None, balance_threshold=0.7, cross_modal_similarity=None):
    if dataset in ['Intentonomy','EmotionROI','ArtPhoto','EmoSet','StanfordCars','StanfordDogs','CUB_200_2011','OxfordFlowers17','Oxford-IIIT_Pet']:   
        if strategy == 'random':    # 随机选择
            n_shot_support_raw = random.sample(support_meta, n_shot)
            n_shot_support = copy.deepcopy(n_shot_support_raw)
        elif strategy == 'similarity': # 验证相似性排序的影响
            n_shot_support = retrieve_similar_demos(query, support_meta, n_shot, similarity_data)
        elif strategy == 'similarity_reverse': # 验证相似性排序的影响-逆序
            n_shot_support = retrieve_similar_demos(query, support_meta, n_shot, similarity_data, reverse_results=True)
        elif strategy == 'various': # 验证类别信息+多样性排序
            n_shot_support = retrieve_category_aware_demos(query, support_meta, n_shot, similarity_data, support_features, query_feature, balance_threshold)
        elif strategy == 'whole':  # 主实验
            n_shot_support = retrieve_hybrid_demos(query, support_meta, n_shot, similarity_data, cross_modal_similarity, support_features, query_feature, balance_threshold)
        elif strategy == 'whole_reverse':  # 主实验-逆序
            n_shot_support = retrieve_hybrid_demos(query, support_meta, n_shot, similarity_data, cross_modal_similarity, support_features, query_feature, balance_threshold, reverse_results=True)
        elif strategy == 'test_diverse': # 验证去掉类别信息后，多样性排序有效性
            n_shot_support = retrieve_diverse_demos(query, support_meta, n_shot, similarity_data, support_features, query_feature)
        elif strategy == 'test_same_similarity': # 与查询相同类别但按相似性排序
            n_shot_support = retrieve_same_class_demos(query, support_meta, n_shot, similarity_data)
        elif strategy == 'test_same_random': # 与查询相同类别但随机排序
            n_shot_support = retrieve_random_demos(support_meta, n_shot, query)
        elif strategy == 'test_text_similarity': # 使用文本相似度，验证重排有效性
            n_shot_support = retrieve_text_demos(query, support_meta, n_shot, similarity_data, cross_modal_similarity)
    # else:
    #     n_shot_support = random.sample(support_meta, n_shot)

    return n_shot_support

def retrieve_similar_demos(query_item, support_meta, n_shot, similarity_data, reverse_results=False):
    """
    基于预计算相似度的Top-N检索
    
    参数：
        query_item: 查询样本的元数据字典（需包含img_id）
        support_meta: 支持集元数据列表
        n_shot: 需要返回的示例数量
        similarity_data: 预加载的相似度字典
        reverse_results: 是否逆序排列返回的示例
    
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
                       reverse=True)
    
    # 按排序结果获取元数据
    selected = []
    for s_id in sorted_ids:
        if s_id in support_dict:
            selected.append(support_dict[s_id])
            if len(selected) >= n_shot:
                break
    
    # 根据reverse_results参数决定是否逆序返回
    return selected[::-1] if reverse_results else selected

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
    all_similarities = {k: v for k, v in similarity_data.get(query_id, {}).items() if k in support_dict}
    
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
    support_dict = {item["img_id"]: item for item in support_meta}
    # all_similarities = similarity_data.get(query_id, {})
    all_similarities = {k: v for k, v in similarity_data.get(query_id, {}).items() if k in support_dict}
    sorted_sims = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
    
    # 预筛选 Top-K 样本
    top_k = min(len(sorted_sims), top_k_ratio)
    pre_filtered_ids = [s_id for s_id, _ in sorted_sims[:top_k]]
    
    # 防御性检查：预筛选为空直接返回 []
    if not pre_filtered_ids:
        return []
    
    # 构建类别分布
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
    """改进版配额分配算法：最大类优先 + 循环分配剩余名额"""
    
    # 筛选有效类别（样本数>0的类别）
    valid_classes = [k for k, v in category_counts.items() if v > 0]
    num_classes = len(valid_classes)
    
    # 场景1：名额不足时，按样本数降序分配
    if n_shot <= num_classes:
        sorted_classes = sorted(valid_classes, key=lambda x: -category_counts[x])
        return {cls: 1 for cls in sorted_classes[:n_shot]}
    
    # 初始化基础分配
    base_alloc = {cls: 1 for cls in valid_classes}
    remaining = n_shot - num_classes
    sorted_classes = sorted(valid_classes, key=lambda x: -category_counts[x])
    proportions = {k: v/total for k, v in category_counts.items()}
    
    # --- 第一阶段：最大类向上取整 ---
    if sorted_classes:
        max_cls = sorted_classes[0]
        max_alloc = math.ceil(proportions[max_cls] * remaining)
        allocated = min(max_alloc, remaining)
        base_alloc[max_cls] += allocated
        remaining -= allocated
    
    # --- 第二阶段：其他类四舍五入 ---
    for cls in sorted_classes[1:]:
        if remaining <= 0: break
        alloc = round(proportions[cls] * remaining)
        alloc = min(max(alloc, 0), remaining)  # 防御负值
        base_alloc[cls] += alloc
        remaining -= alloc
    
    # --- 第三阶段：循环分配剩余名额 ---
    if remaining > 0:
        index = 0
        while remaining > 0:
            current_cls = sorted_classes[index % len(sorted_classes)]
            base_alloc[current_cls] += 1
            remaining -= 1
            index += 1
    
    return {k: v for k, v in base_alloc.items() if v > 0}


def retrieve_same_class_demos(query_item, support_meta, n_shot, similarity_data):
    """
    选择与查询图像相同类别的示例
    
    参数：
        query_item: 查询样本的元数据字典（需包含img_id和category）
        support_meta: 支持集元数据列表
        n_shot: 需要返回的示例数量
        similarity_data: 预加载的相似度字典（用于在相同类别内排序）
    
    返回：
        list: 相同类别内按相似度排序的示例列表
    """
    # 获取查询图像的类别
    query_category = query_item["category"]
    
    # 筛选相同类别的样本
    same_class_samples = [item for item in support_meta if item["category"] == query_category]
    
    # 如果相同类别的样本数量不足，返回所有可用的样本
    if len(same_class_samples) <= n_shot:
        return same_class_samples
    
    # 在相同类别内按相似度排序
    query_id = query_item["img_id"]
    if query_id in similarity_data:
        # 构建支持集索引
        support_dict = {item["img_id"]: item for item in same_class_samples}
        
        # 获取相似度并排序
        similarities = {s_id: similarity_data[query_id].get(s_id, 0) 
                       for s_id in support_dict.keys()}
        sorted_ids = sorted(similarities.keys(), 
                          key=lambda x: similarities[x], 
                          reverse=True)[:n_shot]
        
        # 返回排序后的样本
        return [support_dict[s_id] for s_id in sorted_ids]
    else:
        # 如果没有相似度数据，随机选择
        return random.sample(same_class_samples, n_shot)

def retrieve_random_demos(support_meta, n_shot, query_item=None):
    """
    纯随机选择示例，如果提供了query_item，则在同类别样本中随机选择
    
    参数：
        support_meta: 支持集元数据列表
        n_shot: 需要返回的示例数量
        query_item: 查询样本的元数据字典（可选，如果提供则只选择同类别样本）
    
    返回：
        list: 随机选择的示例列表
    """
    # 如果提供了query_item，则只选择同类别样本
    if query_item is not None:
        query_category = query_item["category"]
        same_class_samples = [item for item in support_meta if item["category"] == query_category]
        if len(same_class_samples) <= n_shot:
            return same_class_samples
        return random.sample(same_class_samples, n_shot)
    
    # 如果没有提供query_item，则从所有样本中随机选择
    if len(support_meta) <= n_shot:
        return support_meta
    return random.sample(support_meta, n_shot)

def retrieve_diverse_demos(
    query_item,
    support_meta,
    n_shot,
    similarity_data,
    support_features,
    query_feature = None,
    top_k_ratio=2,  # 相似性预筛选倍数
    diversity_strategy="mmr"  # 多样性策略选择
):
    """
    基于相似度和多样性的检索策略，不考虑类别信息
    
    参数：
        query_item: 查询样本的元数据字典
        support_meta: 支持集元数据列表
        n_shot: 需要返回的示例数量
        similarity_data: 预加载的相似度字典
        support_features: 支持集特征字典
        query_feature: 查询样本特征
        top_k_ratio: 相似性预筛选倍数
        diversity_strategy: 多样性选择策略
    
    返回：
        list: 多样化的示例列表
    """
    if n_shot == 1:
        return retrieve_similar_demos(query_item, support_meta, n_shot, similarity_data)
    
    # 元数据预处理
    support_dict = {item["img_id"]: item for item in support_meta}
    feature_lookup = {k: v.numpy() for k, v in support_features.items()}

    # 第一阶段：全局相似性筛选
    query_id = query_item["img_id"]
    all_similarities = similarity_data.get(query_id, {})
    
    # 按相似度排序并取Top-K
    sorted_sims = sorted(
        all_similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_k = min(len(sorted_sims), top_k_ratio * n_shot)
    pre_filtered_ids = [s_id for s_id, _ in sorted_sims[:top_k]]

    # 第二阶段：直接进行多样性选择
    final_selected = select_with_diversity(
        candidate_ids=pre_filtered_ids,
        support_dict=support_dict,
        features=feature_lookup,
        query_feature=query_feature,
        n_shot=n_shot,
        strategy=diversity_strategy
    )
    
    return final_selected

def retrieve_hybrid_demos(query, support_meta, n_shot, visual_similarity, cross_modal_similarity, support_features, query_feature, balance_threshold, min_class_ratio=0.2, reverse_results=False):
    """
    三阶段检索策略：
    1. 第一阶段：使用视觉相似度选择5*n_shot个候选
    2. 第二阶段：根据类别分布判断主类/平衡模式，使用跨模态相似度重排序选择2*n_shot个候选
    3. 第三阶段：使用多样性策略选择最终的n_shot个样本
    
    参数：
        query: 查询样本信息
        support_meta: 支持集元数据
        n_shot: 需要检索的样本数量
        visual_similarity: 视觉相似度矩阵
        cross_modal_similarity: 语义相似度矩阵
        support_features: 支持集特征
        query_feature: 查询特征
        balance_threshold: 主类模式阈值
        min_class_ratio: 最小类别占比阈值(新增参数)
        reverse_results: 是否逆序排列返回的示例
    """
    # 将support_meta转换为字典形式，以img_id为键
    support_dict = {item['img_id']: item for item in support_meta}
    
    # 单样本特殊处理
    if n_shot == 1:
        query_id = query['img_id']
        
        # 第一阶段：视觉相似度预筛选
        visual_sims = visual_similarity[query_id]
        sorted_visual = sorted(visual_sims.items(), key=lambda x: x[1], reverse=True)
        top_k = min(5, len(sorted_visual))
        pre_filtered = {
            s_id: sim_score 
            for s_id, sim_score in sorted_visual[:top_k]
            if s_id in support_dict  # 确保样本在support_meta中
        }
        
        # 计算综合得分
        candidate_scores = {}
        for cand_id, visual_score in pre_filtered.items():
            if cand_id in cross_modal_similarity[query_id]:
                cross_score = cross_modal_similarity[query_id][cand_id]
                combined_score = 0.7 * visual_score + 0.3 * cross_score
                candidate_scores[cand_id] = combined_score
        
        # 按类别统计
        category_counter = defaultdict(list)
        for s_id, score in candidate_scores.items():
            category = support_dict[s_id]['category']
            category_counter[category].append((s_id, score))
        
        # 计算类别分布
        category_counts = {k: len(v) for k, v in category_counter.items()}
        total = sum(category_counts.values())
        
        if total == 0:
            return []
        
        # 过滤小类别（新增）
        valid_categories = {}
        if total > 0:
            valid_categories = {
                k: v for k, v in category_counts.items()
                if v / total >= min_class_ratio
            }
            
        # 回退机制：如果过滤后没有有效类别，使用原始分布
        if not valid_categories:
            valid_categories = category_counts
            
        # 判断是否存在主类
        total_valid = sum(valid_categories.values())
        max_proportion = max(v / total_valid for v in valid_categories.values()) if total_valid > 0 else 0
        
        if max_proportion > balance_threshold:
            # 主类模式：选择主类中得分最高的样本
            main_class = max(valid_categories.items(), key=lambda x: x[1])[0]
            main_class_samples = category_counter[main_class]
            best_sample_id = max(main_class_samples, key=lambda x: x[1])[0]
            return [support_dict[best_sample_id]]
        else:
            # 平衡模式：直接选择得分最高的样本
            best_sample_id = max(candidate_scores.items(), key=lambda x: x[1])[0]
            return [support_dict[best_sample_id]]
    
    # 多样本处理逻辑
    query_id = query['img_id']
    
    # ---------- 第一阶段：视觉相似度选择 ----------
    first_stage_size = min(5 * n_shot, len(support_dict))
    visual_sims = visual_similarity[query_id]
    sorted_visual = sorted(visual_sims.items(), key=lambda x: x[1], reverse=True)
    
    first_stage_candidates = {
        support_id: sim_score 
        for support_id, sim_score in sorted_visual[:first_stage_size]
        if support_id in support_dict  # 确保样本在support_meta中
    }
    
    # ---------- 第二阶段：跨模态重排序和类别平衡 ----------
    second_stage_size = 2 * n_shot
    cross_sims = cross_modal_similarity[query_id]
    
    candidate_scores = {}
    for cand_id in first_stage_candidates:
        if cand_id in cross_sims:
            visual_score = first_stage_candidates[cand_id]
            cross_score = cross_sims[cand_id]
            combined_score = 0.7 * visual_score + 0.3 * cross_score
            candidate_scores[cand_id] = combined_score
    
    category_groups = defaultdict(list)
    for support_id, score in candidate_scores.items():
        category = support_dict[support_id]['category']
        category_groups[category].append((support_id, score))
    
    # 计算类别分布
    category_counts = {k: len(v) for k, v in category_groups.items()}
    total_samples = sum(category_counts.values())
    
    if total_samples == 0:
        return []
    
    # 过滤小类别（新增）
    filtered_categories = {
        k: v for k, v in category_counts.items()
        if v / total_samples >= min_class_ratio
    }
    
    # 回退机制：如果过滤后没有有效类别，使用原始分布
    if not filtered_categories:
        filtered_categories = category_counts
        
    # 重新计算总数和最大占比
    filtered_total = sum(filtered_categories.values())
    max_proportion = max(v / filtered_total for v in filtered_categories.values())
    
    second_stage_candidates = {}
    if max_proportion > balance_threshold:
        main_class = max(filtered_categories.items(), key=lambda x: x[1])[0]
        main_class_samples = sorted(category_groups[main_class], key=lambda x: x[1], reverse=True)
        
        for support_id, score in main_class_samples[:second_stage_size]:
            second_stage_candidates[support_id] = support_dict[support_id]
    else:
        allocated = allocate_quota(filtered_categories, filtered_total, second_stage_size)
        
        for category, quota in allocated.items():
            sorted_samples = sorted(category_groups[category], key=lambda x: x[1], reverse=True)
            for support_id, score in sorted_samples[:quota]:
                second_stage_candidates[support_id] = support_dict[support_id]
    
    # ---------- 第三阶段：多样性选择 ----------
    feature_lookup = {k: v.numpy() for k, v in support_features.items()}
    final_selected = select_with_diversity(
        candidate_ids=list(second_stage_candidates.keys()),
        support_dict=support_dict,
        features=feature_lookup,
        query_feature=query_feature,
        n_shot=n_shot,
        strategy="mmr"
    )
    
    # 根据reverse_results参数决定是否逆序返回
    return final_selected[::-1] if reverse_results else final_selected

def retrieve_text_demos(query, support_meta, n_shot, visual_similarity, cross_modal_similarity):
    """
    两阶段检索策略：
    1. 第一阶段：使用视觉相似度选择5*n_shot个候选
    2. 第二阶段：使用视觉相似度和语义相似度的加权平均重排序，选择前n个样本
    
    参数：
        query: 查询样本信息
        support_meta: 支持集元数据
        n_shot: 需要检索的样本数量
        visual_similarity: 视觉相似度矩阵
        cross_modal_similarity: 语义相似度矩阵
    """
    query_id = query['img_id']
    
    # 第一阶段：视觉相似度预筛选
    visual_sims = visual_similarity[query_id]
    sorted_visual = sorted(visual_sims.items(), key=lambda x: x[1], reverse=True)
    pre_filtered = {
        support_id: sim_score 
        for support_id, sim_score in sorted_visual[:5 * n_shot]
    }
    
    # 第二阶段：加权平均重排序
    if cross_modal_similarity is not None and query_id in cross_modal_similarity:
        cross_sims = cross_modal_similarity[query_id]
        # 计算加权平均分数
        candidate_scores = {}
        for s_id, visual_score in pre_filtered.items():
            if s_id in cross_sims:
                cross_score = cross_sims[s_id]
                combined_score = 0.9 * visual_score + 0.1 * cross_score
                candidate_scores[s_id] = combined_score
        
        # 按加权平均分数排序
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        selected_ids = [s_id for s_id, _ in sorted_candidates[:n_shot]]
    else:
        # 如果没有语义相似度，就直接用视觉相似度的排序
        selected_ids = list(pre_filtered.keys())[:n_shot]
    
    support_dict = {item["img_id"]: item for item in support_meta}
    # 返回选中的样本
    selected_samples = []
    for s_id in selected_ids:
        if s_id in support_dict:
            selected_samples.append(support_dict[s_id])
            if len(selected_samples) >= n_shot:
                break
    
    return selected_samples


def select_with_diversity(candidate_ids, support_dict, features, 
                        query_feature, n_shot, strategy):
    """查询感知的多样性选择"""
    valid_features = []
    valid_ids = []
    for s_id in candidate_ids:
        if s_id in features:
            feat = features[s_id].astype(np.float32)
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
    elif dataset == 'EmotionROI':
        if description == 'concise':
            instr = 'Select the emotional category to which the image belongs from the options below: ["anger","disgust","fear","joy","sadness","surprise"]. Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    elif dataset == 'ArtPhoto':
        if description == 'concise':
            instr = 'Select the emotional category to which the image belongs from the options below: ["awe","contentment","sad","amusement","excitement","fear","anger","disgust"]. Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    elif dataset == 'EmoSet':
        if description == 'concise':
            instr = 'Select the emotional category to which the image belongs from the options below: ["amusement","anger","awe","contentment","disgust","excitement","fear","sadness"]. Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    elif dataset == 'StanfordCars':
        if description == 'concise':
            instr = 'Select the category to which the object in the image belongs from the options below: ["AM General Hummer SUV 2000","Acura RL Sedan 2012","Acura TL Sedan 2012","Acura TL Type-S 2008","Acura TSX Sedan 2012","Acura Integra Type R 2001","Acura ZDX Hatchback 2012","Aston Martin V8 Vantage Convertible 2012","Aston Martin V8 Vantage Coupe 2012","Aston Martin Virage Convertible 2012","Aston Martin Virage Coupe 2012","Audi RS 4 Convertible 2008","Audi A5 Coupe 2012","Audi TTS Coupe 2012","Audi R8 Coupe 2012","Audi V8 Sedan 1994","Audi 100 Sedan 1994","Audi 100 Wagon 1994","Audi TT Hatchback 2011","Audi S6 Sedan 2011","Audi S5 Convertible 2012","Audi S5 Coupe 2012","Audi S4 Sedan 2012","Audi S4 Sedan 2007","Audi TT RS Coupe 2012","BMW ActiveHybrid 5 Sedan 2012","BMW 1 Series Convertible 2012","BMW 1 Series Coupe 2012","BMW 3 Series Sedan 2012","BMW 3 Series Wagon 2012","BMW 6 Series Convertible 2007","BMW X5 SUV 2007","BMW X6 SUV 2012","BMW M3 Coupe 2012","BMW M5 Sedan 2010","BMW M6 Convertible 2010","BMW X3 SUV 2012","BMW Z4 Convertible 2012","Bentley Continental Supersports Conv. Convertible 2012","Bentley Arnage Sedan 2009","Bentley Mulsanne Sedan 2011","Bentley Continental GT Coupe 2012","Bentley Continental GT Coupe 2007","Bentley Continental Flying Spur Sedan 2007","Bugatti Veyron 16.4 Convertible 2009","Bugatti Veyron 16.4 Coupe 2009","Buick Regal GS 2012","Buick Rainier SUV 2007","Buick Verano Sedan 2012","Buick Enclave SUV 2012","Cadillac CTS-V Sedan 2012","Cadillac SRX SUV 2012","Cadillac Escalade EXT Crew Cab 2007","Chevrolet Silverado 1500 Hybrid Crew Cab 2012","Chevrolet Corvette Convertible 2012","Chevrolet Corvette ZR1 2012","Chevrolet Corvette Ron Fellows Edition Z06 2007","Chevrolet Traverse SUV 2012","Chevrolet Camaro Convertible 2012","Chevrolet HHR SS 2010","Chevrolet Impala Sedan 2007","Chevrolet Tahoe Hybrid SUV 2012","Chevrolet Sonic Sedan 2012","Chevrolet Express Cargo Van 2007","Chevrolet Avalanche Crew Cab 2012","Chevrolet Cobalt SS 2010","Chevrolet Malibu Hybrid Sedan 2010","Chevrolet TrailBlazer SS 2009","Chevrolet Silverado 2500HD Regular Cab 2012","Chevrolet Silverado 1500 Classic Extended Cab 2007","Chevrolet Express Van 2007","Chevrolet Monte Carlo Coupe 2007","Chevrolet Malibu Sedan 2007","Chevrolet Silverado 1500 Extended Cab 2012","Chevrolet Silverado 1500 Regular Cab 2012","Chrysler Aspen SUV 2009","Chrysler Sebring Convertible 2010","Chrysler Town and Country Minivan 2012","Chrysler 300 SRT-8 2010","Chrysler Crossfire Convertible 2008","Chrysler PT Cruiser Convertible 2008","Daewoo Nubira Wagon 2002","Dodge Caliber Wagon 2012","Dodge Caliber Wagon 2007","Dodge Caravan Minivan 1997","Dodge Ram Pickup 3500 Crew Cab 2010","Dodge Ram Pickup 3500 Quad Cab 2009","Dodge Sprinter Cargo Van 2009","Dodge Journey SUV 2012","Dodge Dakota Crew Cab 2010","Dodge Dakota Club Cab 2007","Dodge Magnum Wagon 2008","Dodge Challenger SRT8 2011","Dodge Durango SUV 2012","Dodge Durango SUV 2007","Dodge Charger Sedan 2012","Dodge Charger SRT-8 2009","Eagle Talon Hatchback 1998","FIAT 500 Abarth 2012","FIAT 500 Convertible 2012","Ferrari FF Coupe 2012","Ferrari California Convertible 2012","Ferrari 458 Italia Convertible 2012","Ferrari 458 Italia Coupe 2012","Fisker Karma Sedan 2012","Ford F-450 Super Duty Crew Cab 2012","Ford Mustang Convertible 2007","Ford Freestar Minivan 2007","Ford Expedition EL SUV 2009","Ford Edge SUV 2012","Ford Ranger SuperCab 2011","Ford GT Coupe 2006","Ford F-150 Regular Cab 2012","Ford F-150 Regular Cab 2007","Ford Focus Sedan 2007","Ford E-Series Wagon Van 2012","Ford Fiesta Sedan 2012","GMC Terrain SUV 2012","GMC Savana Van 2012","GMC Yukon Hybrid SUV 2012","GMC Acadia SUV 2012","GMC Canyon Extended Cab 2012","Geo Metro Convertible 1993","HUMMER H3T Crew Cab 2010","HUMMER H2 SUT Crew Cab 2009","Honda Odyssey Minivan 2012","Honda Odyssey Minivan 2007","Honda Accord Coupe 2012","Honda Accord Sedan 2012","Hyundai Veloster Hatchback 2012","Hyundai Santa Fe SUV 2012","Hyundai Tucson SUV 2012","Hyundai Veracruz SUV 2012","Hyundai Sonata Hybrid Sedan 2012","Hyundai Elantra Sedan 2007","Hyundai Accent Sedan 2012","Hyundai Genesis Sedan 2012","Hyundai Sonata Sedan 2012","Hyundai Elantra Touring Hatchback 2012","Hyundai Azera Sedan 2012","Infiniti G Coupe IPL 2012","Infiniti QX56 SUV 2011","Isuzu Ascender SUV 2008","Jaguar XK XKR 2012","Jeep Patriot SUV 2012","Jeep Wrangler SUV 2012","Jeep Liberty SUV 2012","Jeep Grand Cherokee SUV 2012","Jeep Compass SUV 2012","Lamborghini Reventon Coupe 2008","Lamborghini Aventador Coupe 2012","Lamborghini Gallardo LP 570-4 Superleggera 2012","Lamborghini Diablo Coupe 2001","Land Rover Range Rover SUV 2012","Land Rover LR2 SUV 2012","Lincoln Town Car Sedan 2011","MINI Cooper Roadster Convertible 2012","Maybach Landaulet Convertible 2012","Mazda Tribute SUV 2011","McLaren MP4-12C Coupe 2012","Mercedes-Benz 300-Class Convertible 1993","Mercedes-Benz C-Class Sedan 2012","Mercedes-Benz SL-Class Coupe 2009","Mercedes-Benz E-Class Sedan 2012","Mercedes-Benz S-Class Sedan 2012","Mercedes-Benz Sprinter Van 2012","Mitsubishi Lancer Sedan 2012","Nissan Leaf Hatchback 2012","Nissan NV Passenger Van 2012","Nissan Juke Hatchback 2012","Nissan 240SX Coupe 1998","Plymouth Neon Coupe 1999","Porsche Panamera Sedan 2012","Ram C/V Cargo Van Minivan 2012","Rolls-Royce Phantom Drophead Coupe Convertible 2012","Rolls-Royce Ghost Sedan 2012","Rolls-Royce Phantom Sedan 2012","Scion xD Hatchback 2012","Spyker C8 Convertible 2009","Spyker C8 Coupe 2009","Suzuki Aerio Sedan 2007","Suzuki Kizashi Sedan 2012","Suzuki SX4 Hatchback 2012","Suzuki SX4 Sedan 2012","Tesla Model S Sedan 2012","Toyota Sequoia SUV 2012","Toyota Camry Sedan 2012","Toyota Corolla Sedan 2012","Toyota 4Runner SUV 2012","Volkswagen Golf Hatchback 2012","Volkswagen Golf Hatchback 1991","Volkswagen Beetle Hatchback 2012","Volvo C30 Hatchback 2012","Volvo 240 Sedan 1993","Volvo XC90 SUV 2007","smart fortwo Convertible 2012"]. Your response must strictly follow this format: "The category of object in the image is [category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    elif dataset == 'OxfordFlowers17':
        if description == 'concise':
            instr = 'Select the category to which the object in the image belongs from the options below: ["Bluebell","Buttercup","Coltsfoot","Cowslip","Crocus","Daffodil","Daisy","Dandelion","Fritillary","Iris","Lilyvalley","Pansy","Snowdrop","Sunflower","Tigerlily","Tulip","Windflower"]. Your response must strictly follow this format: "The category of object in the image is [category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    elif dataset == 'Oxford-IIIT_Pet':
        if description == 'concise':
            instr = 'Select the category to which the object in the image belongs from the options below: ["Abyssinian","Bengal","Birman","Bombay","British_Shorthair","Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue","Siamese","Sphynx","american_bulldog","american_pit_bull_terrier","basset_hound","beagle","boxer","chihuahua","english_cocker_spaniel","english_setter","german_shorthaired","great_pyrenees","havanese","japanese_chin","keeshond","leonberger","miniature_pinscher","newfoundland","pomeranian","pug","saint_bernard","samoyed","scottish_terrier","shiba_inu","staffordshire_bull_terrier","wheaten_terrier","yorkshire_terrier"]. Your response must strictly follow this format: "The category of object in the image is [category]".'
        elif description == 'detailed':
            instr = 'When you see this picture, which of the following categories do you think this picture belongs to?'
    return instr

def format_answer(answer, dataset, query=None):
    if dataset in ['Intentonomy','EmotionROI','ArtPhoto','EmoSet','StanfordCars','StanfordDogs','CUB_200_2011','OxfordFlowers17','Oxford-IIIT_Pet']:
        answer = str(answer)
    return answer


