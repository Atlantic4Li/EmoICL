import pickle
import os
import numpy as np
from collections import defaultdict
import math


def select_demonstration(support_meta, n_shot, dataset, query, args, query_features=None):
    """Whole-strategy ICL: hybrid visual + cross-modal rerank + MMR diversity."""
    data_path = args.dataDir
    with open(f'{data_path}/{dataset}/query/{args.query_image_features}', 'rb') as f:
        query_image_features = pickle.load(f)
    with open(f'{data_path}/{dataset}/support/{args.support_image_features}', 'rb') as f:
        support_image_features = pickle.load(f)
    with open(f'{data_path}/{dataset}/similarity/{args.clip_text_image_similarity}', 'rb') as f:
        clip_text_image_similarity = pickle.load(f)
    with open(f'{data_path}/{dataset}/similarity/{args.image_image_similarity}', 'rb') as f:
        image_image_similarity = pickle.load(f)

    if query_features is None:
        query_features = query_image_features
    if query and query['img_id'] in query_features:
        query_feature = query_features[query['img_id']].numpy()
    else:
        query_feature = None

    balance_threshold = args.balance_threshold
    control_score_weight = args.score_weight
    control_lambda_param = args.alpha_weight

    return retrieve_hybrid_demos(
        query, support_meta, n_shot, image_image_similarity, clip_text_image_similarity,
        support_image_features, query_feature, balance_threshold,
        score_weight=control_score_weight, lambda_param=control_lambda_param,
    )


def allocate_quota(category_counts, total, n_shot):
    """Quota allocation: largest class first + round-robin remainder."""
    valid_classes = [k for k, v in category_counts.items() if v > 0]
    num_classes = len(valid_classes)

    if n_shot <= num_classes:
        sorted_classes = sorted(valid_classes, key=lambda x: -category_counts[x])
        return {cls: 1 for cls in sorted_classes[:n_shot]}

    base_alloc = {cls: 1 for cls in valid_classes}
    remaining = n_shot - num_classes
    sorted_classes = sorted(valid_classes, key=lambda x: -category_counts[x])
    proportions = {k: v / total for k, v in category_counts.items()}

    if sorted_classes:
        max_cls = sorted_classes[0]
        max_alloc = math.ceil(proportions[max_cls] * remaining)
        allocated = min(max_alloc, remaining)
        base_alloc[max_cls] += allocated
        remaining -= allocated

    for cls in sorted_classes[1:]:
        if remaining <= 0:
            break
        alloc = round(proportions[cls] * remaining)
        alloc = min(max(alloc, 0), remaining)
        base_alloc[cls] += alloc
        remaining -= alloc

    if remaining > 0:
        index = 0
        while remaining > 0:
            current_cls = sorted_classes[index % len(sorted_classes)]
            base_alloc[current_cls] += 1
            remaining -= 1
            index += 1

    return {k: v for k, v in base_alloc.items() if v > 0}


def retrieve_hybrid_demos(
    query, support_meta, n_shot, visual_similarity, cross_modal_similarity,
    support_features, query_feature, balance_threshold, min_class_ratio=0.2,
    reverse_results=False, score_weight=0.7, lambda_param=0.6,
):
    support_dict = {item['img_id']: item for item in support_meta}

    if n_shot == 1:
        query_id = query['img_id']
        visual_sims = visual_similarity[query_id]
        sorted_visual = sorted(visual_sims.items(), key=lambda x: x[1], reverse=True)
        top_k = min(5, len(sorted_visual))
        pre_filtered = {
            s_id: sim_score
            for s_id, sim_score in sorted_visual[:top_k]
            if s_id in support_dict
        }
        candidate_scores = {}
        for cand_id, visual_score in pre_filtered.items():
            if cand_id in cross_modal_similarity[query_id]:
                cross_score = cross_modal_similarity[query_id][cand_id]
                combined_score = score_weight * visual_score + (1 - score_weight) * cross_score
                candidate_scores[cand_id] = combined_score

        category_counter = defaultdict(list)
        for s_id, score in candidate_scores.items():
            category = support_dict[s_id]['category']
            category_counter[category].append((s_id, score))

        category_counts = {k: len(v) for k, v in category_counter.items()}
        total = sum(category_counts.values())

        if total == 0:
            return []

        valid_categories = {
            k: v for k, v in category_counts.items()
            if v / total >= min_class_ratio
        } if total > 0 else {}

        if not valid_categories:
            valid_categories = category_counts

        total_valid = sum(valid_categories.values())
        max_proportion = max(v / total_valid for v in valid_categories.values()) if total_valid > 0 else 0

        if max_proportion > balance_threshold:
            main_class = max(valid_categories.items(), key=lambda x: x[1])[0]
            main_class_samples = category_counter[main_class]
            best_sample_id = max(main_class_samples, key=lambda x: x[1])[0]
            return [support_dict[best_sample_id]]
        best_sample_id = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return [support_dict[best_sample_id]]

    query_id = query['img_id']
    first_stage_size = min(5 * n_shot, len(support_dict))
    visual_sims = visual_similarity[query_id]
    sorted_visual = sorted(visual_sims.items(), key=lambda x: x[1], reverse=True)

    first_stage_candidates = {
        support_id: sim_score
        for support_id, sim_score in sorted_visual[:first_stage_size]
        if support_id in support_dict
    }

    second_stage_size = 2 * n_shot
    cross_sims = cross_modal_similarity[query_id]

    candidate_scores = {}
    for cand_id in first_stage_candidates:
        if cand_id in cross_sims:
            visual_score = first_stage_candidates[cand_id]
            cross_score = cross_sims[cand_id]
            combined_score = score_weight * visual_score + (1 - score_weight) * cross_score
            candidate_scores[cand_id] = combined_score

    category_groups = defaultdict(list)
    for support_id, score in candidate_scores.items():
        category = support_dict[support_id]['category']
        category_groups[category].append((support_id, score))

    category_counts = {k: len(v) for k, v in category_groups.items()}
    total_samples = sum(category_counts.values())

    if total_samples == 0:
        return []

    filtered_categories = {
        k: v for k, v in category_counts.items()
        if v / total_samples >= min_class_ratio
    }

    if not filtered_categories:
        filtered_categories = category_counts

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

    feature_lookup = {k: v.numpy() for k, v in support_features.items()}
    final_selected = select_with_diversity(
        candidate_ids=list(second_stage_candidates.keys()),
        support_dict=support_dict,
        features=feature_lookup,
        query_feature=query_feature,
        n_shot=n_shot,
        lambda_param=lambda_param,
    )

    return final_selected[::-1] if reverse_results else final_selected


def select_with_diversity(candidate_ids, support_dict, features, query_feature, n_shot, lambda_param=0.6):
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
    indices = mmr_selection(feature_matrix, query_feature, n_shot, lambda_param=lambda_param)
    return [support_dict[valid_ids[i]] for i in indices if valid_ids[i] in support_dict]


def mmr_selection(features, query_feature, n_shot, lambda_param=0.6):
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


def get_task_instruction(args):
    dataset = args.dataset
    description = args.task_description
    if description == 'nothing':
        return ''

    if dataset == 'EmotionROI':
        if description == 'concise':
            return (
                'Select the emotional category to which the image belongs from the options below: '
                '["anger","disgust","fear","joy","sadness","surprise"]. '
                'Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
            )
        return 'When you see this picture, which of the following categories do you think this picture belongs to?'

    if dataset == 'ArtPhoto':
        if description == 'concise':
            return (
                'Select the emotional category to which the image belongs from the options below: '
                '["awe","contentment","sad","amusement","excitement","fear","anger","disgust"]. '
                'Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
            )
        return 'When you see this picture, which of the following categories do you think this picture belongs to?'

    if dataset == 'EmoSet':
        if description == 'concise':
            return (
                'Select the emotional category to which the image belongs from the options below: '
                '["amusement","anger","awe","contentment","disgust","excitement","fear","sadness"]. '
                'Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
            )
        return 'When you see this picture, which of the following categories do you think this picture belongs to?'

    return ''


def format_answer(answer, dataset, query=None):
    return str(answer)
