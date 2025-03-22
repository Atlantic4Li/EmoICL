import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# 全局加载语义相似度模型（推荐中文优化模型）
SIM_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def eval_scores(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset == 'Intentonomy':
        score = semantic_accuracy(results)
    return score


# def semantic_accuracy(results, threshold=0.8):
#     """
#     基于语义相似度的准确率计算
#     :param results: 数据列表，每个元素包含 'prediction' 和 'answer'
#     :param threshold: 相似度阈值，默认0.8（需根据数据分布调整）
#     :return: 准确率（0~1）
#     """
#     acc = []
#     for result in results:
#         # 预处理：保留原有截断逻辑（按换行符或句号截断）
#         prediction = result['prediction'].strip().strip('\n')
#         trunc_index = prediction.find('\n') or prediction.find('.')
#         if trunc_index > 0:
#             prediction = prediction[:trunc_index]
        
#         # 计算相似度
#         answer = str(result['answer']).lower()
#         prediction_processed = str(prediction).lower()
        
#         # 编码文本并计算余弦相似度
#         embeddings = SIM_MODEL.encode([prediction_processed, answer])
#         similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
#         acc.append(1 if similarity >= threshold else 0)
    
#     return np.mean(acc)

def semantic_accuracy(results, thresholds=[0.7, 0.8, 0.9]):
    """支持句号分句的语义评估函数"""
    prefix = "the intention of this image is "
    acc = {th: [] for th in thresholds}
    
    for result in results:
        # 数据预处理
        raw_pred = result['prediction'].strip().lower()
        raw_answer = str(result['answer']).strip().lower()
        
        raw_pred = raw_pred.replace('\n', ' ')
        
        # 分句处理逻辑 ---------------------------------------------------
        valid_contents = []
        
        # 按句号分句并过滤空内容
        sentences = [s.strip() for s in raw_pred.split('.') if s.strip()]
        
        for sent in sentences:
            # 处理可能存在的嵌套分句（如末尾带标点）
            clean_sent = sent.split('!')[0].split('?')[0].strip()
            if clean_sent.startswith(prefix):
                content = clean_sent[len(prefix):].strip()
                if content:  # 防止空内容
                    valid_contents.append(content)
        
        # 唯一性验证
        unique_contents = list(set(valid_contents))
        if len(unique_contents) > 1:
            for th in thresholds:
                acc[th].append(0)
            continue
            
        # 内容提取策略
        pred_final = unique_contents[0] if valid_contents else ""
        
        # 兼容无前缀但可能包含意图的情况
        if not valid_contents:
            if raw_pred.startswith(prefix):
                pred_final = raw_pred[len(prefix):].strip()
            else:
                pred_final = raw_pred
        
        # 答案处理
        answer_final = raw_answer[len(prefix):].strip() \
                      if raw_answer.startswith(prefix) \
                      else raw_answer
        
        # 语义计算
        embeddings = SIM_MODEL.encode([pred_final, answer_final])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # 记录结果
        for th in thresholds:
            acc[th].append(1 if similarity >= th else 0)
    
    return {th: np.mean(acc[th]) for th in thresholds}


