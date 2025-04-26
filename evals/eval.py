import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SIM_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def eval_scores(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset in ['Intentonomy','EmotionROI','ArtPhoto','EmoSet','StanfordCars','StanfordDogs','CUB_200_2011','OxfordFlowers17','Oxford-IIIT_Pet']:
        score = semantic_accuracy(dataset, results, model)
    return score


def semantic_accuracy(dataset, results, model, thresholds=[0.7, 0.8, 0.9]):
    """支持句号分句的语义评估函数"""
    if dataset == 'Intentonomy':
        prefix = "the intention of this image is "
    elif dataset in ['EmotionROI', 'ArtPhoto', 'EmoSet']:
        prefix = "the emotion of this image is"
    elif dataset in ['StanfordCars','OxfordFlowers17','Oxford-IIIT_Pet']:
        prefix = "the category of object in the image is"
    elif dataset == 'StanfordDogs':
        prefix = "the category of object in the image is"  
    elif dataset == 'CUB_200_2011':
        prefix = "the category of object in the image is"
    else: prefix = ''
    acc = {th: [] for th in thresholds}
    
    for result in results:
        # 数据预处理
        raw_pred = result['prediction'].strip().lower()
        raw_answer = str(result['answer']).strip().lower()
        
        # 针对idefics-9b-instruct模型的特殊处理
        if model == 'idefics-9b-instruct':
            pred_final = ""
            
            # 方法1：提取第一个换行符前的内容
            first_line = raw_pred.split('\n')[0].strip()
            if first_line.startswith(prefix):
                pred_final = first_line[len(prefix):].strip()
            
            # 方法2：如果第一行不符合要求，寻找第一个符合prefix的句子
            if not pred_final:
                # 保留原始换行符以防万一需要按行检查
                lines = raw_pred.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(prefix):
                        pred_final = line[len(prefix):].strip()
                        break
            
            # 如果找到有效预测
            if pred_final:
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
                continue
        
        # 如果不是特殊模型或特殊处理失败，进入标准流程
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


