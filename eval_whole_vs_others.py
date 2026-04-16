import os
import json

# 固定参数
resultDir = '/data1/yq_log/IntenICL/ICL_log'
dataset = 'EmoSet'
engine = 'idefics-9b-instruct'
shot = '4'
balance_threshold = 0.5
seed = 0
strategies = ['whole', 'similarity', 'qtmt', 'mmices', 'muier']

# 结果文件路径

result_files = {}
for strategy in strategies:
    file1 = f"{resultDir}/{dataset}_{strategy}/{engine}_{shot}-shot-balance{balance_threshold}-seed{seed}.json"
    file2 = f"{resultDir}/{dataset}_{strategy}/{engine}_{shot}-shot-balance{balance_threshold}.json"
    if os.path.exists(file1):
        result_files[strategy] = file1
    elif os.path.exists(file2):
        result_files[strategy] = file2
    else:
        result_files[strategy] = None

# 加载所有策略结果
results = {}
for strategy, file_path in result_files.items():
    if not os.path.exists(file_path):
        print(f"Warning: Result file not found: {file_path}")
        continue
    with open(file_path, "r") as f:
        results[strategy] = json.load(f)

# 以查询id为索引，收集每个查询的预测结果
query_pred = {}
for strategy, res_list in results.items():
    for item in res_list:
        qid = item.get('img_id')
        pred = item.get('prediction')
        answer = item.get('answer') if 'answer' in item else item.get('gt')
        if qid not in query_pred:
            query_pred[qid] = {}
        query_pred[qid][strategy] = {'pred': pred, 'answer': answer}

# 评估正确性（假设eval.eval_scores支持单样本评估，否则直接用==判断）
def is_correct(pred, answer):
    return str(pred).strip() == str(answer).strip()

# 查找whole正确且其他策略均错误的查询样本
success_qids = []
for qid, preds in query_pred.items():
    if 'whole' not in preds:
        continue
    whole_ok = is_correct(preds['whole']['pred'], preds['whole']['answer'])
    others_fail = all(
        (s == 'whole' or s not in preds or not is_correct(preds[s]['pred'], preds[s]['answer']))
        for s in strategies
    )
    if whole_ok and others_fail:
        success_qids.append(qid)

print(f"whole策略下预测成功，其他策略均失败的查询样本数: {len(success_qids)}")
print("查询样本ID如下：")
for qid in success_qids:
    print(qid)
