import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import json
import argparse
from evals import eval
import transformers


metrics = {
    "EmotionROI": "Acc",
    "ArtPhoto": "Acc",
    "EmoSet": "Acc",
    'StanfordCars': "Acc",
    'StanfordDogs': "Acc",
    'Oxford-IIIT_Pet': "Acc",
    'Intentonomy': "Acc",
    'OxfordFlowers17': "Acc",
}

def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Inference')

    parser.add_argument('--dataDir', default='/data1/yq_log/IntenICL/DataSet', type=str, help='Data directory.')
    parser.add_argument('--resultDir', default='/data1/yq_log/IntenICL/ICL_log', type=str, help='result directory.')
    parser.add_argument('--dataset', default=['EmotionROI'], type=str, choices=['Intentonomy','EmotionROI','ArtPhoto','EmoSet','StanfordCars','OxfordFlowers17','StanfordDogs','CUB_200_2011','Oxford-IIIT_Pet'])
    parser.add_argument('--sim_path', default='cross_similarity.pkl', type=str, help='the similarity of all images')
    parser.add_argument('--support_path', default='support_features.pkl', type=str, help='the features of support images')
    parser.add_argument('--query_path', default='query_features.pkl', type=str, help='the features of query images')
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-80b-instruct', 'gpt4v', 'llava-onevision-7b',
                                                   'llava-onevision-0.5b'],
                        default=['llava-onevision-7b',"qwen-vl"], nargs="+")
    parser.add_argument('--strategy', default=['various'], type=str, choices=['random', 'similarity', 'various','test_same_similarity','test_same_random','test_text_similarity','whole','test_diverse','circles','cdsp'], help='Example selection strategy.')
    parser.add_argument('--n_shot', default=[1,2,4,8], nargs="+", help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=256, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='concise', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')
    parser.add_argument('--ft', default=False, type=bool, help='Whether to use fine-tuning.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--balance_threshold', default=0.5, type=int, help='Random seed.')
    parser.add_argument('--exp_scale', default=10000, type=int, help='emoset support scale')
    parser.add_argument('--score_weight', default=0.7, type=float, help='the weight of similarity score and diversity score')
    parser.add_argument('--alpha_weight', default=0.8, type=float, help='the weight of alpha score')  
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 遍历所有数据集
    for dataset in args.dataset:
        print(f"\n=== Evaluating Dataset: {dataset} ===")
        
        # 遍历所有策略
        for strategy in args.strategy:
            print(f"\n--- Strategy: {strategy} ---")
            
            # 构建结果文件路径
            if args.ft:
                result_files = [f"{args.resultDir}/{dataset}_{strategy}/{engine}_{shot}-shot-ft2-balance{args.balance_threshold}.json" 
                              for engine in args.engine 
                              for shot in args.n_shot]
            else:

                result_files = [f"{args.resultDir}/{dataset}_{strategy}/{engine}_{shot}-shot-balance{args.balance_threshold}-seed{args.seed}-score_weight{args.score_weight}-alpha_weight{args.alpha_weight}.json"
                              for engine in args.engine
                              for shot in args.n_shot]
            
            # 评估每个结果文件
            for result_file in result_files:
                try:
                    # engine, shot, suffix = result_file.split("/")[-1].replace(".json", "").split("_")
                    engine, shot, score_weight, alpha_weight = result_file.split("/")[-1].replace(".json", "").split("_")
                    
                    # 检查文件是否存在
                    if not os.path.exists(result_file):
                        print(f"Warning: Result file not found: {result_file}")
                        continue
                        
                    with open(result_file, "r") as f:
                        results_dict = json.load(f)
                    
                    score = eval.eval_scores(results_dict, dataset, engine)
                    formatted_scores = ', '.join([f"{thresh}→{acc*100:.2f}%" for thresh, acc in sorted(score.items())])
                    # print(f'{dataset}_{strategy} {metrics[dataset]} of {engine} {shot}_{suffix}: {formatted_scores}', flush=True)
                    print(f'{dataset}_{strategy} {metrics[dataset]} of {engine} {shot} {score_weight} {alpha_weight}: {formatted_scores}', flush=True)
                    
                except Exception as e:
                    print(f"Error processing {result_file}: {str(e)}")
                    continue