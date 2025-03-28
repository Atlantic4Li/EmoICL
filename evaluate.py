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
    parser.add_argument('--dataset', default='Oxford-IIIT_Pet', type=str, choices=['Intentonomy','EmotionROI','ArtPhoto','EmoSet','StanfordCars','OxfordFlowers17','StanfordDogs','CUB_200_2011','Oxford-IIIT_Pet'])
    parser.add_argument('--sim_path', default='cross_similarity.pkl', type=str, help='the similarity of all images')
    parser.add_argument('--support_path', default='support_features.pkl', type=str, help='the features of support images')
    parser.add_argument('--query_path', default='query_features.pkl', type=str, help='the features of query images')
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-80b-instruct', 'gpt4v', 'llava-onevision-7b',
                                                   'llava-onevision-0.5b'],
                        default=['llava-onevision-0.5b'], nargs="+")
    parser.add_argument('--strategy', default=['various'], type=str, choices=['random', 'similarity', 'various'], help='Example selection strategy.')
    parser.add_argument('--n_shot', default=[2,4,8], nargs="+", help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=256, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='concise', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for strategy in args.strategy:

        result_files = [f"{args.resultDir}/{args.dataset}_{strategy}/{engine}_balance0.5_{shot}-shot.json" for engine in args.engine for shot in args.n_shot]
        
        for result_file in result_files:
            engine, balance, shot = result_file.split("/")[-1].replace(".json", "").split("_")
            with open(result_file, "r") as f:
                results_dict = json.load(f)
            
            score = eval.eval_scores(results_dict, args.dataset)
            # print(f'{args.dataset} {metrics[args.dataset]} of {engine} {shot}: ', f"{score * 100.0:.2f}", flush=True)

            formatted_scores = ', '.join([f"{thresh}→{acc*100:.2f}%" for thresh, acc in sorted(score.items())])

            print(f'{args.dataset}_{strategy} {metrics[args.dataset]} of {engine} {balance} {shot}:', formatted_scores, flush=True)