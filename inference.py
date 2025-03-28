import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from tqdm import tqdm
import json
import argparse
import gc
import sys
from utils import model_inference, utils, ICL_utils, load_models
import datetime

sys.path.insert(0, "/data1/yq_log/IntenICL/LLaVA-NeXT")
sys.path.insert(0, "/data1/yq_log/IntenICL/Qwen-VL")



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
    parser.add_argument('--balance_threshold', default=0.5, type=int, help='Random seed.')
    return parser.parse_args()

def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, strategy, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens
    with open(f'{args.dataDir}/{args.dataset}/{args.sim_path}', 'rb') as f:
        similarity_data = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/{args.support_path}', 'rb') as f:
        support_features = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/{args.query_path}', 'rb') as f:
        query_features = pickle.load(f)


    for query in tqdm(query_meta):

        query_feature = query_features[query['img_id']].numpy()
        
        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, strategy, query, similarity_data, support_features, query_feature, args.balance_threshold)

        predicted_answer = model_inference.ICL_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                      n_shot_support, data_path, processor, max_new_tokens) 
        
        query['support'] = []
        for support_sample in n_shot_support:
            query['support'].append({
                "image": support_sample["image"],   # 假设字段名为 "image"
                "answer": support_sample["answer"]  # 假设字段名为 "answer"
            })

        query['prediction'] = predicted_answer
        results.append(query)

    return results
    

if __name__ == "__main__":
    args = parse_args()

    query_meta, support_meta = utils.load_data(args)
    
    for engine in args.engine:
        for strategy in args.strategy:

            model, tokenizer, processor = load_models.load_i2t_model(engine, args)
            print("Loaded model: {}\n".format(engine))
            
            utils.set_random_seed(args.seed)
            current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for shot in args.n_shot:
                results_dict = eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, strategy, int(shot))
                os.makedirs(f"{args.resultDir}/{args.dataset}_{strategy}", exist_ok=True)
                with open(f"{args.resultDir}/{args.dataset}_{strategy}/{engine}_balance{args.balance_threshold}_{shot}-shot.json", "w") as f:
                    json.dump(results_dict, f, indent=4)

            del model, tokenizer, processor
            torch.cuda.empty_cache()
            gc.collect()