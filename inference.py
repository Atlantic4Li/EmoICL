import os
import pickle
<<<<<<< HEAD
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
=======
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
>>>>>>> bc7a5d0 (baselines)

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
    parser.add_argument('--dataset', default='EmotionROI', type=str, choices=['Intentonomy','EmotionROI','ArtPhoto','EmoSet','StanfordCars','OxfordFlowers17','StanfordDogs','CUB_200_2011','Oxford-IIIT_Pet'])
    parser.add_argument('--query_image_features', default='query_image_features.pkl', type=str, help='features of query images')
    parser.add_argument('--query_text_features', default='query_text_features.pkl', type=str, help='features of query texts')
    parser.add_argument('--support_image_features', default='support_image_features.pkl', type=str, help='features of support images')
    parser.add_argument('--support_text_features', default='support_text_features.pkl', type=str, help='features of support texts')
    parser.add_argument('--text_text_similarity', default='text_text_similarity.pkl', type=str, help='similarity between texts')
    parser.add_argument('--clip_text_image_similarity', default='clip_text_image_similarity.pkl', type=str, help='cross-modal similarity between text and image')
    parser.add_argument('--image_image_similarity', default='image_image_similarity.pkl', type=str, help='similarity between images')
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "qwen-vl", 'internlm-x2', 
                                                   'idefics-9b-instruct', 
<<<<<<< HEAD
                                                   'llava-onevision-7b','llava-onevision-0.5b','llava16-7b','llava16-13b'], default=['qwen-vl'], nargs="+")
    parser.add_argument('--strategy', default=['random'], nargs='+', type=str, choices=['random', 'similarity', 'various','test_same_similarity','test_same_random','test_text_similarity','test_diverse','whole'], help='Example selection strategy.')
    parser.add_argument('--n_shot', default=[1,2,4,8], nargs='+', type=int, help='Number of support images.')
=======
                                                   'llava-onevision-7b','llava-onevision-0.5b','llava16-7b','llava16-13b'], default=['otter-llama'], nargs="+")
    parser.add_argument('--strategy', default=['whole'], nargs='+', type=str, choices=['random', 'similarity', 'various','test_same_similarity','test_same_random','test_text_similarity','test_diverse','whole','qtmt','mmices','muier','circles','cdsp'], help='Example selection strategy.')
    parser.add_argument('--n_shot', default=[1,2,4], nargs='+', type=int, help='Number of support images.')
>>>>>>> bc7a5d0 (baselines)

    parser.add_argument('--max-new-tokens', default=32, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='concise', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--balance_threshold', default=0.5, type=float, help='Balance threshold for various strategy.')
    parser.add_argument('--exp_scale', default=10000, type=int, help='emoset support scale')
    parser.add_argument('--score_weight', default=0.7, type=float, help='emoset support scale')
    parser.add_argument('--alpha_weight', default=0.8, type=float, help='emoset support scale')
    # CIRCLES-specific arguments
    parser.add_argument('--num_attributes', default=1, type=int, help='Number of attributes for CIRCLES.')
    parser.add_argument('--attribute_k', default=None, type=int, help='Number of composed retrievals per attribute (defaults to n_shot).')
    parser.add_argument('--clip_model', default='openai/clip-vit-base-patch32', type=str, help='CLIP model name for CIRCLES composed retrieval.')
    return parser.parse_args()

def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, strategy, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens
    
    # 预加载query_image_features以提高效率
    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_image_features}', 'rb') as f:
        query_features = pickle.load(f)

    for query in tqdm(query_meta):
        # Build kwargs for CIRCLES strategy (model/tokenizer/processor needed for VLM calls)
        circles_kwargs = {}
        if strategy == 'circles':
            circles_kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                engine=engine,
                data_path=data_path,
                clip_model_name=args.clip_model,
                num_attributes=args.num_attributes,
                attribute_k=args.attribute_k if args.attribute_k is not None else n_shot,
            )
            args.attribute_k_current = circles_kwargs['attribute_k']
            args.n_shot_current = n_shot

        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, strategy, query, args, query_features, **circles_kwargs)

        predicted_answer = model_inference.ICL_inference(args, engine, args.dataset, model, tokenizer, query,
                                                      n_shot_support, data_path, processor, max_new_tokens)

        query['support'] = []
        if isinstance(n_shot_support, dict) and 'original_retrievals' in n_shot_support:
            # CIRCLES structured result
            for support_sample in n_shot_support.get('original_retrievals', []):
                query['support'].append({
                    "image": support_sample["image"],
                    "answer": support_sample["answer"],
                    "retrieval_type": "original",
                })
            for cir in n_shot_support.get('composed_retrievals', []):
                for support_sample in cir.get('retrieved_items', []):
                    query['support'].append({
                        "image": support_sample["image"],
                        "answer": support_sample["answer"],
                        "retrieval_type": "composed",
                        "attribute": cir.get("attribute", ""),
                    })
        else:
            for support_sample in n_shot_support:
                query['support'].append({
                    "image": support_sample["image"],
                    "answer": support_sample["answer"],
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

                with open(f"{args.resultDir}/{args.dataset}_{strategy}/{engine}_{shot}-shot-balance{args.balance_threshold}-seed{args.seed}-score_weight{args.score_weight}-alpha_weight{args.alpha_weight}.json", "w") as f:
                    json.dump(results_dict, f, indent=4)

            del model, tokenizer, processor
            torch.cuda.empty_cache()
            gc.collect()