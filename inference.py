import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

STRATEGY = "whole"


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Inference (whole strategy only)')

    parser.add_argument('--dataDir', default='/data1/yq_log/IntenICL/DataSet', type=str, help='Data directory.')
    parser.add_argument('--resultDir', default='/data1/yq_log/IntenICL/ICL_log', type=str, help='result directory.')
    parser.add_argument(
        '--dataset', default='EmotionROI', type=str,
        choices=['EmotionROI', 'ArtPhoto', 'EmoSet'],
    )
    parser.add_argument('--query_image_features', default='query_image_features.pkl', type=str)
    parser.add_argument('--query_text_features', default='query_text_features.pkl', type=str)
    parser.add_argument('--support_image_features', default='support_image_features.pkl', type=str)
    parser.add_argument('--support_text_features', default='support_text_features.pkl', type=str)
    parser.add_argument('--text_text_similarity', default='text_text_similarity.pkl', type=str)
    parser.add_argument('--clip_text_image_similarity', default='clip_text_image_similarity.pkl', type=str)
    parser.add_argument('--image_image_similarity', default='image_image_similarity.pkl', type=str)
    parser.add_argument(
        "--engine", "-e",
        choices=["otter-llama", "qwen-vl", "idefics-9b-instruct", "llava-onevision-7b", "llava-onevision-0.5b"],
        default=['otter-llama'],
        nargs="+",
    )
    parser.add_argument('--n_shot', default=[1, 2, 4], nargs='+', type=int, help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=32, type=int, help='Max new tokens for generation.')
    parser.add_argument(
        '--task_description', default='concise', type=str,
        choices=['nothing', 'concise', 'detailed'],
        help='Detailed level of task description.',
    )
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--balance_threshold', default=0.5, type=float, help='Balance threshold for whole retrieval.')
    parser.add_argument('--exp_scale', default=10000, type=int, help='EmoSet support pool size (100, 1000, or 10000).')
    parser.add_argument('--score_weight', default=0.7, type=float, help='Visual vs cross-modal weight in whole retrieval.')
    parser.add_argument('--alpha_weight', default=0.6, type=float, help='MMR lambda in whole retrieval.')
    return parser.parse_args()


def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens

    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_image_features}', 'rb') as f:
        query_features = pickle.load(f)

    for query in tqdm(query_meta):
        n_shot_support = ICL_utils.select_demonstration(
            support_meta, n_shot, args.dataset, query, args, query_features,
        )

        predicted_answer = model_inference.ICL_inference(
            args, engine, args.dataset, model, tokenizer, query,
            n_shot_support, data_path, processor, max_new_tokens,
        )

        query['support'] = []
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

    out_subdir = f"{args.dataset}_{STRATEGY}"

    for engine in args.engine:
        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))

        utils.set_random_seed(args.seed)
        for shot in args.n_shot:
            results_dict = eval_questions(
                args, query_meta, support_meta, model, tokenizer, processor, engine, int(shot),
            )
            os.makedirs(f"{args.resultDir}/{out_subdir}", exist_ok=True)

            out_name = (
                f"{engine}_{shot}-shot-balance{args.balance_threshold}-seed{args.seed}"
                f"-score_weight{args.score_weight}-alpha_weight{args.alpha_weight}.json"
            )
            with open(f"{args.resultDir}/{out_subdir}/{out_name}", "w") as f:
                json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()
