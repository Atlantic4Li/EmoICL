import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import argparse
from evals import eval

metrics = {
    "EmotionROI": "Acc",
    "ArtPhoto": "Acc",
    "EmoSet": "Acc",
}

STRATEGY = "whole"


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL evaluation')

    parser.add_argument('--dataDir', default='/data1/yq_log/IntenICL/DataSet', type=str, help='Data directory.')
    parser.add_argument('--resultDir', default='/data1/yq_log/IntenICL/ICL_log', type=str, help='result directory.')
    parser.add_argument(
        '--dataset', default=['EmotionROI'], nargs='+', type=str,
        choices=['EmotionROI', 'ArtPhoto', 'EmoSet'],
    )
    parser.add_argument('--sim_path', default='cross_similarity.pkl', type=str)
    parser.add_argument('--support_path', default='support_features.pkl', type=str)
    parser.add_argument('--query_path', default='query_features.pkl', type=str)
    parser.add_argument(
        "--engine", "-e",
        choices=["otter-llama", "qwen-vl", "idefics-9b-instruct", "llava-onevision-7b", "llava-onevision-0.5b"],
        default=['llava-onevision-7b', "qwen-vl"],
        nargs="+",
    )
    parser.add_argument('--n_shot', default=[1, 2, 4, 8], nargs="+", help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=256, type=int)
    parser.add_argument('--task_description', default='concise', type=str, choices=['nothing', 'concise', 'detailed'])
    parser.add_argument('--ft', default=False, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--balance_threshold', default=0.5, type=float)
    parser.add_argument('--exp_scale', default=10000, type=int)
    parser.add_argument('--score_weight', default=0.7, type=float)
    parser.add_argument('--alpha_weight', default=0.6, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for dataset in args.dataset:
        print(f"\n=== Evaluating Dataset: {dataset} ===")

        if args.ft:
            result_files = [
                f"{args.resultDir}/{dataset}_{STRATEGY}/{engine}_{shot}-shot-ft2-balance{args.balance_threshold}.json"
                for engine in args.engine
                for shot in args.n_shot
            ]
        else:
            result_files = [
                f"{args.resultDir}/{dataset}_{STRATEGY}/{engine}_{shot}-shot-balance{args.balance_threshold}-seed{args.seed}-score_weight{args.score_weight}-alpha_weight{args.alpha_weight}.json"
                for engine in args.engine
                for shot in args.n_shot
            ]

        for result_file in result_files:
            try:
                engine, shot, score_weight, alpha_weight = result_file.split("/")[-1].replace(".json", "").split("_")

                if not os.path.exists(result_file):
                    print(f"Warning: Result file not found: {result_file}")
                    continue

                with open(result_file, "r") as f:
                    results_dict = json.load(f)

                score = eval.eval_scores(results_dict, dataset, engine)
                formatted_scores = ', '.join([f"{thresh}->{acc*100:.2f}%" for thresh, acc in sorted(score.items())])
                print(f'{dataset}_{STRATEGY} {metrics[dataset]} of {engine} {shot} {score_weight} {alpha_weight}: {formatted_scores}', flush=True)

            except Exception as e:
                print(f"Error processing {result_file}: {str(e)}")
                continue
