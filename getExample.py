import argparse
import pickle
import os
from utils import ICL_utils, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Copy whole-strategy support images for queries')
    parser.add_argument('--dataDir', default='/data1/yq_log/IntenICL/DataSet', type=str)
    parser.add_argument(
        '--dataset', default='EmoSet', type=str,
        choices=['EmotionROI', 'ArtPhoto', 'EmoSet'],
    )
    parser.add_argument(
        '--query_id', nargs='+',
        default=['a5a7e0921cf74346a9422e9ef88356b2'],
        type=str,
    )
    parser.add_argument('--n_shot', default=4, type=int)
    parser.add_argument('--balance_threshold', default=0.5, type=float)
    parser.add_argument('--score_weight', default=0.7, type=float)
    parser.add_argument('--alpha_weight', default=0.6, type=float)
    parser.add_argument('--exp_scale', default=10000, type=int, help='EmoSet support pool (100, 1000, 10000)')
    parser.add_argument('--output', default='example_images.json', type=str)
    parser.add_argument('--query_image_features', default='query_image_features.pkl', type=str)
    parser.add_argument('--query_text_features', default='query_text_features.pkl', type=str)
    parser.add_argument('--support_image_features', default='support_image_features.pkl', type=str)
    parser.add_argument('--support_text_features', default='support_text_features.pkl', type=str)
    parser.add_argument('--text_text_similarity', default='text_text_similarity.pkl', type=str)
    parser.add_argument('--clip_text_image_similarity', default='clip_text_image_similarity.pkl', type=str)
    parser.add_argument('--image_image_similarity', default='image_image_similarity.pkl', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    query_meta, support_meta = utils.load_data(args)

    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_image_features}', 'rb') as f:
        query_image_features = pickle.load(f)

    icl_args = argparse.Namespace(
        dataDir=args.dataDir,
        dataset=args.dataset,
        query_image_features=args.query_image_features,
        support_image_features=args.support_image_features,
        query_text_features=args.query_text_features,
        support_text_features=args.support_text_features,
        text_text_similarity=args.text_text_similarity,
        clip_text_image_similarity=args.clip_text_image_similarity,
        image_image_similarity=args.image_image_similarity,
        balance_threshold=args.balance_threshold,
        score_weight=args.score_weight,
        alpha_weight=args.alpha_weight,
    )

    import shutil
    strategy_dir_name = "whole"

    for qid in args.query_id:
        query = next((item for item in query_meta if item['img_id'] == qid), None)
        if query is None:
            print(f"未找到img_id为{qid}的查询样本")
            continue
        base_out_dir = f"/data1/yq_log/IntenICL/icl_log_new/{qid}"
        os.makedirs(base_out_dir, exist_ok=True)

        query_img_field = query.get('image', None)
        if query_img_field is not None:
            query_img_list = query_img_field if isinstance(query_img_field, list) else [query_img_field]
            for img_path in query_img_list:
                if not isinstance(img_path, str):
                    print(f"警告: 查询图片路径类型错误: {img_path}")
                    continue
                img_path_full = img_path if os.path.isabs(img_path) else os.path.join(args.dataDir, img_path)
                if os.path.exists(img_path_full):
                    target_path = os.path.join(base_out_dir, os.path.basename(img_path_full))
                    shutil.copy(img_path_full, target_path)
                else:
                    print(f"警告: 找不到查询图片 {img_path_full}")

        strategy_dir = os.path.join(base_out_dir, strategy_dir_name)
        os.makedirs(strategy_dir, exist_ok=True)
        n_shot_support = ICL_utils.select_demonstration(
            support_meta, args.n_shot, args.dataset, query, icl_args, query_image_features,
        )
        for s in n_shot_support:
            img_field = s['image']
            img_list = img_field if isinstance(img_field, list) else [img_field]
            for img_path in img_list:
                if not isinstance(img_path, str):
                    print(f"警告: 非法图片路径类型: {img_path}")
                    continue
                img_path_full = img_path if os.path.isabs(img_path) else os.path.join(args.dataDir, img_path)
                if os.path.exists(img_path_full):
                    target_path = os.path.join(strategy_dir, os.path.basename(img_path_full))
                    shutil.copy(img_path_full, target_path)
                else:
                    print(f"警告: 找不到图片 {img_path_full}")
        print(f"whole 策略示例及查询图片已保存到 {base_out_dir}")


if __name__ == "__main__":
    main()
