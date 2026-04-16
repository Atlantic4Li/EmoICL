import argparse
import pickle
import os
import json
from utils import ICL_utils

def parse_args():
    parser = argparse.ArgumentParser(description='Get Example Images for a Query')
    parser.add_argument('--dataDir', default='/data1/yq_log/IntenICL/DataSet', type=str, help='数据目录')
    parser.add_argument('--dataset', default='EmoSet', type=str, help='数据集名称')
    parser.add_argument('--query_id', nargs='+', default=['a5a7e0921cf74346a9422e9ef88356b2','6afeb0980ab24d5e9aca12109a909cfe','947891bf6b75407589ce7e8385b7466e','dc685cb55a884ca5a268ea7195125cd5','d8b9aefffc9c409883473053ea23a11c','81015145fbe8400782c2ee8448afb7e9'], type=str, help='查询样本的img_id列表')
    parser.add_argument('--strategies', nargs='+', default=['random','similarity','whole','qtmt','mmices','muier','cdsp'], help='示例选择策略')
    parser.add_argument('--n_shot', default=4, type=int, help='示例数量')
    parser.add_argument('--balance_threshold', default=0.5, type=float, help='类别平衡阈值')
    parser.add_argument('--output', default='example_images.json', type=str, help='输出文件名')
    parser.add_argument('--query_image_features', default='query_image_features.pkl', type=str)
    parser.add_argument('--query_text_features', default='query_text_features.pkl', type=str)
    parser.add_argument('--support_image_features', default='support_image_features.pkl', type=str)
    parser.add_argument('--support_text_features', default='support_text_features.pkl', type=str)
    parser.add_argument('--text_text_similarity', default='text_text_similarity.pkl', type=str)
    parser.add_argument('--clip_text_image_similarity', default='clip_text_image_similarity.pkl', type=str)
    parser.add_argument('--image_image_similarity', default='image_image_similarity.pkl', type=str)
    return parser.parse_args()



def load_meta(dataDir, dataset):
    query_file = os.path.join(dataDir, dataset, 'query.json')
    
    # 为EmoSet数据集选择不同规模的支持集
    if dataset == 'EmoSet':
        # 根据exp_scale参数选择对应的支持集文件
        support_file = os.path.join(dataDir, dataset, f'support_10000.json')
    else:
        # 对于其他数据集，使用默认的support.json
        support_file = os.path.join(dataDir, dataset, 'support.json')

    with open(query_file, 'r') as f:
        query_meta = json.load(f)
    with open(support_file, 'r') as f:
        support_meta = json.load(f)

    return query_meta, support_meta

def main():
    args = parse_args()
    query_meta, support_meta = load_meta(args.dataDir, args.dataset)

    # 预加载特征
    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_image_features}', 'rb') as f:
        query_image_features = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/support/{args.support_image_features}', 'rb') as f:
        support_image_features = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_text_features}', 'rb') as f:
        query_text_features = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/support/{args.support_text_features}', 'rb') as f:
        support_text_features = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/similarity/{args.text_text_similarity}', 'rb') as f:
        text_text_similarity = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/similarity/{args.clip_text_image_similarity}', 'rb') as f:
        clip_text_image_similarity = pickle.load(f)
    with open(f'{args.dataDir}/{args.dataset}/similarity/{args.image_image_similarity}', 'rb') as f:
        image_image_similarity = pickle.load(f)

    # 构造参数对象
    class Args:
        pass
    icl_args = Args()
    icl_args.dataDir = args.dataDir
    icl_args.dataset = args.dataset
    icl_args.query_image_features = args.query_image_features
    icl_args.support_image_features = args.support_image_features
    icl_args.query_text_features = args.query_text_features
    icl_args.support_text_features = args.support_text_features
    icl_args.text_text_similarity = args.text_text_similarity
    icl_args.clip_text_image_similarity = args.clip_text_image_similarity
    icl_args.image_image_similarity = args.image_image_similarity
    icl_args.balance_threshold = args.balance_threshold

    import shutil
    for qid in args.query_id:
        query = next((item for item in query_meta if item['img_id'] == qid), None)
        if query is None:
            print(f"未找到img_id为{qid}的查询样本")
            continue
        base_out_dir = f"/data1/yq_log/IntenICL/icl_log_new/{qid}"
        os.makedirs(base_out_dir, exist_ok=True)

        # 复制查询图像到 query_id 文件夹下
        query_img_field = query.get('image', None)
        if query_img_field is not None:
            query_img_list = query_img_field if isinstance(query_img_field, list) else [query_img_field]
            for img_path in query_img_list:
                if not isinstance(img_path, str):
                    print(f"警告: 查询图片路径类型错误: {img_path}")
                    continue
                if not os.path.isabs(img_path):
                    img_path_full = os.path.join(args.dataDir, img_path)
                else:
                    img_path_full = img_path
                if os.path.exists(img_path_full):
                    target_path = os.path.join(base_out_dir, os.path.basename(img_path_full))
                    shutil.copy(img_path_full, target_path)
                else:
                    print(f"警告: 找不到查询图片 {img_path_full}")

        for strategy in args.strategies:
            strategy_dir = os.path.join(base_out_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
            n_shot_support = ICL_utils.select_demonstration(
                support_meta, args.n_shot, args.dataset, strategy, query, icl_args, query_image_features
            )
            for s in n_shot_support:
                img_field = s['image']
                img_list = img_field if isinstance(img_field, list) else [img_field]
                for img_path in img_list:
                    if not isinstance(img_path, str):
                        print(f"警告: 非法图片路径类型: {img_path}")
                        continue
                    # 处理绝对/相对路径
                    if not os.path.isabs(img_path):
                        img_path_full = os.path.join(args.dataDir, img_path)
                    else:
                        img_path_full = img_path
                    if os.path.exists(img_path_full):
                        target_path = os.path.join(strategy_dir, os.path.basename(img_path_full))
                        shutil.copy(img_path_full, target_path)
                    else:
                        print(f"警告: 找不到图片 {img_path_full}")
        print(f"所有策略图片及查询图片已保存到 {base_out_dir}")

if __name__ == "__main__":
    main()
