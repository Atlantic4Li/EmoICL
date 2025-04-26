import csv
import torch
from itertools import islice, cycle
import os
import random
import numpy as np
import json
import base64
from PIL import Image


def set_random_seed(seed_number):
    # position of setting seeds also matters
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.random.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def truncate_prediction(prediction: str) -> str:
    """Truncate captions at the first newline character, removing leading spaces."""
    prediction = prediction.strip()  # Remove leading and trailing whitespace
    trunc_index = prediction.find('\n')
    if trunc_index != -1:
        prediction = prediction[:trunc_index].strip()
    else:
        # If no newline is found, find the first period and truncate
        trunc_index = prediction.find('.') + 1
        if trunc_index > 0:
            prediction = prediction[:trunc_index].strip()
    return prediction


def load_image(img_ids, root_path):
    if isinstance(img_ids, str):
        img_ids = [img_ids]
    images = []
    image_paths = []
    for img_id in img_ids:
        image_path = os.path.join(root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        image_paths.append(image_path)
        
    return images, image_paths

def coco_id_to_imgname(img_id, prefix='COCO_val2014_'):
    return f'{prefix}{img_id:012}.jpg'

def load_data(args):
    dataDir = args.dataDir
    query_file = os.path.join(dataDir, args.dataset, 'query.json')
    
    # 为EmoSet数据集选择不同规模的支持集
    if args.dataset == 'EmoSet' and hasattr(args, 'exp_scale'):
        # 根据exp_scale参数选择对应的支持集文件
        if args.exp_scale in [100, 1000, 10000]:
            support_file = os.path.join(dataDir, args.dataset, f'support_{args.exp_scale}.json')
            print(f"使用EmoSet数据集的规模为{args.exp_scale}的支持集: {support_file}")
        else:
            # 如果exp_scale不是预设值，使用默认支持集
            support_file = os.path.join(dataDir, args.dataset, 'support.json')
            print(f"警告: exp_scale值 {args.exp_scale} 无效，使用默认支持集: {support_file}")
    else:
        # 对于其他数据集，使用默认的support.json
        support_file = os.path.join(dataDir, args.dataset, 'support.json')

    with open(query_file, 'r') as f:
        query_meta = json.load(f)
    with open(support_file, 'r') as f:
        support_meta = json.load(f)

    return query_meta, support_meta
    