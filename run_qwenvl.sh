#!/bin/bash

# # 数据集固定为EmoSet
# dataset="EmoSet"

# # random策略，n_shot为[0,1,2,4,8]，balance_threshold为0.5
# echo "运行配置: dataset=$dataset, strategy=random, n_shot=[0,1,2,4,8], balance_threshold=0.5"
# python inference.py --dataset "$dataset" --strategy random --n_shot 0 1 2 4 8 --balance_threshold 0.5

# # similarity策略，n_shot为[1,2,4,8]，balance_threshold为0.5
# echo "运行配置: dataset=$dataset, strategy=similarity, n_shot=[1,2,4,8], balance_threshold=0.5"
# python inference.py --dataset "$dataset" --strategy similarity --n_shot 1 2 4 8 --balance_threshold 0.5

# # whole策略，n_shot为[1,2,4,8]，balance_threshold为0.5
# echo "运行配置: dataset=$dataset, strategy=whole, n_shot=[1,2,4,8], balance_threshold=0.5"
# python inference.py --dataset "$dataset" --strategy whole --n_shot 1 2 4 8 --balance_threshold 0.5

# # whole策略，n_shot为[1,2,4,8]，balance_threshold为0.6
# echo "运行配置: dataset=$dataset, strategy=whole, n_shot=[1,2,4,8], balance_threshold=0.6"
# python inference.py --dataset "$dataset" --strategy whole --n_shot 1 2 4 8 --balance_threshold 0.6

datasets=('EmoSet')

for dataset in "${datasets[@]}"; do
    # similarity_reverse策略，n_shot为[1,2,4,8]，balance_threshold为0.5
    echo "运行配置: dataset=$dataset, strategy=random, n_shot=[0,1,2,4,8], seed=1"
    python inference.py --dataset "$dataset" --strategy random --n_shot 0 1 2 4 8 --seed 1 --exp_scale 10000
    
    # whole_reverse策略，n_shot为[1,2,4,8]，balance_threshold为0.5
    echo "运行配置: dataset=$dataset, strategy=random, n_shot=[0,1,2,4,8], seed=2"
    python inference.py --dataset "$dataset" --strategy random --n_shot 0 1 2 4 8 --seed 2 --exp_scale 10000

    echo "运行配置: dataset=$dataset, strategy=random, n_shot=[0,1,2,4,8], seed=3"
    python inference.py --dataset "$dataset" --strategy random --n_shot 0 1 2 4 8 --seed 3 --exp_scale 10000

    echo "运行配置: dataset=$dataset, strategy=random, n_shot=[0,1,2,4,8], seed=4"
    python inference.py --dataset "$dataset" --strategy random --n_shot 0 1 2 4 8 --seed 4 --exp_scale 10000

done
