#!/usr/bin/env python3
"""concept_extraction_openai.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End‑to‑end **concept‑extraction** utility supporting three VQA‑style datasets
(ScienceQA, m3cot, LLaVA‑W) and three generation back‑ends:

1. **Local HF models**      – original behaviour (GPU required).
2. **OpenAI Chat Completion** – call GPT‑style models via OpenAI API.
3. **Prompt‑only**           – emit ready‑to‑paste prompts without generating.

The CLI remains largely compatible with the original `concept_extraction.py` but
adds a few flags so you can switch back‑ends in one line.

Key CLI flags
-------------
```
--dataset      {scienceqa|m3cot|llava_w}
--split        Dataset split (default: test)
--mode         {prompt|generate}      (default: generate)
--backend      {auto|hf|openai}      (default: auto)
--model_path   HF model path / id    (HF backend)
--openai_model OpenAI model name     (OpenAI backend, default: gpt-4o-mini)
--api_key      OpenAI key or env var OPENAI_API_KEY
--batch_size   Generation batch size (default: 4)
--output       Output JSONL path (required)
--root         Datasets root dir (default: /data1/yq_log/VCot/Datasets)
```

Example usage
-------------
```bash
# 1) Produce prompts only (fast, no GPU)
python concept_extraction_openai.py \
    --dataset scienceqa \
    --mode prompt \
    --output /tmp/sqa_test_prompts.jsonl

# 2) Generate concepts with local Chameleon‑7B (unchanged)
python concept_extraction_openai.py \
    --dataset m3cot \
    --backend hf \
    --model_path /data1/yq_log/VCot/HF/facebook/chameleon-7b \
    --output /tmp/m3cot_concepts_chameleon.jsonl

# 3) **NEW** – Use OpenAI GPT‑4o‑mini via API (no GPU needed)
python concept_extraction_openai.py \
    --dataset llava_w \
    --backend openai \
    --openai_model gpt-4o-mini \
    --api_key sk-...YOUR_KEY... \
    --output /tmp/llava_w_concepts_gpt.jsonl
```
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import pickle
import random
import datetime
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import requests
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Optional, Any, Union
from scipy.spatial.distance import cdist

from tqdm import tqdm  # progress bar for large splits

# 添加父目录到路径，以便导入utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import ICL_utils, utils  # 导入ICL工具函数

# ---------------------------------------------------------------------------
# Constants & utilities
# ---------------------------------------------------------------------------

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # up to 26 choices

# ---------------------------------------------------------------------------
# 任务相关配置和辅助函数
# ---------------------------------------------------------------------------
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # up to 26 choices

# 图像处理函数
def encode_image_to_base64(image_path):
    """将图像文件转换为base64编码字符串"""
    try:
        # 首先尝试处理本地文件
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # 如果本地文件不存在，尝试将其作为URL处理
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            response.raise_for_status()  # 确保请求成功
            return base64.b64encode(response.content).decode('utf-8')
        
        # 如果既不是本地文件也不是有效URL，则抛出异常
        raise FileNotFoundError(f"无法找到图像文件或URL: {image_path}")
    
    except Exception as e:
        print(f"警告：图像编码失败 - {image_path}: {str(e)}")
        return None


def get_full_image_path(image_path, dataset, data_path):
    """获取完整的图像路径"""
    # 处理列表类型的image_path
    if isinstance(image_path, list):
        # 如果是列表，取第一个元素（假设这是主要图像路径）
        # 或者根据需求选择其他处理方式
        if len(image_path) > 0:
            image_path = image_path[0]
        else:
            raise ValueError("空的图像路径列表")
    
    # 如果已经是绝对路径或URL，则直接返回
    if os.path.isabs(image_path) or image_path.startswith(('http://', 'https://')):
        return image_path
    
    # 构建数据集特定的图像路径
    return os.path.join(data_path, image_path)


# 图像缓存字典，避免重复加载和编码
IMAGE_CACHE = {}


def get_encoded_image(image_path, dataset, data_path):
    """获取图像的base64编码，带缓存"""
    # 确保image_path是字符串类型（可哈希）
    if isinstance(image_path, list):
        # 如果是列表，将其转换为字符串以便可以用作字典键
        cache_key = str(image_path)
    else:
        cache_key = image_path
    
    # 使用缓存避免重复加载相同图像
    if cache_key in IMAGE_CACHE:
        return IMAGE_CACHE[cache_key]
    
    # 获取完整路径并编码
    full_path = get_full_image_path(image_path, dataset, data_path)
    encoded = encode_image_to_base64(full_path)
    
    # 缓存结果
    if encoded:
        IMAGE_CACHE[cache_key] = encoded
    
    return encoded


# ICL系统提示模板
def get_icl_system_prompt(task_description='concise'):
    """根据任务描述级别获取系统提示"""
    if task_description == 'nothing':
        return "You are a helpful assistant."
    elif task_description == 'concise':
        return ("You are a helpful visual assistant that can understand images. "
                "Look at the image and answer the question in a concise way.")
    else:  # detailed
        return ("You are a helpful visual assistant that can understand images. "
                "Look at the image and answer the question based on what you see in the image. "
                "Provide a clear, concise and accurate answer based on the visual content.")


def get_dataset_task_instruction(dataset, description='concise'):
    """根据数据集和任务描述级别获取特定的任务指令"""
    if dataset == 'Intentonomy':
        # Intentonomy数据集通用指令
        if description == 'concise':
            return "What is the intent or purpose of this image? Answer directly."
        else:
            return "Analyze the image and determine the intent or purpose. Provide a clear answer."
            
    elif dataset == 'EmotionROI':
        if description == 'concise':
            return 'Select the emotional category to which the image belongs from the options below: ["anger","disgust","fear","joy","sadness","surprise"]. Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
        elif description == 'detailed':
            return 'When you see this picture, which of the following categories do you think this picture belongs to?'
            
    elif dataset == 'ArtPhoto':
        if description == 'concise':
            return 'Select the emotional category to which the image belongs from the options below: ["awe","contentment","sad","amusement","excitement","fear","anger","disgust"]. Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
        elif description == 'detailed':
            return 'When you see this picture, which of the following categories do you think this picture belongs to?'
            
    elif dataset == 'EmoSet':
        if description == 'concise':
            return 'Select the emotional category to which the image belongs from the options below: ["amusement","anger","awe","contentment","disgust","excitement","fear","sadness"]. Your response must strictly follow this format: "The emotion of this image is [emotion category]".'
        elif description == 'detailed':
            return 'When you see this picture, which of the following categories do you think this picture belongs to?'
            
    # 对于其他数据集，提供通用的任务指令
    else:
        if description == 'concise':
            return "What category does this image belong to? Answer directly."
        else:
            return "Analyze the image and determine its category. Provide a clear answer."


# ---------------------------------------------------------------------------
# ICL 格式化和提示构建函数
# ---------------------------------------------------------------------------

def format_icl_example(example: dict, dataset: str, data_path: str) -> dict:
    """格式化单个ICL示例，返回包含文本和图像的消息字典"""
    # 1. 构建示例的文本部分
    text_content = ""
    
    # 2. 根据数据集类型添加不同的格式
    if dataset in ['Intentonomy', 'EmotionROI', 'ArtPhoto', 'EmoSet']:
        # 情感/意图分类任务
        text_content += f"Answer: {example['answer']}\n"
    else:
        # 其他分类任务
        text_content += f"Category: {example['category']}\n"
    
    # 3. 获取图像的base64编码
    image_b64 = get_encoded_image(example['image'], dataset, data_path)
    
    # 4. 构建消息对象（包含文本和图像URL）
    message_content = []
    
    # 添加图像部分
    if image_b64:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })
    
    # 添加文本部分
    message_content.append({
        "type": "text",
        "text": text_content
    })
    
    return {
        "role": "user", 
        "content": message_content
    }


def build_icl_messages(query: dict, support_samples: List[dict], dataset: str, task_desc: str, data_path: str) -> List[Dict]:
    """构建完整的ICL消息列表，适用于OpenAI vision API"""
    messages = []
    
    # 1. 添加系统提示
    messages.append({
        "role": "system", 
        "content": get_icl_system_prompt(task_desc)
    })
    
    # 2. 添加支持示例（每个示例包含图片和回答）
    for example in support_samples:
        # 添加用户消息（图片）
        messages.append(format_icl_example(example, dataset, data_path))
        
        # 添加助手回复（答案）
        messages.append({
            "role": "assistant",
            "content": example.get("answer", example.get("category", ""))
        })
    
    # 3. 添加查询样本（只有图片，没有答案）
    query_message_content = []
    
    # 获取查询图像的base64编码
    if 'image' in query:
        query_image_b64 = get_encoded_image(query['image'], dataset, data_path)
        if query_image_b64:
            query_message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{query_image_b64}"
                }
            })
    
    # 添加数据集特定的任务说明
    task_text = get_dataset_task_instruction(dataset, task_desc)
    
    query_message_content.append({
        "type": "text",
        "text": task_text
    })
    
    messages.append({
        "role": "user",
        "content": query_message_content
    })
    
    return messages


# 删除纯文本备用模式函数


# ---------------------------------------------------------------------------
# Dataset loaders and ICL utilities
# ---------------------------------------------------------------------------

def load_data_and_features(args):
    """加载查询和支持集数据、特征和相似度矩阵"""
    query_meta, support_meta = utils.load_data(args)
    
    data_results = {
        'query_meta': query_meta,
        'support_meta': support_meta
    }
    
    return data_results

# ICL 示例选择函数
def select_icl_examples(args, query, support_meta, n_shot, strategy):
    """根据选择策略为查询样本选择支持样本"""
    # 加载查询特征（这里可能需要优化以避免重复加载）
    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_image_features}', 'rb') as f:
        query_features = pickle.load(f)
    
    # 调用ICL_utils中的select_demonstration函数
    support_examples = ICL_utils.select_demonstration(
        support_meta=support_meta,
        n_shot=n_shot,
        dataset=args.dataset,
        strategy=strategy,
        query=query,
        args=args,
        query_features=query_features
    )
    
    return support_examples


# ICL推理主函数
def icl_inference(args, query, support_examples, client):
    """执行ICL推理，支持多模态输入"""
    # 1. 构建多模态消息列表
    try:
        messages = build_icl_messages(
            query=query,
            support_samples=support_examples,
            dataset=args.dataset,
            task_desc=args.task_description,
            data_path=args.dataDir
        )
        
        # 2. 调用OpenAI API (Vision模型)
        if hasattr(client, "chat"):
            # OpenAI SDK v1
            response = client.chat.completions.create(
                model=args.openai_model,  # 确保使用支持视觉的模型，如gpt-4-vision-preview
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
        else:
            # OpenAI SDK v0
            response = client.ChatCompletion.create(
                model=args.openai_model,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=0.7,
            )
            answer = response["choices"][0]["message"]["content"].strip()
        
        return answer
    
    except Exception as e:
        print(f"[ERROR] 多模态API调用失败: {str(e)}")
        return f"ERROR: 多模态调用失败: {str(e)}"


# 不再需要的数据加载器和加载器字典已删除

# ---------------------------------------------------------------------------
# Prompt wrappers for different back‑ends
# ---------------------------------------------------------------------------

def wrap_prompt_hf(prompt: str, model_type: str, task_desc: str = 'concise') -> str:
    """Return prompt string adjusted for HF causal / ChatML models."""
    system_prompt = get_icl_system_prompt(task_desc)
    if model_type == "qwen":  # ChatML format
        return (
            "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"  # system
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    # chameleon or generic causal‑LM expects plain text
    return f"System\n{system_prompt}\nUser\n{prompt}"


# 此函数已重构为仅用于HF模型


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def run_generation_hf(model, tokenizer, prompts: List[str], generation_cfg: Optional[dict] = None) -> List[str]:
    """Generate completions with a *local* HF model."""
    import torch

    # Ensure padding side & token for batch encoding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    gen_kwargs = generation_cfg.copy() if generation_cfg else {}
    gen_kwargs.setdefault("do_sample", False)
    gen_kwargs.setdefault("max_new_tokens", 256)
    gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    completions: List[str] = []
    for i in range(len(prompts)):
        full = out[i]
        prompt_len = (inputs.input_ids[i] != tokenizer.pad_token_id).sum()
        text = tokenizer.decode(full[prompt_len:], skip_special_tokens=True).strip()
        completions.append(text.lstrip("```\n").rstrip("```\n"))
    return completions


def run_generation_openai(batch: List[List[Dict[str, str]]], model: str, client):
    """Generate completions with either OpenAI SDK v1 (preferred) or v0 fallback."""
    outs = []
    for msg in batch:
        try:
            if hasattr(client, "chat"):
                # SDK ≥1.0 – client is an OpenAI() instance
                resp = client.chat.completions.create(
                    model=model,
                    messages=msg,
                    temperature=0.0,
                )
                # outs.append(resp.choices[0].message.content.strip())
                content = resp.choices[0].message.content
                if content is not None:
                    outs.append(content.strip())
                else:
                    outs.append("")  # 或者记录异常、输出警告
            else:
                # legacy openai module (≤0.28)
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=msg,
                    temperature=0.0,
                )
                # outs.append(resp.choices[0].message.content.strip())
                content = resp.choices[0].message.content
                if content is not None:
                    outs.append(content.strip())
                else:
                    outs.append("")  # 或者记录异常、输出警告
        except Exception as e:
            # <-- key addition: emit the failing uid so we can locate bad samples
            print(f"[ERROR] generation failed")
            outs.append("")  # keep list length consistent
    return outs


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser("ICL image-to-text with GPT – datasets & multiple back‑ends")
    p.add_argument("--dataDir", default='/data1/yq_log/IntenICL/DataSet', type=str, help='Data directory.')
    p.add_argument("--resultDir", default='/data1/yq_log/IntenICL/ICL_log', type=str, help='Result directory.')
    p.add_argument("--dataset", default="EmotionROI", type=str, 
                   choices=["Intentonomy", "EmotionROI", "ArtPhoto", "EmoSet", "StanfordCars", "OxfordFlowers17", "StanfordDogs", "CUB_200_2011", "Oxford-IIIT_Pet"],
                   help="Dataset to use")
    p.add_argument("--split", default="test")
    p.add_argument("--query_image_features", default='query_image_features.pkl', type=str, help='Features of query images')
    p.add_argument("--query_text_features", default='query_text_features.pkl', type=str, help='Features of query texts')
    p.add_argument("--support_image_features", default='support_image_features.pkl', type=str, help='Features of support images')
    p.add_argument("--support_text_features", default='support_text_features.pkl', type=str, help='Features of support texts')
    p.add_argument("--text_text_similarity", default='text_text_similarity.pkl', type=str, help='Similarity between texts')
    p.add_argument("--clip_text_image_similarity", default='clip_text_image_similarity.pkl', type=str, help='Cross-modal similarity between text and image')
    p.add_argument("--image_image_similarity", default='image_image_similarity.pkl', type=str, help='Similarity between images')
    p.add_argument("--mode", choices=["prompt", "generate"], default="generate")
    p.add_argument("--backend", choices=["auto", "hf", "openai"], default="openai", help="Generation back‑end")
    
    # ICL strategy arguments
    p.add_argument("--strategy", default=['qtmt'], nargs='+', type=str, 
                  choices=['random', 'similarity', 'various', 'test_same_similarity', 'test_same_random', 'test_text_similarity',
                           'test_diverse', 'whole', 'qtmt', 'mmices', 'muier', 'circles', 'cdsp'],
                  help='Example selection strategy.')
    p.add_argument("--n_shot", default=[1,2,4], nargs='+', type=int, help='Number of support examples.')
    p.add_argument("--balance_threshold", default=0.5, type=float, help='Balance threshold for various strategy.')
    p.add_argument("--task_description", default='concise', type=str, choices=['nothing', 'concise', 'detailed'], 
                  help='Detailed level of task description.')
    p.add_argument("--seed", default=0, type=int, help='Random seed.')
    p.add_argument("--exp_scale", default=10000, type=int, help='Support scale for EmoSet dataset')
    p.add_argument("--max-new-tokens", default=32, type=int, help='Max new tokens for generation.')
    
    # OpenAI backend specifics
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    p.add_argument("--api_key", type=str, default="sk-Y8irEzDImliHloA2KMGg8uEBExJleczU5Jcfjps3tiPjlZdm", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--base_url", type=str, default="https://api.poixe.com/v1", help="Custom OpenAI‑compatible endpoint")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", type=Path, help="Output file path (if not specified, defaults to resultDir/dataset_strategy)")
    args = p.parse_args(argv)
    
    # 如果未指定输出路径，则根据数据集和策略名称自动生成
    if args.output is None:
        strategy = args.strategy[0] if isinstance(args.strategy, list) else args.strategy
        args.output = Path(f"{args.resultDir}/{args.dataset}_{strategy}")
    
    return args


# ---------------------------------------------------------------------------
# ICL评估和主函数
# ---------------------------------------------------------------------------

def eval_questions(args, query_meta, support_meta, client, strategy, n_shot):
    """执行ICL推理评估并保存结果"""
    results = []
    
    # 预加载query_image_features以提高效率
    with open(f'{args.dataDir}/{args.dataset}/query/{args.query_image_features}', 'rb') as f:
        query_features = pickle.load(f)

    for query in tqdm(query_meta):
        # 选择支持样本
        n_shot_support = ICL_utils.select_demonstration(
            support_meta=support_meta, 
            n_shot=n_shot, 
            dataset=args.dataset, 
            strategy=strategy, 
            query=query, 
            args=args, 
            query_features=query_features
        )
        
        # 执行ICL推理
        predicted_answer = icl_inference(args, query, n_shot_support, client)
        
        # 构造结果
        result_entry = query.copy()
        result_entry['support'] = []
        for support_sample in n_shot_support:
            result_entry['support'].append({
                "image": support_sample["image"],
                "answer": support_sample["answer"] if "answer" in support_sample else support_sample.get("category", "")
            })
        result_entry['prediction'] = predicted_answer
        results.append(result_entry)

    return results


def main(argv: List[str] | None = None):
    """Entry‑point handling CLI parsing, backend initialisation and the full
    dataset → ICL inference → result saving flow."""

    args = parse_args(argv)
    
    # 设置随机种子
    utils.set_random_seed(args.seed)
    
    # ------------------------------------------------------------------
    # 1) 加载数据
    # ------------------------------------------------------------------
    print(f"📂 加载{args.dataset}数据集...")
    query_meta, support_meta = utils.load_data(args)
    
    # ------------------------------------------------------------------
    # 2) OpenAI API 初始化
    # ------------------------------------------------------------------
    print(f"🔌 初始化OpenAI客户端...")
    try:
        import openai
    except ImportError:
        print("❌ 未安装openai包，请使用pip install openai安装")
        return
    
    client = None
    if hasattr(openai, "OpenAI"):  # OpenAI SDK v1.x
        # 优先使用参数中的API密钥，其次使用环境变量
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ 请提供OpenAI API密钥（使用--api_key参数或设置OPENAI_API_KEY环境变量）")
            return
            
        client = openai.OpenAI(
            api_key=api_key,
            base_url=args.base_url
        )
    else:  # OpenAI SDK v0.x
        openai.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not openai.api_key:
            print("❌ 请提供OpenAI API密钥（使用--api_key参数或设置OPENAI_API_KEY环境变量）")
            return
            
        if args.base_url != "https://api.openai.com/v1":
            openai.api_base = args.base_url
        client = openai
    
    # ------------------------------------------------------------------
    # 3) 执行多种策略和shot数的实验
    # ------------------------------------------------------------------
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for strategy in args.strategy:
        print(f"🔄 使用{strategy}策略...")
        
        for shot in args.n_shot:
            print(f"  📊 执行{shot}-shot推理...")
            
            # 执行ICL推理评估
            results = eval_questions(args, query_meta, support_meta, client, strategy, int(shot))
            
            # 创建输出目录
            os.makedirs(f"{args.resultDir}/{args.dataset}_{strategy}", exist_ok=True)
            
            # 保存结果
            output_file = f"{args.resultDir}/{args.dataset}_{strategy}/openai-{args.openai_model}_{shot}-shot-balance{args.balance_threshold}-seed{args.seed}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"  ✅ 结果已保存到 {output_file}")
    
    print(f"🎉 所有实验完成!")


if __name__ == "__main__":
    main()