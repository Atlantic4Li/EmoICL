import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
import json
import random
import numpy as np
from randaugment import RandAugment
from helper import CutoutPIL
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# 禁用不必要的警告
warnings.filterwarnings("ignore")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 启用cudnn基准优化

class BaseDataset(Dataset):
    """基础数据集类"""
    def __init__(self, root, processor, phase='train', img_size=224):
        self.root = root
        self.processor = processor
        self.phase = phase
        self.img_size = img_size
        
        # 设置随机种子以确保可复现性
        self.set_seed(42)
        
        # 加载数据标注
        self.load_annotations()
        
        # 设置数据增强
        self.setup_transforms()
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def setup_transforms(self):
        """设置数据转换"""
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
    
    def load_annotations(self):
        """加载数据标注（由子类实现）"""
        raise NotImplementedError
    
    def _load_image(self, index):
        """加载图像"""
        image_path = os.path.join(self.root, self.annotations[index]['image'][0])
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # 返回一个随机的替代图像
            return self._load_image((index + 1) % len(self))

    def __len__(self):
        """返回数据集大小"""
        return len(self.annotations)

    def __getitem__(self, index):
        """获取单个数据样本"""
        # 加载图像
        image = self._load_image(index)
        
        # 应用数据转换
        if self.transform:
            image = self.transform(image)
        
        # 处理图像，设置do_rescale=False因为ToTensor()已经将像素值归一化到[0,1]
        if self.phase == 'train':
            processed = self.processor(
                images=image,
                return_tensors="pt",
                do_rescale=False,  # 避免重复归一化
                do_random_flip=True,
                do_resize=True,
                do_center_crop=False
            )
        else:
            processed = self.processor(
                images=image,
                return_tensors="pt",
                do_rescale=False  # 避免重复归一化
            )
        
        # 获取类别标签
        category_idx = self.annotations[index]['category']
        
        return {
            'pixel_values': processed['pixel_values'][0],
            'label': torch.tensor(category_idx, dtype=torch.long)
        }

class IntentonomyDataset(BaseDataset):
    """Intentonomy数据集"""
    def __init__(self, root, processor, phase='train', img_size=224):
        self.classnames = [
            "Being physically or personally attractive", "Competing to outperform others",
            "Communicating with others", "Expressing creativity and uniqueness",
            "Living a curious, adventurous, and exciting life", "Having a easy life",
            "Enjoying life", "Appreciating architecture", "Appreciating artistic expression",
            "Appreciating cultural heritage", "Being a good parent with emotional closeness to children",
            "Experiencing happiness", "Working diligently", "Maintaining balance and harmony",
            "Keeping healthy", "Being in love", "Having affection for animals",
            "Inspiring others", "Effectively managing tasks and making plans",
            "Appreciating natural beauty", "Having a passion for something",
            "Being playful", "Sharing feelings", "Maintaining social life and friendships",
            "Achieving occupational success and job satisfaction", "Teaching others",
            "Keeping things orderly", "Engaging in personally fulfilling work"
        ]
        super().__init__(root, processor, phase, img_size)
    
    def load_annotations(self):
        if self.phase in ['train', 'val']:
            support_path = os.path.join(self.root, 'Intentonomy/support.json')
            with open(support_path, 'r') as f:
                support_annotations = json.load(f)
            
            # 准备数据和标签
            indices = list(range(len(support_annotations)))
            labels = [item['category'] for item in support_annotations]
            
            # 使用分层抽样进行训练集/验证集划分
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=0.125,  # 保持原来的划分比例
                stratify=labels,  # 按类别标签进行分层
                random_state=42  # 保持可重复性
            )
            
            # 根据phase选择合适的索引
            selected_indices = train_indices if self.phase == 'train' else val_indices
            
            # 获取选定的样本
            self.annotations = [support_annotations[i] for i in selected_indices]
            
            # 打印训练/验证集分布信息（仅在调试时使用）
            if False:  # 调试开关
                class_count = {}
                for item in self.annotations:
                    c = item['category']
                    class_count[c] = class_count.get(c, 0) + 1
                print(f"{self.phase} set class distribution: {class_count}")
                
        else:
            query_path = os.path.join(self.root, 'Intentonomy/query.json')
            with open(query_path, 'r') as f:
                self.annotations = json.load(f)

class EmoSetDataset(BaseDataset):
    """EmoSet数据集"""
    def __init__(self, root, processor, phase='train', img_size=224):
        self.classnames = ["amusement","anger","awe","contentment","disgust","excitement","fear","sadness"]
        super().__init__(root, processor, phase, img_size)
    
    def load_annotations(self):
        if self.phase in ['train', 'val']:
            support_path = os.path.join(self.root, 'EmoSet/support.json')
            with open(support_path, 'r') as f:
                support_annotations = json.load(f)
            
            # 准备数据和标签
            indices = list(range(len(support_annotations)))
            labels = [item['category'] for item in support_annotations]
            
            # 使用分层抽样进行训练集/验证集划分
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=0.125,  # 保持原来的划分比例
                stratify=labels,  # 按类别标签进行分层
                random_state=42  # 保持可重复性
            )
            
            # 根据phase选择合适的索引
            selected_indices = train_indices if self.phase == 'train' else val_indices
            
            # 获取选定的样本
            self.annotations = [support_annotations[i] for i in selected_indices]
            
        else:
            query_path = os.path.join(self.root, 'EmoSet/query.json')
            with open(query_path, 'r') as f:
                self.annotations = json.load(f)

class ArtPhotoDataset(BaseDataset):
    """ArtPhoto数据集"""
    def __init__(self, root, processor, phase='train', img_size=224):
        self.classnames = ["awe","contentment","sad","amusement","excitement","fear","anger","disgust"]
        super().__init__(root, processor, phase, img_size)
    
    def load_annotations(self):
        if self.phase in ['train', 'val']:
            support_path = os.path.join(self.root, 'ArtPhoto/support.json')
            with open(support_path, 'r') as f:
                support_annotations = json.load(f)
            
            # 准备数据和标签
            indices = list(range(len(support_annotations)))
            labels = [item['category'] for item in support_annotations]
            
            # 使用分层抽样进行训练集/验证集划分
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=0.125,  # 保持原来的划分比例
                stratify=labels,  # 按类别标签进行分层
                random_state=42  # 保持可重复性
            )
            
            # 根据phase选择合适的索引
            selected_indices = train_indices if self.phase == 'train' else val_indices
            
            # 获取选定的样本
            self.annotations = [support_annotations[i] for i in selected_indices]
            
        else:
            query_path = os.path.join(self.root, 'ArtPhoto/query.json')
            with open(query_path, 'r') as f:
                self.annotations = json.load(f)

class EmotionROIDataset(BaseDataset):
    """EmotionROI数据集"""
    def __init__(self, root, processor, phase='train', img_size=224):
        self.classnames = ["anger","disgust","fear","joy","sadness","surprise"]
        super().__init__(root, processor, phase, img_size)
    
    def load_annotations(self):
        if self.phase in ['train', 'val']:
            support_path = os.path.join(self.root, 'EmotionROI/support.json')
            with open(support_path, 'r') as f:
                support_annotations = json.load(f)
            
            # 准备数据和标签
            indices = list(range(len(support_annotations)))
            labels = [item['category'] for item in support_annotations]
            
            # 使用分层抽样进行训练集/验证集划分
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=0.125,  # 保持原来的划分比例
                stratify=labels,  # 按类别标签进行分层
                random_state=42  # 保持可重复性
            )
            
            # 根据phase选择合适的索引
            selected_indices = train_indices if self.phase == 'train' else val_indices
            
            # 获取选定的样本
            self.annotations = [support_annotations[i] for i in selected_indices]
            
        else:
            query_path = os.path.join(self.root, 'EmotionROI/query.json')
            with open(query_path, 'r') as f:
                self.annotations = json.load(f)

def get_dataset(dataset_name):
    """数据集工厂函数"""
    datasets = {
        'intentonomy': IntentonomyDataset,
        'emoset': EmoSetDataset,
        'artphoto': ArtPhotoDataset,
        'emotionroi': EmotionROIDataset
    }
    return datasets.get(dataset_name.lower())

class CLIPImageTuner(nn.Module):
    def __init__(self, num_classes, lora_rank=8, checkpoint_path=None):
        super().__init__()
        # 初始化CLIP模型
        try:
            config = CLIPConfig.from_pretrained("/data1/yq_log/IntenICL/huggingface/openai/clip-vit-base-patch32")
            self.clip = CLIPModel.from_pretrained("/data1/yq_log/IntenICL/huggingface/openai/clip-vit-base-patch32", config=config)
        except Exception as e:
            print(f"Error loading CLIP model: {str(e)}")
            raise
        
        # 冻结文本编码器
        for param in self.clip.text_model.parameters():
            param.requires_grad = False
            
        # 冻结视觉编码器的大部分层，只保留最后几层可训练
        vision_model = self.clip.vision_model
        
        # 冻结前面的层 (例如：embddings, 大部分encoder layers)
        trainable_layers = ['encoder.layers.11', 'pre_layrnorm', 'post_layernorm']  # 只保留最后一个transformer层和归一化层可训练
        
        for name, param in vision_model.named_parameters():
            param.requires_grad = any(trainable_layer in name for trainable_layer in trainable_layers)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["mlp.fc1", "mlp.fc2"],  # 只对最后层的MLP应用LoRA
            modules_to_save=["visual_projection"],
            lora_dropout=0.1,
            bias="none"
        )
        self.clip.vision_model = get_peft_model(vision_model, lora_config)
        
        # 添加视觉-情感语义对齐层
        self.vision_to_semantic = nn.Sequential(
            nn.Linear(self.clip.projection_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加dropout以增强泛化能力
            nn.Linear(512, 512),  # 增加一层以增强表达能力
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.clip.projection_dim),
            nn.LayerNorm(self.clip.projection_dim)
        )
        
        # 分类头
        self.classifier = nn.Linear(self.clip.projection_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # 加载检查点（如果有）
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path))

    def extract_features(self, pixel_values, merge_lora=True):
        """提取图像特征"""
        with torch.no_grad():
            if merge_lora and hasattr(self.clip.vision_model, 'merge_and_unload'):
                merged_model = self.clip.vision_model.merge_and_unload()
                original_vision_model = self.clip.vision_model
                self.clip.vision_model = merged_model
                image_features = self.clip.get_image_features(pixel_values=pixel_values)
                self.clip.vision_model = original_vision_model
                return image_features
            else:
                return self.clip.get_image_features(pixel_values=pixel_values)

    def forward(self, pixel_values):
        # 获取图像特征
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        
        # 将图像特征映射到情感语义空间
        aligned_features = self.vision_to_semantic(image_features)
        
        # 返回分类结果
        return self.classifier(aligned_features)
    
    def compute_similarity(self, pixel_values, text_inputs, temperature=0.07):
        """
        计算图像与文本的相似度
        Args:
            pixel_values: 图像输入
            text_inputs: 文本输入
            temperature: 温度系数
        Returns:
            similarity: 相似度分数
        """
        # 获取图像特征
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        # 将图像特征映射到情感语义空间
        aligned_features = self.vision_to_semantic(image_features)
        # 归一化特征
        aligned_features = F.normalize(aligned_features, p=2, dim=1)
        
        # 获取文本特征
        text_features = self.clip.get_text_features(**text_inputs)
        # 归一化特征
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度
        similarity = torch.matmul(aligned_features, text_features.t()) / temperature
        
        return similarity
    
    def zero_shot_classification(self, pixel_values, classnames, processor, templates=None, temperature=0.07):
        """
        零样本分类
        Args:
            pixel_values: 图像输入
            classnames: 类别名称列表
            processor: 文本处理器
            templates: 文本提示模板列表，默认使用5个通用模板
            temperature: 温度系数
        Returns:
            probs: 每个类别的概率
            preds: 预测的类别索引
        """
        if templates is None:
            templates = [
                "an image depicting {}",
                "an image about {}",
                "an image that shows {}",
                "an image with the emotion of {}",
                "an image that makes people feel {}"
            ]
        
        # 提取文本特征
        all_text_features = []
        
        with torch.no_grad():
            for classname in classnames:
                # 对每个类别应用多个提示模板
                texts = [template.format(classname) for template in templates]
                text_inputs = processor(
                    text=texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                # 提取文本特征并求平均
                text_features = self.clip.get_text_features(**text_inputs)
                mean_text_feature = F.normalize(text_features.mean(dim=0, keepdim=True), p=2, dim=1)
                all_text_features.append(mean_text_feature)
            
            # 合并所有文本特征
            class_text_features = torch.cat(all_text_features, dim=0)
            
            # 获取图像特征
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            # 将图像特征映射到情感语义空间
            aligned_features = self.vision_to_semantic(image_features)
            # 归一化特征
            aligned_features = F.normalize(aligned_features, p=2, dim=1)
            
            # 计算图像与各类别文本的相似度
            similarity = torch.matmul(aligned_features, class_text_features.t()) / temperature
            
            # 使用softmax将相似度转换为概率
            probs = F.softmax(similarity, dim=1)
            
            # 获取预测结果
            _, preds = torch.max(probs, dim=1)
            
            return probs, preds

    def extract_aligned_features(self, pixel_values, normalize=True):
        """
        提取图像特征并映射到情感语义空间
        Args:
            pixel_values: 图像输入
            normalize: 是否归一化特征
        Returns:
            aligned_features: 对齐后的特征
        """
        # 获取图像特征
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        # 将图像特征映射到情感语义空间
        aligned_features = self.vision_to_semantic(image_features)
        
        # 归一化特征（如果需要）
        if normalize:
            aligned_features = F.normalize(aligned_features, p=2, dim=1)
            
        return aligned_features

def evaluate_model(model, test_loader, device, classnames):
    """
    在测试集上评估模型
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        classnames: 类别名称列表
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    total_test = 0
    
    # 初始化混淆矩阵
    num_classes = len(classnames)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    
    # 每个类别的正确数和总数
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # 获取图像特征
            image_features = model.clip.get_image_features(inputs)
            # 将图像特征映射到情感语义空间
            aligned_features = model.vision_to_semantic(image_features)
            # 计算分类结果
            outputs = model.classifier(aligned_features)
            
            loss = F.cross_entropy(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            # 更新统计信息
            test_correct += (preds == labels).sum().item()
            total_test += labels.size(0)
            
            # 更新混淆矩阵
            for t, p in zip(labels, preds):
                confusion_matrix[t.item(), p.item()] += 1
            
            # 更新每个类别的统计信息
            for i in range(num_classes):
                mask = labels == i
                class_correct[i] += (preds[mask] == labels[mask]).sum()
                class_total[i] += mask.sum()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各种指标
    test_loss = test_loss / total_test
    test_acc = test_correct / total_test
    class_accs = (class_correct / class_total).cpu().numpy()
    
    # 计算每个类别的精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # 计算宏平均和加权平均指标
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 整理评估结果
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'class_accs': class_accs,
        'confusion_matrix': confusion_matrix.cpu().numpy(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    # 打印评估结果
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\nPer-class Results:")
    for i, name in enumerate(classnames):
        print(f"{name}:")
        print(f"  Accuracy: {class_accs[i]:.4f}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1: {f1[i]:.4f}")
    
    return results

def compute_feature_metrics(model, dataloader, device):
    """
    计算特征空间的质量指标，包括类内距离、类间距离和特征分离度
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        dict: 包含特征指标的字典
    """
    model.eval()
    all_features = []
    all_labels = []
    
    # 提取所有样本的特征
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing feature metrics"):
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # 提取对齐后的特征
            image_features = model.clip.get_image_features(inputs)
            aligned_features = model.vision_to_semantic(image_features)
            # 归一化特征
            features = F.normalize(aligned_features, p=2, dim=1)
            
            all_features.append(features)
            all_labels.append(labels)
    
    # 合并所有批次的特征和标签
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算类内距离和类间距离
    intra_class_dist = []
    inter_class_dist = []
    
    unique_labels = torch.unique(all_labels)
    for cls in unique_labels:
        # 当前类别的所有特征
        cls_features = all_features[all_labels == cls]
        # 其他类别的所有特征
        other_features = all_features[all_labels != cls]
        
        # 计算类内平均距离（使用余弦距离：1 - 余弦相似度）
        if len(cls_features) > 1:
            # 计算特征间的相似度矩阵
            sim_matrix = torch.mm(cls_features, cls_features.t())
            # 创建掩码排除自身相似度
            mask = 1.0 - torch.eye(len(cls_features), device=device)
            # 计算平均距离 (1 - 余弦相似度)
            intra_dist = 1.0 - (sim_matrix * mask).sum() / (mask.sum() + 1e-8)
            intra_class_dist.append(intra_dist)
        
        # 计算与其他类别的平均距离
        if len(cls_features) > 0 and len(other_features) > 0:
            # 计算与其他类别特征的相似度
            sim_matrix = torch.mm(cls_features, other_features.t())
            # 计算平均距离
            inter_dist = 1.0 - sim_matrix.mean()
            inter_class_dist.append(inter_dist)
    
    # 平均类内距离
    mean_intra_dist = torch.stack(intra_class_dist).mean().item()
    # 平均类间距离
    mean_inter_dist = torch.stack(inter_class_dist).mean().item()
    # 特征分离度 (越大越好)
    feature_separation = mean_inter_dist / (mean_intra_dist + 1e-8)
    
    return {
        'intra_class_dist': mean_intra_dist,
        'inter_class_dist': mean_inter_dist,
        'feature_separation': feature_separation
    }

def compute_text_image_alignment(model, dataloader, processor, classnames, device):
    """
    计算图像特征与文本特征的对齐程度，以及零样本分类准确率
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        processor: 文本处理器
        classnames: 类别名称列表
        device: 计算设备
    
    Returns:
        dict: 包含对齐指标的字典
    """
    model.eval()
    text_image_sims = []
    correct_matches = 0
    total = 0
    
    # 为每个类别创建多个文本提示
    text_templates = [
        "an image depicting {}",
        "an image about {}",
        "an image that shows {}",
        "an image with the emotion of {}",
        "an image that makes people feel {}"
    ]
    
    with torch.no_grad():
        # 获取所有类别的文本特征
        all_text_features = []
        for classname in classnames:
            # 对每个类别应用多个提示模板
            texts = [template.format(classname) for template in text_templates]
            text_inputs = processor(
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # 提取文本特征并求平均
            text_features = model.clip.get_text_features(**text_inputs)
            mean_text_feature = F.normalize(text_features.mean(dim=0, keepdim=True), p=2, dim=1)
            all_text_features.append(mean_text_feature)
        
        # 合并所有类别的文本特征
        text_features = torch.cat(all_text_features, dim=0)
        
        # 处理每个批次的图像
        for batch in tqdm(dataloader, desc="Computing text-image alignment"):
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # 提取图像特征并对齐
            image_features = model.clip.get_image_features(inputs)
            aligned_features = model.vision_to_semantic(image_features)
            # 归一化特征
            image_features_norm = F.normalize(aligned_features, p=2, dim=1)
            
            # 计算与所有文本的相似度
            similarities = torch.matmul(image_features_norm, text_features.t())
            
            # 计算与正确类别的相似度
            for i, label in enumerate(labels):
                text_image_sims.append(similarities[i, label].item())
            
            # 计算零样本分类准确率
            _, pred_labels = similarities.max(dim=1)
            correct_matches += (pred_labels == labels).sum().item()
            total += labels.size(0)
    
    return {
        'mean_text_image_sim': np.mean(text_image_sims),
        'zero_shot_acc': correct_matches / total
    }

def train_model(
    data_root,
    dataset_name,
    batch_size=64,
    lr=1e-4,
    epochs=20,
    lora_rank=8,
    grad_clip=1.0,
    checkpoint_dir="checkpoints",
    resume=None
):
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 获取数据集类
    dataset_cls = get_dataset(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 初始化数据处理器
    processor = CLIPProcessor.from_pretrained("/data1/yq_log/IntenICL/huggingface/openai/clip-vit-base-patch32")
    
    # 创建数据集
    train_dataset = dataset_cls(data_root, processor, 'train')
    val_dataset = dataset_cls(data_root, processor, 'val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    # 初始化模型
    model = CLIPImageTuner(
        num_classes=len(train_dataset.classnames), 
        lora_rank=lora_rank,
        checkpoint_path=resume
    ).to(device)
    
    # 为每个类别创建文本提示
    text_templates = [
        "an image depicting {}",
        "an image about {}",
        "an image that shows {}",
        "an image with the emotion of {}",
        "an image that makes people feel {}"
    ]
    
    all_text_features = []
    model.eval()  # 文本编码器处于评估模式
    
    with torch.no_grad():
        for classname in train_dataset.classnames:
            # 对每个类别应用多个提示模板
            texts = [template.format(classname) for template in text_templates]
            text_inputs = processor(
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # 提取文本特征并求平均
            text_features = model.clip.get_text_features(**text_inputs)
            mean_text_feature = F.normalize(text_features.mean(dim=0, keepdim=True), p=2, dim=1)
            all_text_features.append(mean_text_feature)
        
        # 合并所有文本特征
        class_text_features = torch.cat(all_text_features, dim=0)
    
    # 优化器配置
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.05
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # 添加指标记录
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'val_accs': [],
        'best_acc': 0.0,
        'best_epoch': 0,
        'best_combined_score': 0.0,  # 添加综合分数记录
        'feature_metrics': [],
        'alignment_metrics': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # 获取图像特征
                image_features = model.clip.get_image_features(inputs)
                # 将图像特征映射到情感语义空间
                aligned_features = model.vision_to_semantic(image_features)
                # 归一化特征
                aligned_features = F.normalize(aligned_features, p=2, dim=1)
                
                # 计算分类损失
                logits = model.classifier(aligned_features)
                ce_loss = F.cross_entropy(logits, labels)
                
                # 计算对比损失（将图像特征与对应类别的文本特征对齐）
                image_text_similarity = torch.matmul(aligned_features, class_text_features.t())
                
                # 构建标签索引，用于选择正样本
                positive_indices = labels.view(-1, 1)
                
                # 带温度系数的对比损失
                temperature = 0.07
                image_text_similarity = image_text_similarity / temperature
                
                # 交叉熵对比损失（InfoNCE）
                contrast_loss = F.cross_entropy(image_text_similarity, labels)
                
                # 添加全局对比学习损失 - 鼓励批次内不同样本的特征区分性
                batch_size = aligned_features.size(0)
                if batch_size > 1:  # 确保批次大小足够进行对比
                    # 计算批次内样本间的相似度矩阵
                    feature_sim = torch.matmul(aligned_features, aligned_features.t()) / temperature
                    # 创建对角掩码，排除自身相似度
                    mask = torch.eye(batch_size, device=device)
                    # 创建同类别掩码
                    same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
                    
                    # 正样本: 同类别的其他样本
                    pos_mask = same_label_mask * (1 - mask)
                    # 负样本: 不同类别的样本
                    neg_mask = ~same_label_mask
                    
                    # 对每个样本，计算与同类样本的相似度平均值作为正例
                    pos_sim = (feature_sim * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
                    
                    # 提取负样本相似度并计算对比损失
                    feature_sim_exp = torch.exp(feature_sim)
                    # 正例相似度
                    pos_sim_exp = torch.exp(pos_sim)
                    # 负例相似度之和 (排除自己)
                    neg_sim_exp_sum = (feature_sim_exp * neg_mask).sum(dim=1)
                    
                    # 对比损失 = -log(正例相似度 / (正例相似度 + 负例相似度之和))
                    feature_contrast_loss = -torch.log(pos_sim_exp / (pos_sim_exp + neg_sim_exp_sum + 1e-8))
                    feature_contrast_loss = feature_contrast_loss.mean()
                else:
                    feature_contrast_loss = torch.tensor(0.0, device=device)
                
                # 总损失 = 主要使用对比损失，分类损失作为辅助
                # loss = ce_loss + contrast_loss  # 原始权重
                # loss = 0.1 * ce_loss + 0.8 * contrast_loss + 0.2 * feature_contrast_loss  # 更强调特征对齐
                loss = 0.8 * contrast_loss + 0.2 * feature_contrast_loss  # 更强调特征对齐
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # 记录梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 计算训练准确率
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({
                'loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'cont_loss': contrast_loss.item(),
                'feat_cont_loss': feature_contrast_loss.item() if batch_size > 1 else 0.0,
                'acc': train_correct / total_train,
                'grad_norm': grad_norm.item()
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        class_correct = torch.zeros(len(train_dataset.classnames), device=device)
        class_total = torch.zeros(len(train_dataset.classnames), device=device)
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['pixel_values'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                # 获取图像特征
                image_features = model.clip.get_image_features(inputs)
                # 将图像特征映射到情感语义空间
                aligned_features = model.vision_to_semantic(image_features)
                # 计算分类结果
                outputs = model.classifier(aligned_features)
                
                # 计算分类损失
                loss = F.cross_entropy(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)
                
                # 计算每个类别的准确率
                for i in range(len(train_dataset.classnames)):
                    mask = labels == i
                    class_correct[i] += (preds[mask] == labels[mask]).sum()
                    class_total[i] += mask.sum()
        
        # 计算指标
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / total_train
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / total_val
        class_accs = (class_correct / class_total).cpu().numpy()
        
        # 每5个epoch计算一次特征质量指标（可以根据需求调整频率）
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print("\nComputing feature quality metrics...")
            feature_metrics = compute_feature_metrics(model, val_loader, device)
            alignment_metrics = compute_text_image_alignment(model, val_loader, processor, train_dataset.classnames, device)
            
            # 记录特征指标
            metrics['feature_metrics'].append(feature_metrics)
            metrics['alignment_metrics'].append(alignment_metrics)
            
            # 打印特征质量指标
            print(f"Feature metrics:")
            print(f"  Intra-class distance: {feature_metrics['intra_class_dist']:.4f}")
            print(f"  Inter-class distance: {feature_metrics['inter_class_dist']:.4f}")
            print(f"  Feature separation: {feature_metrics['feature_separation']:.4f}")
            print(f"Alignment metrics:")
            print(f"  Mean text-image similarity: {alignment_metrics['mean_text_image_sim']:.4f}")
            print(f"  Zero-shot accuracy: {alignment_metrics['zero_shot_acc']:.4f}")
            
            # 计算综合得分
            combined_score = (0.4 * val_acc + 
                             0.3 * feature_metrics['feature_separation'] + 
                             0.3 * alignment_metrics['mean_text_image_sim'])
            
            print(f"Combined quality score: {combined_score:.4f}")
            
            # 使用综合指标选择最佳模型
            if combined_score > metrics['best_combined_score']:
                metrics['best_combined_score'] = combined_score
                metrics['best_epoch'] = epoch + 1
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': val_acc,
                    'best_combined_score': combined_score,
                    'metrics': metrics,
                    'feature_metrics': feature_metrics,
                    'alignment_metrics': alignment_metrics
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"best_model_{dataset_name}.pth"))
                print(f"New best model saved with combined score: {combined_score:.4f}")
        else:
            # 在不计算特征指标的epoch，仍然使用分类准确率保存潜在的最佳模型
            if val_acc > metrics['best_acc']:
                metrics['best_acc'] = val_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': val_acc,
                    'metrics': metrics
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"best_acc_model_{dataset_name}.pth"))
                print(f"New best accuracy model saved: {val_acc:.4f}")
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 记录指标
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['val_accs'].append(val_acc)
        
        # 打印详细信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # 打印每个类别的准确率
        print("\nPer-class Accuracy:")
        for i, (name, acc) in enumerate(zip(train_dataset.classnames, class_accs)):
            print(f"{name}: {acc:.4f}")
        print(f"\nBest Accuracy: {metrics['best_acc']:.4f} at epoch {metrics['best_epoch']}")
    
    # 训练完成后进行测试
    print("\nEvaluating on test set...")
    test_dataset = dataset_cls(data_root, processor, 'test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    test_results = evaluate_model(model, test_loader, device, train_dataset.classnames)
    
    # 保存测试结果
    results_path = os.path.join(checkpoint_dir, f"test_results_{dataset_name}.json")
    with open(results_path, 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        serializable_results = {
            k: v.tolist() if isinstance(v, (np.ndarray, np.float32, np.float64)) else v
            for k, v in test_results.items()
        }
        json.dump(serializable_results, f, indent=4)
    
    return model, metrics, test_results

def inference(
    image_path,
    model_path,
    dataset_name,
    img_size=224,
    use_classifier=False,  # 默认使用相似度计算而非分类头
    custom_classnames=None,
    custom_templates=None
):
    """
    使用训练好的模型对单张图像进行推理
    Args:
        image_path: 图像路径
        model_path: 模型路径
        dataset_name: 数据集名称
        img_size: 图像大小
        use_classifier: 是否使用分类头（默认False，使用相似度计算）
        custom_classnames: 自定义类别名称列表
        custom_templates: 自定义文本模板列表
    Returns:
        pred_class: 预测的类别
        confidence: 置信度
    """
    # 获取数据集类，加载类别名称
    dataset_cls = get_dataset(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 实例化一个数据集对象，只为了获取类别名称
    processor = CLIPProcessor.from_pretrained("/data1/yq_log/IntenICL/huggingface/openai/clip-vit-base-patch32")
    dummy_dataset = dataset_cls("dummy", processor)
    classnames = custom_classnames or dummy_dataset.classnames
    
    # 加载模型
    model = CLIPImageTuner(num_classes=len(classnames)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 加载并预处理图像
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)
        
        # 处理图像
        processed = processor(
            images=image_tensor,
            return_tensors="pt",
            do_rescale=False
        )
        pixel_values = processed['pixel_values'].to(device)
        
        with torch.no_grad():
            if not use_classifier:
                # 使用特征相似度计算（默认方式）
                probs, preds = model.zero_shot_classification(
                    pixel_values, 
                    classnames, 
                    processor,
                    templates=custom_templates
                )
                pred_idx = preds.item()
                confidence = probs[0, pred_idx].item()
            else:
                # 使用标准分类头
                outputs = model(pixel_values)
                probs = F.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                confidence = confidence.item()
                pred_idx = pred_idx.item()
            
            pred_class = classnames[pred_idx]
        
        return pred_class, confidence
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None, 0.0

# 添加一个提取图像特征向量的函数
def extract_image_features(model, images, processor, normalize=True):
    """
    使用训练好的模型提取图像的特征向量
    
    Args:
        model: 训练好的模型
        images: 图像列表(PIL.Image.Image)或图像路径列表
        processor: CLIP处理器
        normalize: 是否归一化特征向量
    
    Returns:
        numpy.ndarray: 图像特征向量
    """
    model.eval()
    batch_size = 32
    all_features = []
    
    # 处理图像列表或路径列表
    image_batch = []
    for img in images:
        if isinstance(img, str):
            # 如果是路径，则加载图像
            try:
                img = Image.open(img).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img}: {str(e)}")
                continue
        
        # 添加到批次
        image_batch.append(img)
        
        # 当批次足够大或处理到最后一个图像时，进行批处理
        if len(image_batch) >= batch_size or img == images[-1]:
            # 处理批次
            processed = processor(
                images=image_batch,
                return_tensors="pt",
                padding=True
            )
            
            pixel_values = processed['pixel_values'].to(device)
            
            # 提取特征
            with torch.no_grad():
                # 获取图像特征
                image_features = model.clip.get_image_features(pixel_values)
                # 将图像特征映射到情感语义空间
                aligned_features = model.vision_to_semantic(image_features)
                
                if normalize:
                    # 归一化特征
                    aligned_features = F.normalize(aligned_features, p=2, dim=1)
                
                all_features.append(aligned_features.cpu().numpy())
            
            # 清空批次
            image_batch = []
    
    if all_features:
        return np.vstack(all_features)
    else:
        return np.array([])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # 训练参数
    parser.add_argument('--mode', choices=['train', 'inference'], default='train',
                      help='运行模式: 训练模型或进行推理')
    parser.add_argument('--data_root', default='/data1/yq_log/IntenICL/DataSet', type=str, help='数据集根目录')
    parser.add_argument('--dataset', default='Intentonomy', type=str, 
                      choices=['Intentonomy', 'EmoSet', 'ArtPhoto', 'EmotionROI'], 
                      help='数据集名称')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--checkpoint_dir', default='/data1/yq_log/IntenICL/ICL_log/intCLIP/checkpoints', type=str)
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    # 推理参数
    parser.add_argument('--image_path', type=str, help='要预测的图像路径')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--use_classifier', action='store_true', help='使用分类头而不是相似度计算')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, metrics, test_results = train_model(
            data_root=args.data_root,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            lora_rank=args.lora_rank,
            grad_clip=args.grad_clip,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume
        )
    else:
        # 推理模式
        if not args.image_path or not args.model_path:
            parser.error("推理模式需要指定 --image_path 和 --model_path")
        
        pred_class, confidence = inference(
            image_path=args.image_path,
            model_path=args.model_path,
            dataset_name=args.dataset,
            use_classifier=args.use_classifier
        )
        
        print(f"预测类别: {pred_class}")
        print(f"置信度: {confidence:.4f}")