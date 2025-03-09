from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np


class StratifiedSplit:
    """分层抽样划分器"""
    def __init__(self, labels, ratios=(0.6, 0.2, 0.2), seed=42):
        self.ratios = ratios
        self.seed = seed
        
        # 获取各类别索引
        unique_labels = np.unique(labels)
        self.class_indices = {
            lbl: np.where(labels == lbl)[0] for lbl in unique_labels
        }

    def split(self):
        """返回划分后的索引列表"""
        splits = []
        rng = np.random.default_rng(self.seed)
        
        for ratio in self.ratios:
            split_indices = []
            for lbl, indices in self.class_indices.items():
                n_total = len(indices)
                n_split = int(ratio * n_total)
                
                # 随机选择该类的样本
                selected = rng.choice(indices, n_split, replace=False)
                split_indices.extend(selected)
                
                # 更新剩余可用索引
                self.class_indices[lbl] = np.setdiff1d(indices, selected)
                
            splits.append(split_indices)
        return splits

class GraphDataLoader:
    def __init__(self, 
                 dataset=None,
                 train_ratio=0.6,
                 val_ratio=0.2,
                 test_ratio=0.2,
                 batch_size=32,
                 seed=42,
                 train_transforms=None,
                 eval_transforms=None,
                 ):
        """
        参数说明：
        - *_ratio: 各数据集的划分比例
        - batch_size: 批大小
        - seed: 随机种子
        - train_transforms: 训练集的数据增强
        - eval_transforms: 验证/测试集的转换
        - dataset: 传入的数据集
        """
        self.full_dataset = dataset
        labels = [data.y.argmax() for data in self.full_dataset]
        splitter = StratifiedSplit(labels, 
                                 ratios=(train_ratio, val_ratio, test_ratio),
                                 seed=seed)
        train_idx, val_idx, test_idx = splitter.split()
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.train_set = self.apply_transform([self.full_dataset[idx] for idx in train_idx], train_transforms)
        self.val_set = self.apply_transform([self.full_dataset[idx] for idx in val_idx], eval_transforms)
        self.test_set = self.apply_transform([self.full_dataset[idx] for idx in test_idx], eval_transforms)
        
        self.batch_size = batch_size
        self.train_loader = self.create_loader(self.train_set, shuffle=True)
        self.val_loader = self.create_loader(self.val_set)
        self.test_loader = self.create_loader(self.test_set)

    def apply_transform(self, subset, transforms):
        # cur_data = [subset.dataset[idx] for idx in subset.indices]
        cur_data = subset
        cur_data_cp = subset
        if transforms == None:
            return cur_data
        for transform in transforms:
            cur_data = transform(cur_data) + cur_data_cp
        return cur_data

    def create_loader(self, dataset, shuffle=False):
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            follow_batch=[]  # 不需要特殊跟踪
        )
    @property
    def train(self):
        return self.train_loader
    
    @property
    def val(self):
        return self.val_loader
    
    @property
    def test(self):
        return self.test_loader
