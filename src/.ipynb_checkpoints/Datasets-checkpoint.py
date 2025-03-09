from tkinter import NO
import torch
from torch_geometric.data import Data, Dataset
import os
import numpy as np
from typing import Optional, Callable


class AlzheimerDataset(Dataset):
    def __init__(self, root: str, 
                 threshold: float = 0.0,
                 node_feature_type: str = 'identity',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """
        Args:
            root (str): 数据根目录
            threshold (float): 边权重阈值，绝对值小于该值的边将被过滤
            node_feature_type (str): 节点特征类型，可选 'identity' 或 'strength' 或 'raw'
            transform (callable, optional): 动态数据变换
            pre_transform (callable, optional): 数据预处理的函数
        """
        self.threshold = threshold
        self.node_feature_type = node_feature_type
        self.label_dict = {'AD': [1., 0.], 'CN': [0., 1.]}
        self.processed_data = []
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        ad_dir = os.path.join(self.raw_dir, 'AD')
        cn_dir = os.path.join(self.raw_dir, 'CN')
        
        ad_files = [os.path.join('AD', f) for f in os.listdir(ad_dir) if f.endswith('.txt')]
        cn_files = [os.path.join('CN', f) for f in os.listdir(cn_dir) if f.endswith('.txt')]
        return ad_files + cn_files

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

    def download(self):
        pass  # 数据已经存在

    def process(self):
        for idx, raw_path in enumerate(self.raw_file_names):
            # 获取标签
            label_str = raw_path.split(os.sep)[0]
            label = self.label_dict[label_str]

            # 加载矩阵
            matrix = np.loadtxt(os.path.join(self.raw_dir, raw_path))
            matrix = torch.tensor(matrix, dtype=torch.float)

            # 生成边信息
            # 静态阈值过滤
            mask = (matrix.abs() > self.threshold)
            rows, cols = torch.where(mask)
            edge_index = torch.stack([rows, cols], dim=0)  # (2, E)

            # 生成节点特征
            if self.node_feature_type == 'identity':
                x = torch.eye(matrix.size(0), dtype=torch.float)  # (90, 90)
            elif self.node_feature_type == 'strength':
                strength = matrix.sum(dim=1)  # (90,)
                x = strength.view(-1, 1)        # (90, 1)
            elif self.node_feature_type == 'raw':
                x = matrix                      # (90, 90)
            else:
                raise ValueError(f"Unsupported node feature type: {self.node_feature_type}")

            # 创建Data对象
            data = Data(x=x, 
                        edge_index=edge_index,
                        y=torch.tensor([label], dtype=torch.long))

            # 保存处理后的数据
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            self.processed_data.append(data)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)


