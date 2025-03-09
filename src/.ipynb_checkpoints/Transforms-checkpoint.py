from ast import List
from tkinter import NO
from torch_geometric.data import Data
import torch
import random

class RandomEdgeDrop:
    def __init__(self, drop_rate=0.2):
        """
        随机丢弃边的数据增强
        Args:
            drop_rate (float): 边的丢弃概率 (0.0-1.0)
        """
        self.drop_rate = drop_rate

    def __call__(self, data_list: List) -> Data:
        for data in data_list:
            # 随机选择保留的边
            num_edges = data.edge_index.shape[1]
            keep_mask = torch.rand(num_edges) > self.drop_rate
            data.edge_index = data.edge_index[:, keep_mask]
        return data_list
    
    def __repr__(self):
        return f'{self.__class__.__name__}(drop_rate={self.drop_rate})'
    
class NodeNoise:
    def __init__(self, noise_scale=0.1, mode='gaussian'):
        """
        添加节点特征噪声
        Args:
            noise_scale (float): 噪声强度
            mode (str): 噪声类型 ('gaussian'/'uniform')
        """
        self.noise_scale = noise_scale
        self.mode = mode

    def __call__(self, data_list: List) -> Data:
        for data in data_list:
            if self.mode == 'gaussian':
                noise = torch.randn_like(data.x) * self.noise_scale
            elif self.mode == 'uniform':
                noise = (torch.rand_like(data.x) * 2 - 1) * self.noise_scale
            else:
                raise ValueError("Unsupported noise mode")

            # 保留原始特征范围
            data.x = torch.clamp(data.x + noise, 
                            data.x.min(), 
                            data.x.max())
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.noise_scale}, mode={self.mode})'
    

class NodeMask:
    def __init__(self, mask_ratio=0.3, mask_strategy='zero'):
        """
        节点特征掩码增强
        Args:
            mask_ratio (float): 掩码比例 (0.0-1.0)
            mask_strategy (str): 掩码策略 
                'zero' - 用0填充
                'mean' - 用特征均值填充
                'random' - 用随机值填充
        """
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy

    def __call__(self, data_list: List) -> Data:
        for data in data_list:
            # 生成掩码
            mask = torch.rand(data.x.size(0)) < self.mask_ratio
            
            if self.mask_strategy == 'zero':
                data.x[mask] = 0
            elif self.mask_strategy == 'mean':
                data.x[mask] = data.x.mean(dim=0)
            elif self.mask_strategy == 'random':
                data.x[mask] = torch.rand(data.x.size(1))
            else:
                raise ValueError("Unknown mask strategy")

        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(ratio={self.mask_ratio}, strategy={self.mask_strategy})'
    
class AdaptiveNodeNoise(NodeNoise):
    def __call__(self, data_list: List) -> Data:
        std = torch.cat([data.x for data in data_list], dim=0).std(dim=0)
        for data in data_list:
            # 根据特征标准差自动调整噪声幅度
            noise = torch.randn_like(data.x) * std * self.noise_scale
            data.x += noise
        return data_list
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
class Mixup:
    def __init__(self, samples_num, mode='beta', distrib_params={'alpha':0.5, 'beta':0.5}):
        """
        Mixup数据混合增强
        Args:
            samples_num (int): Mixup增强生成的样本数量
            mode (str): Mixup混合采用的分布
                'beta' - Beta分布
                'uniform' - 均匀分布
            distrib_parmas (dict): 所用分布的超参
        """
        self.samples_num = samples_num
        self.mode = mode
        self.distrib_params = distrib_params

    def __call__(self, data_list: List) -> Data:
        total_data_num = len(data_list)
        idx1, idx2 = [random.randint(0, total_data_num - 1) for _ in range(self.samples_num)], [random.randint(0, total_data_num - 1) for _ in range(self.samples_num)]
        prob = None
        if self.mode == "beta":
            prob = [random.betavariate(**self.distrib_params) for _ in range(self.samples_num)]
        elif self.mode == "uniform":
            prob = [random.uniform(0, 1) for _ in range(self.samples_num)]
        else:
            raise ValueError("Unknown Distribution")
        output_data_list = []
        for i in range(self.samples_num):
            cur_x = data_list[idx1[i]].x * (prob[i]) + data_list[idx2[i]].x * (1 - prob[i])
            cur_edge_index = torch.concat([data_list[idx1[i]].edge_index[:, torch.rand(data_list[idx1[i]].edge_index.shape[1]) < prob[i]],
                                           data_list[idx2[i]].edge_index[:, torch.rand(data_list[idx2[i]].edge_index.shape[1]) < 1 - prob[i]]],dim=1)
            cur_y = data_list[idx1[i]].y * (prob[i]) + data_list[idx2[i]].y * (1 - prob[i])
            data = Data(x=cur_x, 
                        edge_index=cur_edge_index,
                        y=cur_y)
            output_data_list.append(data)
        return output_data_list
    
    def __repr__(self):
        return f'{self.__class__.__name__}(sample_num={self.samples_num}, distribution={self.mode}, distrib_params={self.distrib_params})'
        
    
    
