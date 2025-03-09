from src.Datasets import AlzheimerDataset
from src.DataLoader import GraphDataLoader
from src.Transforms import RandomEdgeDrop, AdaptiveNodeNoise, Mixup
import torch

# 示例transform配置
train_transforms = [
    # T.RandomNodeSplit(num_val=0.1, num_test=0.1),  # 节点分类任务需要
    RandomEdgeDrop(drop_rate=0.2),  # 随机丢弃30%的边
    # EdgeNoise(noise_scale=0.1),    # 添加10%比例的高斯噪声
    # NodeMask(mask_ratio=0.2, mask_strategy="mean"), # 随机掩盖20%的节点并该图节点均值填充
    # NodeNoise(noise_scale=0.2, mode="uniform"),     # 随机添加噪声
    AdaptiveNodeNoise(noise_scale=0.2),
    Mixup(samples_num = 100, 
          mode='beta', 
          distrib_params={'alpha':0.5, 
                          'beta':0.5})
    # T.AddSelfLoops(),
    # T.NormalizeFeatures()
]
dataset = AlzheimerDataset(root='data',
                             threshold=0,
                             node_feature_type='strength')
dataset.process()
# 初始化数据加载器
loader = GraphDataLoader(
    dataset=dataset,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    batch_size=10,
    seed=2023,
    train_transforms=train_transforms,
)

# 统计信息输出
def print_stats(loader, name):
    counts = torch.cat([batch.y.argmax(dim = 1) for batch in loader]).unique(return_counts=True)[1]
    print(f"{name}: AD={counts[1]}, CN={counts[0]}, 总样本={sum(counts)}")

print_stats(loader.train, "训练集")
print_stats(loader.val, "验证集")
print_stats(loader.test, "测试集")

# 检查数据维度
sample = next(iter(loader.train))
print(sample.y)
print("样本特征维度:", sample.x.shape)
print("边索引维度:", sample.edge_index.shape)
print("标签分布:", sample.y.argmax(dim = 1).unique(return_counts=True))