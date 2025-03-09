import csv
from datetime import datetime
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.Model import Classifier
from src.Datasets import AlzheimerDataset
from src.DataLoader import GraphDataLoader


class ExperimentLogger:
    def __init__(self, config, log_dir="experiments"):
        """
        实验日志记录系统
        Args:
            config (dict): 实验配置字典
            log_dir (str): 所有实验的根目录
        """
        # 生成唯一实验ID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_id = f"exp_{timestamp}"
        
        # 创建实验目录结构
        self.exp_path = os.path.join(log_dir, self.exp_id)
        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(os.path.join(self.exp_path, "checkpoints"), exist_ok=True)
        
        # 保存完整配置
        self._save_config(config)
        
        # 初始化训练日志
        self.train_log_path = os.path.join(self.exp_path, "training_log.csv")
        self._init_csv_logger()
        
    def _save_config(self, config):
        """保存实验配置到JSON文件"""
        config_path = os.path.join(self.exp_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def _init_csv_logger(self):
        """初始化CSV训练日志"""
        header = [
            "epoch", 
            "train_loss", "train_acc", "train_pre","train_recall","train_f1", 
            "val_loss", "val_acc", "val_pre","val_recall","val_f1",
            "lr", "time_elapsed"
        ]
        with open(self.train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
    def log_epoch(self, epoch, train_metrics, val_metrics, lr, time_elapsed):
        """记录每个epoch的训练指标"""
        row = [
            epoch, 
            train_metrics["loss"], 
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
            val_metrics["loss"], 
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            lr,
            time_elapsed
        ]
        
        with open(self.train_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        
        filename = "checkpoint.pth" if not is_best else "best_model.pth"
        save_path = os.path.join(self.exp_path, "checkpoints", filename)
        torch.save(checkpoint, save_path)
        
    def get_log_dir(self):
        """获取实验目录路径"""
        return self.exp_path
    
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data in tqdm(loader, desc="Training"):
        # 迁移到设备
        data = data.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        out = model(data)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, data.y.argmax(dim=1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        target = data.y.argmax(dim = 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    # 计算epoch指标
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, precision, recall, f1

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        out = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, data.y.argmax(dim = 1))
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        target = data.y.argmax(dim = 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, precision, recall, f1

def get_transform_config(transform):
    if transform is None:
        return None
    
    config = []
    for t in transform:
        # 获取transform名称和参数
        transform_info = {
            "name": t.__class__.__name__,
            "params": vars(t)  # 获取对象属性字典
        }
        config.append(transform_info)
    return config

def run_experiment(config):
    dataset_cfg = config['dataset_cfg']
    dataloader_cfg = config['dataloader_cfg']
    model_cfg = config['model_cfg']
    optimizer_cfg = config['optimizer_cfg']
    trainer_cfg = config['trainer_cfg']

    logger = ExperimentLogger({
        **{k: v for k, v in config.items() if k != 'dataloader_cfg' }
    }, log_dir=trainer_cfg.get("log_dir", "experiments"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AlzheimerDataset(root=dataset_cfg['data_root'],
                             threshold=dataset_cfg['threshold'],
                             node_feature_type=dataset_cfg['node_feature_type'])
    dataset.process()
    # 初始化数据加载器
    loader = GraphDataLoader(
        dataset=dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        batch_size=dataloader_cfg['batch_size'],
        seed=2023,
        train_transforms=dataloader_cfg['transforms']['train'],
    )
    transforms_log = {
        "train_transforms": get_transform_config(dataloader_cfg['transforms']['train']),
        "eval_transforms": get_transform_config(dataloader_cfg['transforms']['eval'])
    }
    with open(os.path.join(logger.get_log_dir(), "transforms.json"), 'w') as f:
        json.dump(transforms_log, f, indent=4, default=str)

    model = Classifier(
        model_params = model_cfg["model_params"]
    ).to(device)

    optimizer_type = optimizer_cfg['optimizer_type']
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_cfg['lr'],
            weight_decay=optimizer_cfg.get('weight_decay', 0.01)
        )
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_cfg['lr'],
            weight_decay=optimizer_cfg.get('weight_decay', 0.01),
            momentum=optimizer_cfg.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # 7. 训练循环参数修改
    best_val_acc = 0
    start_time = time.time()
    for epoch in range(trainer_cfg['epochs']):
        epoch_start = time.time()
        train_loss, train_acc, train_pre, train_recall, train_f1 = train(model, loader.train, optimizer, device)
        val_loss, val_acc, val_pre, val_recall, val_f1 = evaluate(model, loader.val, device)
        
        # 计算耗时
        epoch_duration = time.time() - epoch_start
        total_duration = time.time() - start_time
        
        # 记录日志
        logger.log_epoch(
            epoch=epoch+1,
            train_metrics={"loss": train_loss, 
                           "accuracy": train_acc,
                           "precision": train_pre, 
                           "recall": train_recall,
                           "f1": train_f1},
            val_metrics={"loss": val_loss, 
                         "accuracy": val_acc,
                         "precision": val_pre,
                         "recall": val_recall,
                         "f1": val_f1},
            lr=optimizer.param_groups[0]['lr'],
            time_elapsed=total_duration
        )
        
        # 保存checkpoint
        logger.save_checkpoint(model, optimizer, epoch+1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.save_checkpoint(model, optimizer, epoch+1, is_best=True)
        
        # 打印日志
        print(f"Epoch {epoch+1:03d}/{trainer_cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Pre: {train_pre:.4f} Recall: {train_recall:.4f} F1: {train_f1:.4f}| "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Pre: {val_pre:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f}| "
              f"Time: {epoch_duration:.1f}s")

    # 最终测试
    model.load_state_dict(torch.load(os.path.join(logger.get_log_dir(), "checkpoints/best_model.pth"))["state_dict"])
    test_loss, test_acc, test_pre, test_recall, test_f1 = evaluate(model, loader.test, device)
    
    # 保存最终结果
    final_results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_pre,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "training_time": time.time() - start_time
    }
    with open(os.path.join(logger.get_log_dir(), "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nFinal Test Performance: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Pre: {test_pre:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    return model
