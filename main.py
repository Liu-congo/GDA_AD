from src.Trainer import run_experiment
from src.Transforms import Mixup, RandomEdgeDrop, AdaptiveNodeNoise

config = {
    'dataset_cfg': {
        'data_root': 'data',
        'threshold': 0,
        'node_feature_type': 'raw',
    },
    'dataloader_cfg': {
        'batch_size': 16,
        'transforms': {
            'train':  [RandomEdgeDrop(drop_rate=0.2), 
                       AdaptiveNodeNoise(noise_scale=0.2), 
                       Mixup(samples_num = 1000, 
                             mode='beta', 
                             distrib_params={'alpha':0.5, 'beta':0.5})],
            'eval': None
        }
    },
    'model_cfg': {
        'model_params': {
            'model_type': 'GCN',
            'in_channels': 90,
            'hidden_channels': 64,
            'out_channels': 16,
            'num_layers': 3
        }
    },
    'optimizer_cfg': {
        'optimizer_type': 'Adam',
        'lr': 0.001,
        'weight_decay': 0.005
    },
    'trainer_cfg': {
        'epochs': 20,
        'save_path': 'best_model.pth',
        "log_dir": "experiments"
    }
}

# 运行实验
model = run_experiment(config)