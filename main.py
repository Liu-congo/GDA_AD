from src.Trainer import run_experiment
from src.Transforms import Mixup, RandomEdgeDrop, AdaptiveNodeNoise, NodeMask

config = {
    'dataset_cfg': {
        'data_root': 'data',
        'threshold': 0,
        'node_feature_type': 'raw',
    },
    'dataloader_cfg': {
        'batch_size': 16,
        'transforms': {
            # 'train':  [Mixup(samples_num = 1000, 
            #                  mode='beta', 
            #                  distrib_params={'alpha':0.5, 'beta':0.5})], 
            # 'train':[NodeMask(),
            #          AdaptiveNodeNoise(),
            #          RandomEdgeDrop()],
            'train':[Mixup(samples_num = 1000, 
                             mode='uniform', 
                             distrib_params={'alpha':0.5, 'beta':0.5}),
                     NodeMask(),
                     AdaptiveNodeNoise(),
                     RandomEdgeDrop()],
            # 'train': None,
            'eval': None
        }
    },
    'model_cfg': {
        'model_params': {
            'model_type': 'GAT',
            'in_channels': 90,
            'hidden_channels': 8,
            'out_channels': 16,
            'num_layers': 2
        }
    },
    'optimizer_cfg': {
        'optimizer_type': 'Adam',
        'lr': 0.001,
        'weight_decay': 0.005
    },
    'trainer_cfg': {
        'epochs': 15,
        'save_path': 'best_model.pth',
        "log_dir": "GAT_Inter_Intra"
    }
}

# 运行实验
model = run_experiment(config)