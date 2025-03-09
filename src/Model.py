import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, global_mean_pool, MLP

class Classifier(nn.Module):
    def __init__(self, 
                 model_params={
                    'model_type': 'GCN',
                    'in_channels': 90,
                    'hidden_channels': 64,
                    'out_channels': 16,
                    'num_layers': 3
                }):
        """
        GCN图分类模型
        Args:
            model_params: GCN所需超参
        """
        super().__init__()
        self.model_type = model_params['model_type']
        self.model_params = {k: v for k, v in model_params.items() if k != 'model_type' }
        if self.model_type == 'GCN':
            self.gcn = GCN(**self.model_params)
            self.lin = nn.Linear(self.gcn.out_channels, 2)
        elif self.model_type == 'MLP':
            self.mlp = MLP(**self.model_params)
        else:
            raise Exception("model to be implemented")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.model_type == 'GCN':
            x = self.gcn(x=x, edge_index=edge_index, batch=batch)
            x = global_mean_pool(x, batch) 
            x = self.lin(x)
            return x
        elif self.model_type == 'MLP':
            x = self.mlp(x=x, batch=batch)
            return x
        else:
            raise Exception("model to be implemented")



