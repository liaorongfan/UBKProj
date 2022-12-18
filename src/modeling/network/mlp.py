import torch
import torch.nn as nn
from .build import NETWORK_REGISTRY


class MLP(nn.Module):
    def __init__(self,input_dim=194, output_dim=3):
        super().__init__()
        self.inp_layer = nn.Linear(input_dim, 256)
        self.mid_layer = nn.Linear(256, 512)
        self.mid_relu = nn.ReLU()
        self.out_layer = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inp_layer(x)
        x = self.relu(x)
        x = self.mid_relu(self.mid_layer(x))
        x = self.out_layer(x)
        return x


@NETWORK_REGISTRY.register()
def mlp_model(cfg=None):
    model = MLP()
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
