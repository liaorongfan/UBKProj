import torch
import torch.nn as nn
from .build import NETWORK_REGISTRY


class MLP(nn.Module):
    def __init__(self, num_cls=5):
        super().__init__()
        self.inp_layer = nn.Linear(245, 256)
        self.out_layer = nn.Linear(256, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inp_layer(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x


@NETWORK_REGISTRY.register()
def mlp_model(cfg=None):
    model = MLP()
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model