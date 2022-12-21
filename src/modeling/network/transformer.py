import math
import torch
import torch.nn as nn
from .build import NETWORK_REGISTRY
from src.engine.build import device


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Net(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        num_classes=3,
        vocab_size=1,  # length of input
        d_model=200,  # dimension of embedding
        nhead=5,
        dim_feedforward=512,
        num_layers=4,
        dropout=0.5,
    ):
        super().__init__()

        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.project = nn.Sequential(
            nn.Linear(vocab_size * d_model, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, d_model),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.flatten(1)
        x = self.project(x)
        x = self.classifier(x)
        return x


@NETWORK_REGISTRY.register()
def transformer(cfg=None):
    model = Net()
    return model.to(device)


if __name__ == '__main__':
    x_in = torch.randn((32, 1, 200))
    net = Net()
    print(net(x_in).shape)
