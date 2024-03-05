import torch
from torch import nn


class CNN_MaxPool(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size:int, padding:int, stride_pool: int, padding_pool:int, batch_norm: bool, p_dropout: float):
        super().__init__()
        net = [
            nn.Conv2d(in_features, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        if batch_norm:
            net.append(nn.BatchNorm2d(32))

        layer=[nn.MaxPool2d(kernel_size=kernel_size, stride=stride_pool, padding=padding_pool),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        net+=layer
        if not batch_norm and p_dropout>0:
            net.append(nn.Dropout(p_dropout))

        layers=[
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride_pool, padding=padding_pool),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride_pool, padding=padding_pool),
            nn.Conv2d(128, out_features, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        net+=layers
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)