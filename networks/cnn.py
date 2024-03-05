import torch
from torch import nn


class cnn_2d_01(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Conv2d(in_features, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, 128, kernel_size=5, padding="same"),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(128, out_features, kernel_size=5, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            # nn.Conv2d(256, out_features, kernel_size=5, padding="same"),
            # nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)

class cnn_2d_02(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Conv2d(in_features, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, out_features, kernel_size=5, padding="same"),
            nn.ReLU(),            
            # nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            # nn.Conv2d(128, out_features, kernel_size=5, padding="same"),
            # nn.ReLU(),            
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)

class cnn_2d_03(nn.Module):
    def __init__(self, in_features: int, out_features: int, p_dropout: float):
        super().__init__()
        net = [
            #nn.Dropout(p_dropout),
            nn.Conv2d(in_features, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            #nn.Dropout(p_dropout),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, 128, kernel_size=5, padding="same"),
            nn.ReLU(),        
            #nn.Dropout(p_dropout),    
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(128, out_features, kernel_size=5, padding="same"),
            nn.ReLU(),              
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)
