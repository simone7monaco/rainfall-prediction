import torch
from torch import nn


class NN2l_01(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)

class NN2l_02(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)
    
class NN2l_03(nn.Module):
    def __init__(self, in_features: int, out_features: int,p_dropout: float):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)    


class NN3l_01(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),            
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)

class NN3l_02(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),            
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)
    
class NN3l_03(nn.Module):
    def __init__(self, in_features: int, out_features: int,p_dropout: float):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),            
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)   
    
class NN4l_01(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),                         
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)

class NN4l_02(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(), 
            nn.Linear(1000, 1000),
            nn.ReLU(),                        
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)
    
class NN4l_03(nn.Module):
    def __init__(self, in_features: int, out_features: int,p_dropout: float):
        super().__init__()
        net = [
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),   
            nn.Linear(1000, 1000),
            nn.ReLU(),                      
            nn.Linear(1000, out_features),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)      