import torch
from torch import nn


class FCNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_layers: int, n_neurons: int, batch_norm:bool, p_dropout: float):
        super().__init__()
        net=[
             nn.Linear(in_features, n_neurons),
             #nn.ReLU()
            ] 
        if batch_norm:
            net.append(nn.BatchNorm1d(n_neurons))
        elif p_dropout>0:
            net.append(nn.Dropout(p_dropout))
        for i in range(0,n_layers-1):
            net.append(nn.Linear(n_neurons, n_neurons))
            #net.append(nn.ReLU())
        
        final_layer=[
            nn.Linear(n_neurons, out_features),
            nn.ReLU()
        ]

        net+=final_layer
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        return self.net(x)
    
class FCNN_FinalAvgPool(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_layers: int, n_neurons: int, batch_norm:bool, p_dropout: float, kernel_size:int, stride_pool: int, padding_pool:int, mask: torch.Tensor, device: str):
        super().__init__()
        self.mask=mask
        self.device=device
        self.m=nn.AvgPool2d(kernel_size=kernel_size, stride=stride_pool, padding=padding_pool)

        net=[
             nn.Linear(in_features, n_neurons),
             #nn.ReLU()
            ] 
        if batch_norm:
            net.append(nn.BatchNorm1d(n_neurons))
        elif p_dropout>0:
            net.append(nn.Dropout(p_dropout))
        for i in range(0,n_layers-1):
            net.append(nn.Linear(n_neurons, n_neurons))
            #net.append(nn.ReLU())
        
        final_layer=[
            nn.Linear(n_neurons, out_features),
            nn.ReLU(),
        ]

        net+=final_layer
        self.net = nn.Sequential(*net)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B, in_features, H, W) -> (B, out_features, H, W)
        y=self.net(x)
        self.output=torch.zeros([x.size(dim=0),self.mask.size(dim=0), self.mask.size(dim=1)], dtype=torch.float).to(self.device)
        #torch.reshape(self.output,(x.size(dim=0),self.output.size(dim=1),self.output.size(dim=2)))
        # print(y.size())
        # print(self.output[:,self.mask].size())
        self.output[:,self.mask]=y
        return self.m(self.output)    