# https://github.com/jyh6681/VGU-Net/blob/master/VGUNet.py

import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.double_conv(x)


class SpatialGCN(nn.Module):
    def __init__(self, plane,inter_plane=None,out_plane=None):
        super(SpatialGCN, self).__init__()
        if inter_plane==None:
            inter_plane = plane #// 2
        if out_plane==None:
            out_plane = plane
        self.plane = plane
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.conv_wgl = nn.Linear(inter_plane,out_plane)
        self.bn1 = nn.BatchNorm1d(out_plane)
        self.conv_wgl2 = nn.Linear(out_plane, out_plane)
        self.bn2 = nn.BatchNorm1d(out_plane)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        node_k = self.node_k(
            x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)   #####nosym better, softmax better,only one gcn better
        node_q = self.node_q(x)  # x#copy.deepcopy(x)#F.normalize(x,p=1,dim=-1)#
        # print("input:",x.shape,node_k.shape)
        node_v = self.node_v(x)  # x#
        b, c, h, w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)  ##b N C
        node_q = node_q.view(b, c, -1)  ###b c N
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)  ##b N C
        Adj = torch.bmm(node_k, node_q)  ###Q*K^T

        # test using cosine=(a*b)/||a||*||b|| to construct adjacency
        # Adj = torch.bmm(node_k,node_q)#ab_ij=node_i*node_j
        # batch_row_norm = torch.norm(node_k,dim=-1).unsqueeze(-1)
        # Adj = torch.div(Adj,torch.bmm(batch_row_norm,batch_row_norm.permute(0,2,1)))

        Adj = self.softmax(Adj)  ###adjacency matrix of size b N N

        # max = torch.max(Adj, dim=2)
        # min = torch.min(Adj, dim=2)
        # Adj = (Adj - min.values[:, :, None]) / max.values[:, :, None]  # normalized adjacency matrix
        # Adj[Adj<0.5]=0

        AV = torch.bmm(Adj,node_v)###AX
        AVW = F.relu(self.bn1(self.conv_wgl(AV).transpose(1,2)).transpose(1,2))###AXW b n C
        AVW = F.dropout(AVW)
        # add one more layer
        AV = torch.bmm(Adj,AVW)
        AVW = F.relu(self.bn2(self.conv_wgl2(AV).transpose(1,2)).transpose(1,2))
        AVW = F.dropout(AVW)
        # end
        AVW = AVW.transpose(1, 2).contiguous()###AV withj shape NxC,N=mxn
        b,c,n = AVW.shape
        AVW = AVW.view(b, c, h, -1)
        return AVW


class VGUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2,base_nc=64,fix_grad=True, layers_mults=(1, 2, 4)):
        super(VGUNet, self).__init__()
        self.fix_grad = fix_grad
        in_channels = [in_ch] + [base_nc * i for i in layers_mults[:-1]]
        out_channels = [base_nc * i for i in layers_mults]
        self.enc_convs = nn.ModuleList([
            DoubleConv(inc, outc) for inc, outc in zip(in_channels, out_channels)
        ])
        self.pools = nn.ModuleList([
            nn.Conv2d(conv.out_channels, conv.out_channels, 2, stride=2, padding=0, bias=False) for conv in self.enc_convs
        ])

        self.sgcns = nn.ModuleList([nn.Identity()]) # the last block do not use spatial gcn
        for i in range(1, len(self.enc_convs)):
            self.sgcns.append(SpatialGCN(self.enc_convs[i].out_channels))
        self.sgcns.append(SpatialGCN(self.enc_convs[-1].out_channels))
        self.sgcns = self.sgcns[::-1]

        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(self.sgcns[i].plane, self.enc_convs[-(i+1)].out_channels, 2, stride=2, padding=0, bias=False) for i in range(len(self.enc_convs))
        ])
        
        self.dec_convs = nn.ModuleList([
            DoubleConv(self.enc_convs[-(i+1)].out_channels*2, self.enc_convs[-(i+1)].out_channels) for i in range(len(self.enc_convs))
        ])
        self.dec_convs.append(nn.Conv2d(base_nc, out_ch, kernel_size=1, padding=0))
        
    def forward(self,x):
        cs = []
        for i in range(len(self.enc_convs)):
            x = self.enc_convs[i](x)
            cs.append(x)
            x = self.pools[i](x)

        c = self.sgcns[0](x)   ###spatial gcn 4nc

        for i in range(len(self.ups)):
            up = self.ups[i](c)
            merge = torch.cat([up, self.sgcns[i+1](cs[-(i+1)])], dim=1)
            c = self.dec_convs[i](merge)
        c= self.dec_convs[-1](c)
        return c
