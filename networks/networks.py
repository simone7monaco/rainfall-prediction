import torch
from torch import nn

def getNetworkModel(network_model:str, in_features:int):
    if network_model=="NN2l_01":
        from fcnn import NN2l_01
        return NN2l_01(in_features=in_features,out_features=1)
    elif network_model=="NN2l_02":
        from fcnn import NN2l_02
        return NN2l_02(in_features=in_features,out_features=1)
    elif network_model=="NN2l_03":
        from fcnn import NN2l_03
        return NN2l_03(in_features=in_features,out_features=1, p_dropout=0.2)
    elif network_model=="cnn_2d_01":
        from cnn import cnn_2d_01
        return cnn_2d_01(in_features=in_features,out_features=1)
    elif network_model=="cnn_2d_02":
        from cnn import cnn_2d_02
        return cnn_2d_02(in_features=in_features,out_features=1)
    elif network_model=="cnn_2d_03":
        from cnn import cnn_2d_03
        return cnn_2d_03(in_features=in_features,out_features=1, p_dropout=0.2)
