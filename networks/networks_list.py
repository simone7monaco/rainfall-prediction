import torch
from torch import nn

def getNetworkModel(network_model:str, in_features:int, out_features:int, cnn_param: list, mask: torch.Tensor, device:str):
    network_model_split=network_model.split("_")
    if network_model_split[0]=="NN":
        kernel_size=cnn_param[0]
        stride_pool=cnn_param[2]
        padding_pool=cnn_param[3]        
        mask=torch.from_numpy(mask).type(torch.bool).to(device)
        from networks.fcnn_new import FCNN, FCNN_FinalAvgPool
        n_layers=int(network_model_split[1][0])
        batch_norm=False
        p_dropout=0
        if network_model_split[2]=="02":
            batch_norm=True
            p_dropout=0
        elif network_model_split[2]=="03":
            batch_norm=False
            p_dropout=0.2
        n_neurons=int(network_model_split[3])
        # return FCNN(
        #     in_features=in_features,
        #     out_features=out_features,
        #     n_layers=n_layers,
        #     n_neurons=n_neurons,
        #     batch_norm=batch_norm,
        #     p_dropout=p_dropout
        #     )
        return FCNN_FinalAvgPool(
            in_features=in_features,
            out_features=out_features,
            n_layers=n_layers,
            n_neurons=n_neurons,
            batch_norm=batch_norm,
            p_dropout=p_dropout,
            kernel_size=kernel_size,
            stride_pool=stride_pool,
            padding_pool=padding_pool,
            mask=mask,
            device=device
            )

    elif network_model_split[0]=="cnn":
        from networks.cnn_new import CNN_MaxPool
        batch_norm=False
        p_dropout=0
        if network_model_split[2]=="02":
            batch_norm=True
            p_dropout=0
        elif network_model_split[2]=="03":
            batch_norm=False
            p_dropout=0.2

        kernel_size=cnn_param[0]
        padding=cnn_param[1]
        stride_pool=cnn_param[2]
        padding_pool=cnn_param[3]
        return CNN_MaxPool(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride_pool=stride_pool,
            padding_pool=padding_pool,
            batch_norm=batch_norm,
            p_dropout=p_dropout
            )
    elif network_model_split[0]=="unet":
        from networks.unet import UNet
        batch_norm=False
        p_dropout=0
        if network_model_split[2]=="02":
            batch_norm=True
            p_dropout=0
        elif network_model_split[2]=="03":
            batch_norm=False
            p_dropout=0.2
        return UNet(in_features=in_features, out_features=out_features)
    # if network_model=="NN_2l_01":
    #     return NN2l_01(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_2l_02":
    #     from networks.fcnn import NN2l_02
    #     return NN2l_02(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_2l_03":
    #     from networks.fcnn import NN2l_03
    #     return NN2l_03(in_features=in_features,out_features=out_features, p_dropout=0.2)
    # elif network_model=="NN_3l_01":
    #     from networks.fcnn import NN3l_01
    #     return NN3l_01(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_3l_02":
    #     from networks.fcnn import NN3l_02
    #     return NN3l_02(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_3l_03":
    #     from networks.fcnn import NN3l_03
    #     return NN3l_03(in_features=in_features,out_features=out_features, p_dropout=0.2)    
    # elif network_model=="NN_4l_01":
    #     from networks.fcnn import NN4l_01
    #     return NN4l_01(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_4l_02":
    #     from networks.fcnn import NN4l_02
    #     return NN4l_02(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_4l_03":
    #     from networks.fcnn import NN4l_03
    #     return NN4l_03(in_features=in_features,out_features=out_features, p_dropout=0.2)                
    # elif network_model=="cnn_2d_01":
    #     from networks.cnn import cnn_2d_01
    #     return cnn_2d_01(in_features=in_features,out_features=out_features)
    # elif network_model=="cnn_2d_02":
    #     from networks.cnn import cnn_2d_02
    #     return cnn_2d_02(in_features=in_features,out_features=out_features)
    # elif network_model=="cnn_2d_03":
    #     from networks.cnn import cnn_2d_03
    #     return cnn_2d_03(in_features=in_features,out_features=out_features, p_dropout=0.2)
    # elif network_model=="NN_2l_01_500":
    #     from networks.fcnn_500 import NN2l_01_500
    #     return NN2l_01_500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_2l_02_500":
    #     from networks.fcnn_500 import NN2l_02_500
    #     return NN2l_02_500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_2l_03_500":
    #     from networks.fcnn_500 import NN2l_03_500
    #     return NN2l_03_500(in_features=in_features,out_features=out_features, p_dropout=0.2)
    # elif network_model=="NN_3l_01_500":
    #     from networks.fcnn_500 import NN3l_01_500
    #     return NN3l_01_500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_3l_02_500":
    #     from networks.fcnn_500 import NN3l_02_500
    #     return NN3l_02_500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_3l_03_500":
    #     from networks.fcnn_500 import NN3l_03_500
    #     return NN3l_03_500(in_features=in_features,out_features=out_features, p_dropout=0.2)    
    # elif network_model=="NN_4l_01_500":
    #     from networks.fcnn_500 import NN4l_01_500
    #     return NN4l_01_500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_4l_02_500":
    #     from networks.fcnn_500 import NN4l_02_500
    #     return NN4l_02_500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_4l_03_500":
    #     from networks.fcnn_500 import NN4l_03_500
    #     return NN4l_03_500(in_features=in_features,out_features=out_features, p_dropout=0.2)            
    # elif network_model=="NN_2l_01_1500":
    #     from networks.fcnn_1500 import NN2l_01_1500
    #     return NN2l_01_1500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_2l_02_1500":
    #     from networks.fcnn_1500 import NN2l_02_1500
    #     return NN2l_02_1500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_2l_03_1500":
    #     from networks.fcnn_1500 import NN2l_03_1500
    #     return NN2l_03_1500(in_features=in_features,out_features=out_features, p_dropout=0.2)
    # elif network_model=="NN_3l_01_1500":
    #     from networks.fcnn_1500 import NN3l_01_1500
    #     return NN3l_01_1500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_3l_02_1500":
    #     from networks.fcnn_1500 import NN3l_02_1500
    #     return NN3l_02_1500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_3l_03_1500":
    #     from networks.fcnn_1500 import NN3l_03_1500
    #     return NN3l_03_1500(in_features=in_features,out_features=out_features, p_dropout=0.2)    
    # elif network_model=="NN_4l_01_1500":
    #     from networks.fcnn_1500 import NN4l_01_1500
    #     return NN4l_01_1500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_4l_02_1500":
    #     from networks.fcnn_1500 import NN4l_02_1500
    #     return NN4l_02_1500(in_features=in_features,out_features=out_features)
    # elif network_model=="NN_4l_03_1500":
    #     from networks.fcnn_1500 import NN4l_03_1500
    #     return NN4l_03_1500(in_features=in_features,out_features=out_features, p_dropout=0.2)          