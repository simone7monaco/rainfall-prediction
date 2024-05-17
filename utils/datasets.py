import torch
from torch import nn
from torch.utils.data import Dataset

from pathlib import Path
import numpy as np
import pandas as pd
from utils import io

class HideModel(nn.Module):
    """
    Data augmentation strategy to hide one of the channels of the input image, thus simulating the absence of up to 2 NWP models.
    """
    def __init__(self, probs: torch.Tensor = torch.tensor([.3, .3, .3, .3]), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs = probs
        
    def forward(self, img):
        if img.sum(axis=(1, 2)).eq(0).any():
            return img
        
        channel_idx = torch.bernoulli(self.probs)
        while channel_idx.sum() > 2:
            channel_idx = torch.bernoulli(self.probs)
        img[channel_idx.bool(), :, :] = 0
        return img
    
class NWPDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None): # transforms=HideModel()
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform is not None:
            x = self.transform(x)
        if len(self.tensors) == 2:
            return ({'x': x, 'y': self.tensors[1][index]})
        return ({'x': x, 'y': self.tensors[1][index], 'ev_date': self.tensors[2][index]})

    def __len__(self):
        return self.tensors[0].size(0)

class ExtremeDataset(NWPDataset):
    """
    Wrapper of NWPDataset returning the extreme events and the test events for the current split,
    together with a boolean label indicating if the event is extreme or not.
    """
    def __init__(self, split_idx, transform=None, seed=42,
                 input_path=Path("/media/monaco/DATA1/case_study/24h_10mmMAX_OI")):
        case_study = input_path.stem
        case_study_max, available_models, _, _, test_dates, indices_one, indices_zero, _, _, _ = io.get_casestudy_stuff(
			input_path, n_split=split_idx, case_study=case_study, ispadded=True, seed=seed)
 
        extreme_events = pd.concat([pd.read_csv(exev, header=None) for exev in input_path.glob('*extremeEvents.csv')])[0].values
        not_extreme_events = np.array([date for date in test_dates if date not in extreme_events])
        all_dates = np.concatenate([extreme_events, not_extreme_events])
        self.labels = torch.cat([torch.ones(len(extreme_events)), torch.zeros(len(not_extreme_events))]).long()

        X, Y, _, _ = io.load_data(input_path, all_dates, case_study_max, 
                                                       indices_one, indices_zero, available_models)
        super().__init__((torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(1), 
                          torch.from_numpy(all_dates)), transform=transform)
    
    def __getitem__(self, index):
        return super().__getitem__(index) | {'isextreme_label': self.labels[index]}
        
    