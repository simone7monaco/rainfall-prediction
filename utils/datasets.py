# from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset

from torch import nn
import torch


class HideModel(nn.Module):
    def __init__(self, probs=torch.tensor([0.3, 0.3, 0.3, 0.3]), *args, **kwargs):
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
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):  # HideModel()
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform is not None:
            x = self.transform(x)
        if len(self.tensors) == 2:
            return {"x": x, "y": self.tensors[1][index]}
        return {"x": x, "y": self.tensors[1][index], "ev_date": self.tensors[2][index]}

    def __len__(self):
        return self.tensors[0].size(0)
