import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .build import DATA_LOADER_REGISTRY


class UBKData(Dataset):

    def __init__(self):
        self.data = np.random.random([2400, 250])
        self.data = torch.as_tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "data": self.data[idx, :245],
            "label": self.data[idx, -5:],
        }
        return sample


@DATA_LOADER_REGISTRY.register()
def ubk_dataloader(cfg, mode="train"):
    if mode == "train":
        dataset = UBKData()
    elif mode == "valid":
        dataset = UBKData()
    elif mode == "test":
        dataset = UBKData()
    else:
        raise ValueError(
            "mode must be one of 'train' or 'valid' or test' "
        )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader