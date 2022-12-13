import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .build import DATA_LOADER_REGISTRY


class UBKData(Dataset):

    def __init__(self, mode):
        if mode == "train":
            data = np.load("/home/rongfan/18-UBK/UBKProj/dataset/ukb_array_train.npy")
        if mode == "valid":
            data = np.load("/home/rongfan/18-UBK/UBKProj/dataset/ukb_array_valid.npy")
        if mode == "test":
            data = np.load("/home/rongfan/18-UBK/UBKProj/dataset/ukb_array_test.npy")

        self.data = data[:2000]
        self.data = torch.as_tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, -1].clip(min=0).type(torch.LongTensor)

        sample = {
            "data": self.data[idx, :-1],
            "label": label,
        }
        return sample


@DATA_LOADER_REGISTRY.register()
def ubk_dataloader(cfg, mode="train"):
    if mode == "train":
        dataset = UBKData(mode)
    elif mode == "valid":
        dataset = UBKData(mode)
    elif mode == "test":
        dataset = UBKData(mode)
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