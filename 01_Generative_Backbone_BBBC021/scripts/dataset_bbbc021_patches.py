import random
import pandas as pd
import torch
from torch.utils.data import Dataset


class BBBC021PatchDataset(Dataset):
    def __init__(self, metadata_csv, split="train", moa_to_idx=None, augment=True):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if moa_to_idx is None:
            moa_classes = sorted(self.df["moa"].unique().tolist())
            self.moa_to_idx = {m: i for i, m in enumerate(moa_classes)}
        else:
            self.moa_to_idx = moa_to_idx

        self.idx_to_moa = {v: k for k, v in self.moa_to_idx.items()}
        self.split = split
        self.augment = augment and (split == "train")

    def __len__(self):
        return len(self.df)

    def _augment_tensor(self, x):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical

        # random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k=k, dims=[1, 2])

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.load(row["path_pt"], map_location="cpu").float()  # [3, 256, 256], [0,1]

        if self.augment:
            x = self._augment_tensor(x)

        x = x * 2.0 - 1.0  # -> [-1, 1]
        y = torch.tensor(self.moa_to_idx[row["moa"]], dtype=torch.long)
        return x, y
    