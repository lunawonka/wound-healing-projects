import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
from torchvision import transforms


class BBBC021ConditionalDataset(Dataset):
    def __init__(
        self,
        resolved_csv,
        split_json,
        split="train",
        crop_size=256,
        moa_to_idx=None,
    ):
        self.df = pd.read_csv(resolved_csv)

        with open(split_json, "r") as f:
            split_data = json.load(f)

        indices = split_data[f"{split}_indices"]
        self.df = self.df.iloc[indices].reset_index(drop=True)

        if moa_to_idx is None:
            moa_classes = sorted(self.df["moa"].unique().tolist())
            self.moa_to_idx = {m: i for i, m in enumerate(moa_classes)}
        else:
            self.moa_to_idx = moa_to_idx

        self.idx_to_moa = {v: k for k, v in self.moa_to_idx.items()}

        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(crop_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _norm(x):
        x = x.astype(np.float32)
        x = x - x.min()
        denom = x.max() - x.min()
        if denom > 0:
            x = x / denom
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        dapi = self._norm(tiff.imread(row["path_dapi"]))
        actin = self._norm(tiff.imread(row["path_actin"]))
        tubulin = self._norm(tiff.imread(row["path_tubulin"]))

        img = np.stack([dapi, actin, tubulin], axis=-1).astype(np.float32)
        img = self.transform(img)

        label = torch.tensor(self.moa_to_idx[row["moa"]], dtype=torch.long)
        return img, label
        