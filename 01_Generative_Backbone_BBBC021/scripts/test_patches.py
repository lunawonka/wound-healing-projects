from torch.utils.data import DataLoader
from dataset_bbbc021_patches import BBBC021PatchDataset

metadata_csv = "/data/annapan/prop/bbbc021/patches_256_metadata.csv"

train_dataset = BBBC021PatchDataset(metadata_csv, split="train")
val_dataset = BBBC021PatchDataset(metadata_csv, split="val", moa_to_idx=train_dataset.moa_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

x, y = next(iter(train_loader))
print("Train patch batch shape:", x.shape)
print("Train labels shape:", y.shape)
print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))
print("Num classes:", len(train_dataset.moa_to_idx))
print("Value range:", x.min().item(), x.max().item())
