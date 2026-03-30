from torch.utils.data import DataLoader
from dataset_bbbc021 import BBBC021ConditionalDataset

resolved_csv = "/data/annapan/prop/bbbc021/bbbc021_moa_resolved.csv"
split_json = "/data/annapan/prop/bbbc021/split_80_20_stratified_grouped.json"

train_dataset = BBBC021ConditionalDataset(
    resolved_csv=resolved_csv,
    split_json=split_json,
    split="train",
    crop_size=256,
)

val_dataset = BBBC021ConditionalDataset(
    resolved_csv=resolved_csv,
    split_json=split_json,
    split="val",
    crop_size=256,
    moa_to_idx=train_dataset.moa_to_idx,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

x, y = next(iter(train_loader))

print("Train batch image shape:", x.shape)
print("Train batch label shape:", y.shape)
print("Number of classes:", len(train_dataset.moa_to_idx))
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Unique train labels in batch:", y.unique().tolist())
