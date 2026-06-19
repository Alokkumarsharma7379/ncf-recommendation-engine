import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class NCFDataset(Dataset):
    """BCE-style dataset: (user, item, label) rows."""

    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(df["user"].values, dtype=torch.long)
        self.items = torch.tensor(df["item"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]


class BPRDataset(Dataset):
    """BPR-style dataset: (user, pos_item, neg_item) triplets, no label."""

    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(df["user"].values, dtype=torch.long)
        self.pos_items = torch.tensor(df["pos_item"].values, dtype=torch.long)
        self.neg_items = torch.tensor(df["neg_item"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]


def get_loader(df: pd.DataFrame, batch_size: int = 1024, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        NCFDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_bpr_loader(df: pd.DataFrame, batch_size: int = 1024, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        BPRDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )