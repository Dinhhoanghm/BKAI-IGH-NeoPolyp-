import torch
from .transform import Transform
from torch.utils.data import Dataset


class NeoPolypDataset(Dataset):
    def __init__(self, session: str = "train") -> None:
        super().__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int):
        pass


def collate_fn(batch):
    pass
