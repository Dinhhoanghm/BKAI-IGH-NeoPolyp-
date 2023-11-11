import os
from .transform import TrainTransform, TestTransform
from torch.utils.data import Dataset
import cv2


class NeoPolypDataset(Dataset):
    def __init__(self, session: str = "train", path: str = "data") -> None:
        super().__init__()
        self.session = session
        if session == "train":
            self.train_path = []
            for root, dirs, files in os.walk(os.path.join(path, "train")):
                for f in files:
                    self.train_path.append(os.path.join(root, f))
            self.train_gt_path = []
            for root, dirs, files in os.walk(os.path.join(path, "train_gt")):
                for f in files:
                    self.train_gt_path.append(os.path.join(root, f))
            self.len = len(self.train_path)
            self.train_transform = TrainTransform()
        else:
            self.test_path = []
            for root, dirs, files in os.walk(os.path.join(path, "test")):
                for f in files:
                    self.test_path.append(os.path.join(root, f))
            self.len = len(self.test_path)
            self.test_transform = TestTransform()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        if self.session == "train":
            img = cv2.imread(self.train_path[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(self.train_gt_path[index])
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            return self.train_transform(img, gt)
        else:
            img = cv2.imread(self.test_path[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.test_transform(img)
            return img


def collate_fn(batch):
    pass
