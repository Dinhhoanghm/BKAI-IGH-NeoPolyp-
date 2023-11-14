from .transform import TrainTransform, TestTransform, ValTransform
from torch.utils.data import Dataset
import cv2
import numpy as np


class NeoPolypDataset(Dataset):
    def __init__(
        self,
        image_dir: list,
        gt_dir: list | None = None,
        session: str = "train"
    ) -> None:
        super().__init__()
        self.session = session
        if session == "train":
            self.train_path = image_dir
            self.train_gt_path = gt_dir
            self.len = len(self.train_path)
            self.train_transform = TrainTransform()
        elif session == "val":
            self.val_path = image_dir
            self.val_gt_path = gt_dir
            self.len = len(self.val_path)
            self.val_transform = ValTransform()
        else:
            self.test_path = image_dir
            self.len = len(self.test_path)
            self.test_transform = TestTransform()

    @staticmethod
    def _read_mask(mask_path):
        image = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        red_mask = lower_mask + upper_mask
        red_mask[red_mask != 0] = 2

        # boundary RED color range values; Hue (36 - 70)
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
        green_mask[green_mask != 0] = 1

        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = full_mask.astype(np.uint8)
        return full_mask

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        if self.session == "train":
            img = cv2.imread(self.train_path[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = self._read_mask(self.train_gt_path[index])
            return self.train_transform(img, gt)
        elif self.session == "val":
            img = cv2.imread(self.val_path[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt = self._read_mask(self.val_gt_path[index])
            return self.val_transform(img, gt)
        else:
            img = cv2.imread(self.test_path[index])
            H, W, _ = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.test_transform(img)
            file_id = self.test_path[index].split('/')[-1].split('.')[0]
            return img, file_id, H, W
