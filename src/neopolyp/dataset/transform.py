import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch


class TrainTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(240, 320, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            # A.Rotate(limit=30, p=0.3),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        data = self.transform(image=img, mask=mask)
        image = data['image'] / 255.
        mask = data['mask'] / 255.
        mask = torch.where(mask > 0.65, 1.0, 0.0)
        mask[:, :, 2] = 0.0001
        mask = torch.argmax(mask, 2).type(torch.int64)

        return {
            'image': image,
            'mask': mask
        }


class TestTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(240, 320, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)['image'] / 255.
