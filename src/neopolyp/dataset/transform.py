import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class TrainTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(480, 640, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=0.3),
            A.RandomCrop(240, 320),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class ValTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(240, 320, interpolation=cv2.INTER_LINEAR),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class TestTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(240, 320, interpolation=cv2.INTER_LINEAR),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)['image']
