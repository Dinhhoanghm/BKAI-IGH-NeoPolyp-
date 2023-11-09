import albumentations as A
from albumentations.pytorch import ToTensorV2


class TrainTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class TestTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.pytorch.ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)
