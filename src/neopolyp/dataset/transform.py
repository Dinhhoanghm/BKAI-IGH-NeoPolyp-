import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class TrainTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), eps=None, always_apply=False, p=0.3),
            A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(),
                     A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class ValTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img, mask):
        return self.transform(image=img, mask=mask)


class TestTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)['image']
