import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2 as cv


def _transform_mask(image):
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv.inRange(image, lower1, upper1)
    upper_mask = cv.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask
    red_mask[red_mask != 0] = 2

    # boundary GREEN color range values; Hue (36 - 70)
    green_mask = cv.inRange(image, (36, 25, 25), (70, 255, 255))
    green_mask[green_mask != 0] = 1

    full_mask = cv.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    return full_mask


class TrainTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
        ])

        self.to_tensor = ToTensorV2()

    def __call__(self, img, mask):
        data = self.transform(image=img, mask=mask)
        return {
            'image': self.to_tensor(
                image=data['image'] / 255.
            )['image'],
            'mask': self.to_tensor(
                image=_transform_mask(data['mask'])
            )['image']
        }


class TestTransform:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ])

    def __call__(self, img):
        return self.transform(image=img)
