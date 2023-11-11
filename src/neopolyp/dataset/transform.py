import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def label_mask(mask):

    A = mask[:, :, 0]  # Red channel
    B = mask[:, :, 1]  # Green channel
    C = mask[:, :, 2]  # Blue channel

    max_values = np.maximum(np.maximum(A, B), C)

    result = np.zeros_like(max_values)

    result[max_values == A] = 1
    result[max_values == B] = 2
    result[max_values == C] = 0

    one_hot = np.zeros((256, 256, 3), dtype=np.float32)
    for i in range(1, 3):
        one_hot[:, :, i] = result == i

    return one_hot


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
                image=label_mask(data['mask'])
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
