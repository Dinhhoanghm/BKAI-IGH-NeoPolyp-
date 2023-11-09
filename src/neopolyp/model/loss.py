import torch
from torch import Tensor


def dice_loss(
    image: Tensor,
    mask: Tensor,
    epsilon: float = 1e-6
):
    inter = 2 * (image * mask).sum()
    sets_sum = image.sum() + mask.sum()
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice


if __name__ == "__main__":
    image = torch.rand(4, 3, 256, 256)
    mask = torch.rand(4, 3, 256, 256)
    print(dice_loss(image.flatten(0, 1), mask.flatten(0, 1)))
