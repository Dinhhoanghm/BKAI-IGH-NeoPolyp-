import torch
from torch import Tensor


def dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    sum_dim = (-1, -2, -3)
    out_softmax = torch.softmax(logits, dim=1)
    inter = 2 * (out_softmax * target).sum(dim=sum_dim)
    sets_sum = out_softmax.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return 1 - dice.mean()


if __name__ == "__main__":
    a = torch.rand(4, 3, 8, 8)
    b = torch.randint(0, 2, (4, 3, 8, 8))
    print(dice_loss(a, b))
