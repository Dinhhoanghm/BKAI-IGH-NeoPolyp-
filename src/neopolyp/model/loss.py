import torch
from torch import Tensor
import torch.nn.functional as F

def dice_score(
    logits: Tensor,
    target: Tensor,
    smooth: float = 1e-6
):
    inputs = F.sigmoid(logits)
    inputs = inputs.reshape(-1)
    targets = target.reshape(-1)

    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()
    dice = (2.*intersection + smooth)/(union + smooth)

    return dice


def dice_loss(
    logits: Tensor,
    target: Tensor,
    epsilon: float = 1e-6
):
    return 1 - dice_score(logits, target, epsilon)


def focal_tversky_loss(
    logits: Tensor,
    targets: Tensor,
    smooth: float = 1e-6,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 1
):

    inputs = F.sigmoid(logits)
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
    
    TP = (inputs * targets).sum()
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    focal_tversky = (1 - tversky)**gamma

    return focal_tversky


if __name__ == "__main__":
    a = torch.rand(4, 3, 8, 8)
    b = torch.randint(0, 2, (4, 3, 8, 8))
    print(dice_loss(a, b))
    print(focal_tversky_loss(a, b))
