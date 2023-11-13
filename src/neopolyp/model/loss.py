import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor,
            num_classes: int,
            eps: float = 1e-6
            ) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.
        Label: (N, H, W) -> Onehot: (N, C, H, W)
    """
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width).to(labels.device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


@torch.no_grad()
def dice_score(
    inputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # compute softmax over the classes axis
    input_one_hot = one_hot(inputs.argmax(dim=1), num_classes=inputs.shape[1])

    # create the labels one hot tensor
    target_one_hot = one_hot(targets, num_classes=inputs.shape[1])

    # compute the actual dice score
    dims = (2, 3)
    intersection = torch.sum(input_one_hot * target_one_hot, dims)
    cardinality = torch.sum(input_one_hot + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + 1e-6)
    return dice_score.mean()


class DiceLoss(nn.Module):
    def __init__(self, weights=torch.Tensor([[0.4, 0.55, 0.05]])) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.weights: torch.Tensor = weights

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(targets, num_classes=inputs.shape[1])

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        dice_score = torch.sum(
            dice_score * self.weights.to(dice_score.device),
            dim=1
        )
        return torch.mean(1. - dice_score)


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):

        # compute softmax over the classes axis
        input_soft = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(targets, num_classes=inputs.shape[1])
        # flatten label and prediction tensors
        input_soft = input_soft.view(-1)
        target_one_hot = target_one_hot.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (input_soft * target_one_hot).sum()
        FP = ((1-target_one_hot) * input_soft).sum()
        FN = (target_one_hot * (1-input_soft)).sum()

        tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        focal_tversky = (1 - tversky)**gamma

        return focal_tversky

