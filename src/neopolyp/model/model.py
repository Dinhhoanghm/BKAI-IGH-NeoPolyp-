import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .unet import UNet
from .loss import dice_loss


class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.model = UNet(in_channels=3)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].permute(0, 3, 1, 2)
        y_hat = self(image)
        e_loss = F.cross_entropy(y_hat, mask.float())
        # d_loss = dice_loss(
        #     F.softmax(y_hat, dim=1).float().flatten(0, 1),
        #     F.one_hot(mask.long(), 3).permute(0, 3, 1, 2).float().flatten(0, 1)
        # )
        loss = e_loss  # + d_loss
        self.log("train_loss", e_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].permute(0, 3, 1, 2)
        y_hat = self(image)
        e_loss = F.cross_entropy(y_hat, mask.float())
        # d_loss = dice_loss(
        #     F.softmax(y_hat, dim=1).float().flatten(0, 1),
        #     F.one_hot(mask.long(), 3).float().flatten(0, 1)
        # )
        loss = e_loss  # + d_loss
        self.log("val_loss", e_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
