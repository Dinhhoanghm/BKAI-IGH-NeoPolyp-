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
        e_loss = F.cross_entropy(y_hat, torch.argmax(mask, dim=1))
        d_loss = dice_loss(y_hat, mask)
        loss = e_loss + d_loss
        self.log("train_loss", loss)
        self.log("train_entropy_loss", e_loss)
        self.log("train_dice_loss", d_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].permute(0, 3, 1, 2)
        y_hat = self(image)
        e_loss = F.cross_entropy(y_hat, torch.argmax(mask, dim=1))
        d_loss = dice_loss(y_hat, mask)
        loss = e_loss + d_loss
        self.log("val_loss", e_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
