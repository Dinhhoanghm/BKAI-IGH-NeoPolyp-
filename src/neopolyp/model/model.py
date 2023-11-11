import torch
import torch.nn as nn
import pytorch_lightning as pl
from .unet import UNet
from .loss import DiceLoss, FocalTverskyLoss


class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.model = UNet(in_channels=3)
        self.lr = lr
        self.ce_loss = nn.CrossEntropyLoss()
        self.ft_loss = FocalTverskyLoss()
        self.d_loss = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        logits = self(image)
        ce_loss = self.ce_loss(logits, mask)
        ft_loss = self.ft_loss(logits, mask)
        with torch.no_grad():
            d_loss, d_score = self.d_loss(logits, mask)
        loss = ce_loss + ft_loss
        self.log_dict(
            {
                "train_loss": loss,
                "train_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        logits = self(image)
        ce_loss = self.ce_loss(logits, mask)
        ft_loss = self.ft_loss(logits, mask)
        with torch.no_grad():
            d_loss, d_score = self.d_loss(logits, mask)
        loss = ce_loss + ft_loss
        self.log_dict(
            {
                "val_loss": loss,
                "val_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
