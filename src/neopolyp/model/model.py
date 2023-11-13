import torch
import torch.nn as nn
import pytorch_lightning as pl
from .unet import UNet
from .loss import dice_score, FocalTverskyLoss, DiceLoss


class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.model = UNet(in_channels=3)
        self.lr = lr
        self.ce_loss = nn.CrossEntropyLoss()
        self.d_loss = DiceLoss()
        self.ft_loss = FocalTverskyLoss()

    def forward(self, x):
        return self.model(x)

    def _forward(self, batch, batch_idx, name="train"):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        logits = self(image)
        ce_loss = self.ce_loss(logits, mask)
        # ft_loss = self.ft_loss(logits, mask)
        d_loss = self.d_loss(logits, mask)
        with torch.no_grad():
            d_score = dice_score(logits, mask)
        # loss = ce_loss + ft_loss
        loss = (ce_loss + d_loss) / 2
        self.log_dict(
            {
                f"{name}_loss": loss,
                f"{name}_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
