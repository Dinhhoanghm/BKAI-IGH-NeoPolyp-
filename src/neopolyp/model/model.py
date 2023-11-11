import torch
import pytorch_lightning as pl
from .unet import UNet
from .loss import DiceLoss


class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.model = UNet(in_channels=3)
        self.lr = lr
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        logits = self(image)
        d_loss, d_score = self.criterion(logits, mask)
        self.log_dict(
            {
                "train_loss": d_loss,
                "train_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True,prog_bar=True
        )
        return d_loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        logits = self(image)
        d_loss, d_score = self.criterion(logits, mask)
        self.log_dict(
            {
                "val_loss": d_loss,
                "val_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True,prog_bar=True
        )
        return d_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
