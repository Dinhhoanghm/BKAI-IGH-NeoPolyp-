import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .unet import UNet
from .loss import dice_score, focal_tversky_loss


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
        f_loss = focal_tversky_loss(y_hat, mask)
        d_score = dice_score(y_hat, mask)
        loss = e_loss + f_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_entropy_loss", e_loss, on_step=False, on_epoch=True)
        self.log("train_ft_loss", f_loss, on_step=False, on_epoch=True)
        self.log("train_dice_score", d_score, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].permute(0, 3, 1, 2)
        y_hat = self(image)
        e_loss = F.cross_entropy(y_hat, torch.argmax(mask, dim=1))
        f_loss = focal_tversky_loss(y_hat, mask)
        d_score = dice_score(y_hat, mask)
        loss = (e_loss + f_loss) / 2
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_entropy_loss", e_loss, on_step=False, on_epoch=True)
        self.log("val_ft_loss", f_loss, on_step=False, on_epoch=True)
        self.log("val_dice_score", d_score, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
