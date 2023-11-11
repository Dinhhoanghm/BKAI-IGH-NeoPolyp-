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
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        one_hot_mask = F.one_hot(mask, num_classes=3).permute(0, 3, 1, 2)
        logits = self(image)
        e_loss = F.cross_entropy(logits, mask)
        # f_loss = focal_tversky_loss(logits, one_hot_mask)
        d_score = dice_score(logits, one_hot_mask)
        # loss = (e_loss + f_loss) / 2
        self.log_dict(
            {
                "train_loss": e_loss,
                # "train_entropy_loss": e_loss,
                # "train_ft_loss": f_loss,
                "train_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True
        )
        return e_loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        one_hot_mask = F.one_hot(mask, num_classes=3).permute(0, 3, 1, 2)
        logits = self(image)
        e_loss = F.cross_entropy(logits, mask)
        # f_loss = focal_tversky_loss(logits, one_hot_mask)
        d_score = dice_score(logits, one_hot_mask)
        # loss = (e_loss + f_loss) / 2
        self.log_dict(
            {
                "val_loss": e_loss,
                # "val_entropy_loss": e_loss,
                # "val_ft_loss": f_loss,
                "val_dice_score": d_score
            },
            on_step=False, on_epoch=True, sync_dist=True
        )
        return e_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
