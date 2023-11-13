import torch
import pytorch_lightning as pl
from .unet import UNet
from .loss import dice_score, ComboLoss


class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, weight: bool = False):
        super().__init__()
        self.model = UNet(in_channels=3)
        self.lr = lr
        if weight:
            self.criterion = ComboLoss(weight=torch.Tensor([[0.4, 0.55, 0.05]]))
        else:
            self.criterion = ComboLoss()

    def forward(self, x):
        return self.model(x)

    def _forward(self, batch, batch_idx, name="train"):
        image, mask = batch['image'].float(), batch['mask'].squeeze(1).long()
        logits = self(image)
        loss = self.criterion(logits, mask)
        d_score = dice_score(logits, mask)
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
