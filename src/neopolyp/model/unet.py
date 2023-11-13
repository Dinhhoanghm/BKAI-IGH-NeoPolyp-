import torch
import torch.nn as nn
from torchsummary import summary


def _make_layers(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Attention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _make_layers(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super().__init__()
        self.attention = attention
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if attention:
            self.attn = Attention(out_channels, out_channels, out_channels//2)
        self.conv = _make_layers(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.convT(x1)
        if self.attention:
            x2 = self.attn(x1, x2)
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attention: bool = False
    ):
        super().__init__()
        self.attention = attention
        self.conv_in = _make_layers(in_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)
        self.up1 = UpSample(1024, 512, attention=attention)
        self.up2 = UpSample(512, 256, attention=attention)
        self.up3 = UpSample(256, 128, attention=attention)
        self.up4 = UpSample(128, 64, attention=attention)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, attention=True).to(device)
    print(summary(model, input_size=(3, 256, 256), device=device.type))
