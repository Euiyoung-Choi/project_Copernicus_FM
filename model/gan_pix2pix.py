from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_bn: bool = True):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 4):
        super().__init__()
        self.e1 = _ConvBlock(in_channels, 64, down=True, use_bn=False)    # 256 -> 128
        self.e2 = _ConvBlock(64, 128, down=True, use_bn=True)             # 128 -> 64
        self.e3 = _ConvBlock(128, 256, down=True, use_bn=True)            # 64 -> 32
        self.e4 = _ConvBlock(256, 512, down=True, use_bn=True)            # 32 -> 16

        self.d1 = _ConvBlock(512, 256, down=False, use_bn=True)           # 16 -> 32
        self.d2 = _ConvBlock(256 + 256, 128, down=False, use_bn=True)     # 32 -> 64
        self.d3 = _ConvBlock(128 + 128, 64, down=False, use_bn=True)      # 64 -> 128
        self.d4 = nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1)  # 128 -> 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        d1 = self.d1(e4)
        d2 = self.d2(torch.cat([d1, e3], dim=1))
        d3 = self.d3(torch.cat([d2, e2], dim=1))
        d4 = self.d4(torch.cat([d3, e1], dim=1))
        return torch.sigmoid(d4)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 4):
        super().__init__()
        c = in_channels + out_channels
        self.net = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([inp, out], dim=1))


def build_pix2pix_generator(in_channels: int = 2, out_channels: int = 4):
    return Pix2PixGenerator(in_channels=in_channels, out_channels=out_channels)


def build_patchgan_discriminator(in_channels: int = 2, out_channels: int = 4):
    return PatchDiscriminator(in_channels=in_channels, out_channels=out_channels)
