import torch.nn as nn
from config import *
from utils import utils


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv1(x)
        return skip, self.maxpool(skip)


class AAMBlock(nn.Module):
    def __init__(self, in_channels_low: int, in_channels_high: int):
        super(AAMBlock, self).__init__()

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_low, in_channels_high, kernel_size=1, padding="same"),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_high, in_channels_high, kernel_size=1, padding="same"),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels_high, in_channels_high, kernel_size=1, padding="same"),
            nn.Softmax(dim=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels_low, in_channels_high, kernel_size=1, padding="same"),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_low, x_high):
        g_x_low = self.global_avg_pooling(x_low)
        g_x_high = self.global_avg_pooling(x_high)

        conv_g_x_low = self.conv1(g_x_low)
        conv_g_x_high = self.conv2(g_x_high)

        weight_map = self.conv3(conv_g_x_low + conv_g_x_high)
        conv_x_low = self.conv4(x_low)
        x = conv_x_low.mul(weight_map) + x_high
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels_low: int, in_channels_high: int):
        super(DecoderBlock, self).__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels_low, in_channels_high, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels_high)
        )
        self.attention = AAMBlock(in_channels_low, in_channels_high)

    def forward(self, x_low: torch.Tensor, x_high: torch):
        x = self.attention(self.up1(x_low), x_high)
        return x


class RAUNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(RAUNet, self).__init__()

        self.e1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.e2 = EncoderBlock(in_channels=64, out_channels=128)
        self.e3 = EncoderBlock(in_channels=128, out_channels=256)

        self.b1 = ConvBlock(in_channels=256, out_channels=512)

        self.d1 = DecoderBlock(in_channels_low=512, in_channels_high=256)
        self.d2 = DecoderBlock(in_channels_low=256, in_channels_high=128)
        self.d3 = DecoderBlock(in_channels_low=128, in_channels_high=64)

        self.final_conv = ConvBlock(in_channels=64, out_channels=out_channels)

    def forward(self, x: torch.Tensor):
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)

        x = self.b1(x)

        d1 = self.d1(x, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        return self.final_conv(d3)


if __name__ == '__main__':
    model = RAUNet(in_channels=1, out_channels=1)
    utils.get_model_summary(model, input_size=(1, 256, 256))
