import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations=(1, 3, 5)):
        super(ResBlock, self).__init__()

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilations
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilations
            ]
        )

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(self.leaky_relu(x))
            xt = c2(self.leaky_relu(xt))
            x = xt + x
        return x


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations=(1, 3, 5)):
        super(MRF, self).__init__()

        self.resblocks = nn.ModuleList(
            [ResBlock(channels, k, dilations) for k in kernel_sizes]
        )

    def forward(self, x):
        return sum([block(x) for block in self.resblocks])


class Generator(nn.Module):
    def __init__(
        self,
        input_channels=80,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super(Generator, self).__init__()

        self.conv_pre = nn.Conv1d(
            input_channels, upsample_initial_channel, 7, padding=3
        )

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        curr_channels = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    curr_channels, curr_channels // 2, k, stride=u, padding=(k - u) // 2
                )
            )
            curr_channels //= 2
            self.mrfs.append(MRF(curr_channels, resblock_kernel_sizes))

        self.conv_post = nn.Conv1d(curr_channels, 1, 7, padding=3)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x[:, :, :-1]
        x = self.conv_pre(x)

        for up, mrf in zip(self.ups, self.mrfs):
            x = self.leaky_relu(up(x))
            x = mrf(x)

        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
