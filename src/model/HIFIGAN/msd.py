import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm


class SubDiscriminator(nn.Module):
    def __init__(self, norm):
        super(SubDiscriminator, self).__init__()

        self.conv_layers = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        1,
                        128,
                        kernel_size=15,
                        stride=1,
                        groups=1,
                        dilation=1,
                        padding=7,
                    )
                ),
                norm(
                    nn.Conv1d(
                        128,
                        256,
                        kernel_size=41,
                        stride=4,
                        groups=4,
                        dilation=1,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        256,
                        512,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        dilation=1,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        512,
                        1024,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        dilation=1,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        1024,
                        1024,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        dilation=1,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        1024,
                        1024,
                        kernel_size=5,
                        stride=1,
                        groups=16,
                        dilation=1,
                        padding=2,
                    )
                ),
            ]
        )

        self.conv_post = norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        fmap = []

        for conv in self.conv_layers:
            x = conv(x)
            x = self.leaky_relu(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [
                SubDiscriminator(norm)
                for norm in [spectral_norm, weight_norm, weight_norm]
            ]
        )

        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out_ = []
        fmap_ = []

        for i, disc in enumerate(self.discriminators):
            if i != 0:
                out = self.avgpool(x)
            else:
                out = x
            out, fmap = disc(out)
            out_.append(out)
            fmap_.append(fmap)

        return out_, fmap_
