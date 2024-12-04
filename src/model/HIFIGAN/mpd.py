import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()
        self.period = period

        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))),
        ])

        self.post = nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        if x.size(2) % self.period != 0:
            pad_len = self.period - (x.size(2) % self.period)
            x = F.pad(x, (0, pad_len))

        x = x.reshape([batch_size, 1, x.size(2) // self.period, self.period])
        fmap = []
        for layer in self.layers:
            x = layer(x)
            x = self.leaky_relu(x)
            fmap.append(x)
        
        x = self.post(x)
        fmap.append(x)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(discriminator_size)
            for discriminator_size in [2, 3, 5, 7, 11]
        ])

    def forward(self, x):
        out_ = []
        fmap_ = []
        for disc in self.discriminators:
            out, fmap = disc(x)
            out_.append(out)
            fmap_.append(fmap)
        return out_, fmap_