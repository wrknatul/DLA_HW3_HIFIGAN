import torch
import torch.nn as nn

from .generator import Generator
from .mpd import MultiPeriodDiscriminator
from .msd import MultiScaleDiscriminator

class HiFiGANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.generator = Generator()
