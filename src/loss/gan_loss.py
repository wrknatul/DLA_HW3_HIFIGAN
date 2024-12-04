import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def melSpectrogramLoss(self, target_mel, predict_mel):
        loss_f = nn.L1Loss()
        return loss_f(target_mel, predict_mel)

    def discriminatorLoss(self, disk_real_outputs, disk_generated_outputs):
        loss = 0.0
        for sub_dr, sub_dg in zip(disk_real_outputs, disk_generated_outputs):
            loss += torch.mean((sub_dr - 1.0) ** 2) + torch.mean(sub_dg ** 2)
        return loss

    def generatorLoss(self, disk_outputs):
        loss = 0.0
        for output in disk_outputs:
            loss += torch.mean((output - 1.0) ** 2)
        return loss

    def featureMatchingLoss(self, real_samples, generated_samples):
        loss = 0.0
        loss_f = nn.L1Loss()
        for real_sample, generated_sample in zip(real_samples, generated_samples):
            for real, generated in zip(real_sample, generated_sample):
                val = loss_f(real, generated)
                loss += val
        return loss