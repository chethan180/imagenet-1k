import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class MixupLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, input, y_a, y_b, lam):
        return lam * self.criterion(input, y_a) + (1 - lam) * self.criterion(input, y_b)


def get_criterion(config):
    if 'label_smoothing' in config['training'] and config['training']['label_smoothing'] > 0:
        return LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing'])
    else:
        return nn.CrossEntropyLoss()