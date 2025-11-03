import torch
import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AccuracyMeter:
    def __init__(self, topk=(1, 5)):
        self.topk = topk
        self.reset()

    def reset(self):
        self.meters = {f'top{k}': AverageMeter() for k in self.topk}

    def update(self, output, target, n=1):
        acc = accuracy(output, target, self.topk)
        for i, k in enumerate(self.topk):
            self.meters[f'top{k}'].update(acc[i].item(), n)

    def get_avg(self):
        return {name: meter.avg for name, meter in self.meters.items()}

    def get_val(self):
        return {name: meter.val for name, meter in self.meters.items()}