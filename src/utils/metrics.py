import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


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


class ComprehensiveMetrics:
    def __init__(self, num_classes=1000, topk=(1, 5)):
        self.num_classes = num_classes
        self.topk = topk
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.accuracy_meter = AccuracyMeter(self.topk)

    def update(self, output, target, n=1):
        # Update accuracy
        self.accuracy_meter.update(output, target, n)
        
        # Store predictions and targets for F1/Precision/Recall calculation
        _, pred = output.topk(1, 1, True, True)
        pred = pred.squeeze().cpu().numpy()
        target = target.cpu().numpy()
        
        if pred.ndim == 0:  # Handle single sample case
            pred = np.array([pred])
            target = np.array([target])
            
        self.all_predictions.extend(pred.tolist())
        self.all_targets.extend(target.tolist())

    def get_metrics(self):
        metrics = self.accuracy_meter.get_avg()
        
        if len(self.all_predictions) > 0:
            # Calculate F1, Precision, Recall (macro average for multiclass)
            try:
                precision = precision_score(self.all_targets, self.all_predictions, 
                                          average='macro', zero_division=0) * 100
                recall = recall_score(self.all_targets, self.all_predictions, 
                                    average='macro', zero_division=0) * 100
                f1 = f1_score(self.all_targets, self.all_predictions, 
                             average='macro', zero_division=0) * 100
                
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            except Exception as e:
                # Fallback if sklearn metrics fail
                print(f"Warning: Could not calculate F1/Precision/Recall: {e}")
                metrics.update({
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                })
        else:
            metrics.update({
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            })
            
        return metrics
