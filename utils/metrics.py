import torch
import torch.nn as nn

class CustomMAE(nn.Module):
    def __init__(self, reduce_type="mean"):  # "mean" 或 "sum"
        super().__init__()
        self.reduce_type = reduce_type
        self.register_buffer("total_error", torch.tensor(0.0))
        self.register_buffer("total_samples", torch.tensor(0.0))

    def forward(self, pred, target):
        error = torch.abs(pred - target).sum()
        self.total_error += error
        self.total_samples += target.numel()# 返回每个batch中的元素数量
        return error / target.numel()  # 返回当前 batch 的 MAE

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(0.0)
        if self.reduce_type == "mean":
            return self.total_error / self.total_samples
        elif self.reduce_type == "sum":
            return self.total_error

    def reset(self):
        self.total_error.zero_()
        self.total_samples.zero_()

class CustomAccuracy(nn.Module):
    def __init__(self, num_classes, reduce_type="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduce_type = reduce_type
        self.register_buffer("correct", torch.tensor(0.0))
        self.register_buffer("total", torch.tensor(0.0))

    def forward(self, pred, target):
        pred_labels = pred.argmax(dim=-1)
        correct = (pred_labels == target).sum().float()
        self.correct += correct
        self.total += target.numel()
        return correct / target.numel()  # 当前 batch 的准确率

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        if self.reduce_type == "mean":
            return self.correct / self.total
        elif self.reduce_type == "sum":
            return self.correct  # 总数（正确预测数）

    def reset(self):
        self.correct.zero_()
        self.total.zero_()

class OnesCountMetric(nn.Module):
    def __init__(self, reduce_type="sum"):  # 默认求和
        super().__init__()
        self.reduce_type = reduce_type
        self.register_buffer("total_ones", torch.tensor(0.0))

    def forward(self, pred, _):
        ones = (pred.argmax(dim=-1) == 1).sum().float()
        self.total_ones += ones
        return ones  # 当前 batch 的计数

    def compute(self):
        if self.reduce_type == "sum":
            return self.total_ones

    def reset(self):
        self.total_ones.zero_()

class MeanForOnesMetric(nn.Module):
    def __init__(self, eval_target: str, reduce_type="mean"):  # 默认均值
        super().__init__()
        self.eval_target = eval_target
        self.reduce_type = reduce_type
        self.register_buffer("total_sum", torch.tensor(0.0))
        self.register_buffer("total_count", torch.tensor(0.0))

    def forward(self, pred, targets):
        predicted_as_one = pred.argmax(dim=-1) == 1
        if predicted_as_one.sum() > 0:
            values = targets[self.eval_target][predicted_as_one]
            self.total_sum += values.sum()
            self.total_count += predicted_as_one.sum()
            return values.mean()  # 当前 batch 的均值
        return torch.tensor(0.0, device=pred.device)

    def compute(self):
        if self.total_count == 0:
            return torch.tensor(0.0)
        if self.reduce_type == "mean":
            return self.total_sum / self.total_count
        
    def reset(self):
        self.total_sum.zero_()
        self.total_count.zero_()