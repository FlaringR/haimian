import torch
import torch.nn as nn
from typing import List, Dict

class MultiTaskLoss(nn.Module):
    """多任务损失函数，计算各个任务的损失并加权相加。

    参数:
        task_types (Dict[str, str]): 任务类型字典。
        task_weights (Dict[str, float]): 每个任务的权重。
    """
    def __init__(self, task_types: Dict[str, str], task_weights: Dict[str, float]):
        super().__init__()
        self.task_types = task_types
        self.task_weights = task_weights
        self.task_losses = nn.ModuleDict({
            task_name: nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
            for task_name, task_type in task_types.items()
        })

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算多任务损失。

        参数:
            preds (Dict[str, torch.Tensor]): 每个任务的预测张量字典。
            targets (Dict[str, torch.Tensor]): 每个任务的目标张量字典。

        返回:
            torch.Tensor: 加权总损失。
        """
        if set(preds.keys()) != set(self.task_types.keys()):
            raise ValueError(f"预测任务 ({preds.keys()}) 与定义的任务 ({self.task_types.keys()}) 不匹配")

        total_loss = 0.0
        for task_name, task_type in self.task_types.items():
            pred = preds[task_name]
            target = targets[task_name]
            loss_fn = self.task_losses[task_name]
            if task_type == "regression":# 这里似乎可以删掉了
                pred = pred.view(-1)
                assert pred.shape == target.shape
                
            task_loss = loss_fn(pred, target)
            task_weight = self.task_weights.get(task_name, 1.0)
            total_loss += task_weight * task_loss

        return total_loss / len(self.task_types) if len(self.task_types) > 1 else total_loss