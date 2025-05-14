from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Callable, Dict, Any
from omegaconf import DictConfig
import torchmetrics
from utils.metrics import OnesCountMetric, MeanForOnesMetric, CustomAccuracy, CustomMAE
from functools import partial
from utils.logger import get_logger


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        custom_loss: Optional[torch.nn.Module] = None,
        custom_metrics: Optional[List[Callable]] = None,
        custom_metrics_prob_inputs: Optional[List[bool]] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = None,
        **kwargs,
    ):
        """基础模型类，为 PyTorch Lightning 提供标准化的训练和评估。

        参数:
            config (DictConfig): 模型的配置（例如任务类型、输入输出维度）。
            custom_loss (Optional[torch.nn.Module]): 自定义损失函数。如果为 None，则根据任务类型选择默认损失。
            custom_metrics (Optional[List[Callable]]): 自定义指标函数列表。
            custom_metrics_prob_inputs (Optional[List[bool]]): 指示每个指标是否需要概率输入的布尔值列表。
            custom_optimizer (Optional[torch.optim.Optimizer]): 自定义优化器实例或类。
            custom_optimizer_params (Dict): 优化器的参数（例如学习率、权重衰减）。
            **kwargs: 传递给模型的额外关键字参数。
        """
        super().__init__()
        # self.save_hyperparameters()  # 保存所有参数以便后续访问
        self.config = config
        self.task_types = config.task_types  # 任务类型字典，例如 {"y60_duo": "regression"}
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params or {"lr": 1e-3}
        self.kwargs = kwargs
        self.metrics_target_cols = config.metrics_target_cols 

        # 初始化损失函数
        self.loss_fn = self._init_loss(custom_loss)

        # 初始化指标
        self.train_metrics = self._init_metrics(custom_metrics, custom_metrics_prob_inputs)
        self.val_metrics = self._init_metrics(custom_metrics, custom_metrics_prob_inputs)
        self.test_metrics = self._init_metrics(custom_metrics, custom_metrics_prob_inputs)
        

    def _init_loss(self, custom_loss: Optional[torch.nn.Module]) -> nn.Module:
        """根据任务类型或自定义损失初始化损失函数。

        参数:
            custom_loss (Optional[torch.nn.Module]): 自定义损失函数。

        返回:
            nn.Module: 初始化后的损失函数。
        """
        
        # 默认损失函数根据任务类型选择, 选择第一个任务
        if len(self.task_types) == 1:
            task_type = list(self.task_types.values())[0]
            if task_type == "regression":
                return nn.MSELoss()
            elif task_type == "classification":
                return nn.CrossEntropyLoss()
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
        else:
            assert custom_loss is not None
            return custom_loss

    def _init_metrics(self, custom_metrics: Optional[List[Callable]], custom_metrics_prob_inputs: Optional[List[bool]]) -> Dict[str, Callable]:
        """初始化训练和验证的指标。

        参数:
            custom_metrics (Optional[List[Callable]]): 自定义指标函数列表。
            custom_metrics_prob_inputs (Optional[List[bool]]): 指示每个指标是否需要概率输入。

        返回:
            Dict[str, Callable]: 指标名称到指标函数的映射。用ModuleDict 进行封装， 防止计算的时候设备出错。
        """
        metrics = {}
        if custom_metrics is None:
            # 默认指标
            for target, task_type in self.task_types.items():
                if task_type == "regression":
                    metrics[f"{target}-mae"] = CustomMAE(reduce_type="mean")
                elif task_type == "classification":
                    metrics[f"{target}-accuracy"] = CustomAccuracy(num_classes=self.config.output_dims[target], reduce_type="mean")
                    metrics[f"{target}-count"] = OnesCountMetric()
                    # 为每一个统计目标生成均值的指标。 统计指标是y60. y120,y180,return_0类似这种。
                    for metrics_target_col in self.metrics_target_cols:
                        # 对每个统计目标进行参数绑定
                        metrics[f"{target}-{metrics_target_col}_mean"] = MeanForOnesMetric(metrics_target_col)
        else:
            # 自定义指标
            if custom_metrics_prob_inputs is None:
                custom_metrics_prob_inputs = [False] * len(custom_metrics)
            if len(custom_metrics) != len(custom_metrics_prob_inputs):
                raise ValueError("custom_metrics 和 custom_metrics_prob_inputs 的长度必须匹配")
            for i, (metric, prob_input) in enumerate(zip(custom_metrics, custom_metrics_prob_inputs)):
                metrics[f"metric_{i}"] = metric
        return nn.ModuleDict(metrics)
    
    @abstractmethod # 强迫所有子类必须实现这个方法
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """前向传播的抽象方法，子类必须实现。

        参数:
            continuous (torch.Tensor): 连续特征张量。
            categorical (Optional[torch.Tensor]): 分类特征张量。

        返回:
            torch.Tensor: 模型预测结果。
        """
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """标准化的训练步骤。

        参数:
            batch (Dict[str, Any]): 数据批次。
            batch_idx (int): 批次索引。

        返回:
            torch.Tensor: 损失值。
        """
        targets = batch["targets"]
        metrics_targets = batch["metrics_targets"]
        pred = self(batch)  # 前向传播

        # 计算损失
        loss = self._compute_loss(pred, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # 计算指标
        self._compute_metrics(pred, targets,  metrics_targets, stage="train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """标准化的验证步骤。

        参数:
            batch (Dict[str, Any]): 数据批次。
            batch_idx (int): 批次索引。
        """
        targets = batch["targets"]
        metrics_targets = batch["metrics_targets"]
        pred = self(batch)

        # 计算损失
        loss = self._compute_loss(pred, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # 计算指标
        metrics_dict = self._compute_metrics(pred, targets, metrics_targets, stage="val")

    def test_step(self, batch, batch_idx):
        """测试步骤，计算并记录评估指标"""
        pred = self(batch)
        targets = batch["targets"]
        metrics_targets = batch["metrics_targets"]
        loss = self._compute_loss(pred, targets)
        self.log("test_loss", loss, on_epoch=True)

        # 计算指标
        metrics_dict = self._compute_metrics(pred, targets, metrics_targets, stage="test")
    
    def on_test_epoch_end(self):
        """在训练结束的时候记录测试集指标"""
        logger = get_logger()
        metrics_dict = {name: metric.compute().item() for name, metric in self.test_metrics.items()}
        logger.log_metrics("test", metrics_dict, epoch=self.current_epoch)
        # 计算val_profit:
        total_profit = 0
        for target in self.config.target_cols:
            if "class" in target:
                return_mean = metrics_dict.get(f"{target}-return_0_mean", 0.0)
                return_count = metrics_dict.get(f"{target}-count", 0.0)
                profit = (return_mean - 0.0014) * return_count
                total_profit += profit
        self.log("test_profit", total_profit)
        logger.log_metrics("test", {"test_profit": total_profit}, epoch=self.current_epoch)
        for metric in self.test_metrics.values():
            metric.reset()
        
    def predict_step(self, batch, batch_idx):
        """预测步骤，返回字典形式的预测结果"""
        pred = self(batch)  # {"y60_duo": tensor, "y30_class": tensor}
        outputs = {}
        for target_name, pred_tensor in pred.items():
            if self.task_types[target_name] == "classification":
                outputs[target_name] = torch.softmax(pred_tensor, dim=-1)  # 转换为概率
            else:
                outputs[target_name] = pred_tensor  # 回归任务保持原始值
        return outputs  # Dict[str, torch.Tensor]
    
    
    def on_train_epoch_end(self):
        """在训练epoch结束的时候记录指标"""
        # 使用自定义的指标，实现了compute方法，在最后结束的时候会自动计算
        logger = get_logger()
        metrics_dict = {name: metric.compute().item() for name, metric in self.train_metrics.items()}
        logger.log_metrics("train", metrics_dict, epoch=self.current_epoch)
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_end(self):
        """在验证epoch结束的时候记录指标"""
        logger = get_logger()
        metrics_dict = {name: metric.compute().item() for name, metric in self.val_metrics.items()}
        logger.log_metrics("val", metrics_dict, epoch=self.current_epoch)
        for metric in self.val_metrics.values():
            metric.reset()
        # 计算val_profit:
        total_profit = 0
        for target in self.config.target_cols:
            if "class" in target:
                return_mean = metrics_dict.get(f"{target}-return_0_mean", 0.0)
                return_count = metrics_dict.get(f"{target}-count", 0.0)
                profit = (return_mean - 0.0014) * return_count
                total_profit += profit
        self.log("val_profit", total_profit)
        logger.log_metrics("val", {"val_profit": total_profit}, epoch=self.current_epoch)

    def _compute_loss(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """根据任务类型计算损失。

        参数:
            pred (Dict[str, torch.Tensor]): 模型预测。
            targets (Dict[str, torch.Tensor]): 目标值字典。

        返回:
            torch.Tensor: 计算出的损失。
        """
        # 单任务，不需要自定义损失函数
        if len(self.task_types) == 1:
            target_name = list(self.task_types.keys())[0]
            target = targets[target_name]
            pred = preds[target_name]
            return self.loss_fn(pred, target)
        # 多任务，需要有自定义的损失函数
        else:
            return self.loss_fn(preds, targets)
            
    def _compute_metrics(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], metrics_targets: Dict[str, torch.Tensor], stage: str) -> Dict[str, float]:
        """计算并记录指标，支持多任务字典输出。

        参数:
            preds (Dict[str, torch.Tensor]): 模型预测字典，键为任务名称，值为预测张量。
            targets (Dict[str, torch.Tensor]): 目标值字典，键为任务名称，值为目标张量。
            metrics_targets (Dict[str, torch.Tensor]): 用于指标计算的额外目标字典（如 y60_duo）。
            stage (str): 训练阶段（"train" 或 "val"）。

        返回:
            Dict[str, float]: 计算出的指标字典。
        """
        # 根据 stage 选择 metrics 这里是引用， 会正确调用self.train_metrics
        metrics = getattr(self, f"{stage}_metrics")
        metric_dict = {}
        for metric_name, metric_fn in metrics.items():
            # 从指标名称中提取 task_name
            task_name = metric_name.split("-")[0]
            if task_name is None:
                continue  # 如果指标名称中没有任务名，跳过（理论上不应该发生）

            task_pred = preds[task_name]
            task_target = targets[task_name]

            # 根据指标类型选择参数
            if "mean" in metric_name:
                metric_value = metric_fn(task_pred, metrics_targets)
                
            else:
                # 问题：为什么 metric_value 是一个单 batch 的值，但 PL 最终记录的是整个 epoch 的汇总结果？
                # 答案：自定义了缓存区
                metric_value = metric_fn(task_pred, task_target)
            # # 注意
            # self.log(f"{stage}_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True)
            metric_dict[metric_name] = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
        return metric_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器。

        返回:
            Dict[str, Any]: 包含优化器的字典。
        """
        if self.custom_optimizer is not None:
            optimizer = self.custom_optimizer(self.parameters(), **self.custom_optimizer_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), **self.custom_optimizer_params)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }}
