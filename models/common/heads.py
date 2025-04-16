import torch
import torch.nn as nn
from typing import Dict, Any
from utils.nn import _initialize_layers, _linear_dropout_bn, _initialize_layers_base

class BaseHead(nn.Module):
    """抽象头部基类，定义多任务输出接口

    参数:
        input_dim: 输入维度（主干网络输出维度）
        output_dims: 每个任务的输出维度，例如 {"y60_duo": 1, "y30_class": 3}
    """
    def __init__(self, input_dim: int, output_dims: Dict[str, int]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self._build_network()
        _initialize_layers_base(self)

    def _build_network(self):
        """构建头部网络，子类需实现"""
        raise NotImplementedError("Subclasses must implement _build_network")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回多任务预测字典

        参数:
            x: 输入张量 [batch_size, input_dim]

        返回:
            Dict[str, torch.Tensor]: 每个任务的预测，例如 {"y60_duo": tensor, "y30_class": tensor}
        """
        raise NotImplementedError("Subclasses must implement forward")

class LinearHead(BaseHead):
    """线性头部，为每个任务生成简单的线性输出

    参数:
        input_dim: 输入维度（主干网络输出维度）
        output_dims: 每个任务的输出维度，例如 {"y60_duo": 1, "y60_class": 2}
    """
    def _build_network(self):
        """构建线性输出层"""
        self.output_layers = nn.ModuleDict({
            task_name: nn.Linear(self.input_dim, dim)
            for task_name, dim in self.output_dims.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回多任务预测

        参数:
            x: 输入张量 [batch_size, input_dim]

        返回:
            Dict[str, torch.Tensor]: 每个任务的预测
        """
        predictions = {}
        for task_name, layer in self.output_layers.items():
            pred = layer(x)
            predictions[task_name] = pred.squeeze(-1) if pred.shape[-1] == 1 else pred
        return predictions

class MLPMultiHead(BaseHead):
    """多层 MLP 头部，为每个任务生成 MLP 输出

    参数:
        input_dim: 输入维度（主干网络输出维度）
        output_dims: 每个任务的输出维度，例如 {"y60_duo": 1, "y30_class": 3}
        layers: MLP 层结构，例如 "64-32" 表示两层，分别为 64 和 32 单元
        activation: 激活函数，例如 "ReLU"
        dropout: Dropout 比率
        use_batch_norm: 是否使用 BatchNorm
        initialization: 初始化方法，例如 "kaiming"
    """
    def __init__(
        self,
        input_dim: int,
        output_dims: Dict[str, int],
        layers: str = "64-32",
        activation: str = "ReLU",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        initialization: str = "kaiming"
    ):
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.initialization = initialization
        super().__init__(input_dim, output_dims)

    def _build_network(self):
        """构建多任务 MLP 输出层"""
        self.output_layers = nn.ModuleDict()
        for task_name, output_dim in self.output_dims.items():
            # 为每个任务构建独立的 MLP
            layers = []
            curr_units = self.input_dim
            for units in self.layers.split("-"):
                layers.extend(
                    _linear_dropout_bn(
                        self.activation,
                        self.initialization,
                        self.use_batch_norm,
                        curr_units,
                        int(units),
                        self.dropout
                    )
                )
                curr_units = int(units)
            # 最后一层：映射到任务输出维度
            final_layer = nn.Linear(curr_units, output_dim)
            layers.append(final_layer)
            self.output_layers[task_name] = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回多任务预测

        参数:
            x: 输入张量 [batch_size, input_dim]

        返回:
            Dict[str, torch.Tensor]: 每个任务的预测
        """
        predictions = {}
        for task_name, layer in self.output_layers.items():
            pred = layer(x)
            predictions[task_name] = pred.squeeze(-1) if pred.shape[-1] == 1 else pred
        return predictions
    

    
