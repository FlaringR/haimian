import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from omegaconf import DictConfig
from utils.nn import _initialize_layers, _initialize_layers_base
from models.common.embeddings import Embedding1dLayer
from models.base_model import BaseModel

def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    """
    BN -> Linear -> Activation -> Dropout. 一个简单的单层网络。
    """
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        from models.common import BatchNorm1d

        layers.append(BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers

class MLPBackbone(nn.Module):
    def __init__(self, config: DictConfig):
        """MLP 主干网络。

        参数:
            config (DictConfig): 配置，包括输入维度、层结构等。
        """
        super().__init__()
        self.config = config
        self._build_network()

    def _build_network(self):
        """构建主干网络。"""
        layers = []
        # 输入维度：连续特征 + 嵌入后的分类特征
        input_dim = self.config.continuous_dim + self.config.embedded_cat_dim
        curr_units = input_dim

        for units in self.config.layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.config.activation,
                    self.config.initialization,
                    self.config.use_batch_norm,
                    curr_units,
                    int(units),
                    self.config.dropout,
                )
            )
            curr_units = int(units)
        
        self.linear_layers = nn.Sequential(*layers)
        self.output_dim = curr_units
        _initialize_layers(self.config.activation, self.config.initialization, self.linear_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x (torch.Tensor): 输入张量 [batch_size, input_dim]。

        返回:
            torch.Tensor: 主干网络输出 [batch_size, output_dim]。
        """
        return self.linear_layers(x)

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dims: Dict[str, int]):
        """MLP 头部，用于生成多任务的最终输出。

        参数:
            input_dim (int): 输入维度（主干网络输出维度）。
            output_dims (Dict[str, int]): 每个任务的输出维度，例如 {"y60_duo": 1, "y30_class": 3}。
        """
        super().__init__()
        self.output_dims = output_dims
        # 为每个任务创建一个独立的输出层
        self.output_layers = nn.ModuleDict({
            task_name: nn.Linear(input_dim, dim) for task_name, dim in output_dims.items()
        })
        # 初始化权重
        _initialize_layers_base(self)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回多任务预测字典。

        参数:
            x (torch.Tensor): 输入张量 [batch_size, input_dim]。

        返回:
            Dict[str, torch.Tensor]: 每个任务的预测张量字典，例如 {"y60_duo": tensor, "y30_class": tensor}。
        """
        # 为每个任务生成独立的预测，并以任务名称为键
        predictions = {}
        for task_name, layer in self.output_layers.items():
            pred = layer(x)
            if pred.shape[-1] == 1:
                predictions[task_name] = pred.squeeze(-1)
            else:
                predictions[task_name] = pred
                
        return predictions
    
class MLPModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        """MLP 模型，继承自 BaseModel，实现解耦的嵌入层、主干网络和头部。

        参数:
            config (DictConfig): 配置，包括输入维度、层结构、任务类型等。
            **kwargs: 传递给 BaseModel 的额外参数。
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._build_network()

    def _build_network(self):
        """构建网络，包括嵌入层、主干网络和头部。"""
        # 嵌入层
        self._embedding_layer = Embedding1dLayer(
            continuous_dim=self.config.continuous_dim,
            categorical_embedding_dims=self.config.embedding_dims,
            embedding_dropout=self.config.embedding_dropout,
            batch_norm_continuous_input=self.config.batch_norm_continuous_input,
        )
        # 主干网络
        self._backbone = MLPBackbone(self.config)
        # 头部
        self._head = MLPHead(self._backbone.output_dim, self.config.output_dims)

    @property
    def backbone(self):
        """获取主干网络。"""
        return self._backbone

    @property
    def embedding_layer(self):
        """获取嵌入层。"""
        return self._embedding_layer

    @property
    def head(self):
        """获取头部。"""
        return self._head

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """前向传播。

        参数:
            continuous (torch.Tensor): 连续特征张量 [batch_size, continuous_dim]。
            categorical (Optional[torch.Tensor]): 分类特征张量 [batch_size, num_cat_features]。

        返回:
            torch.Tensor: 模型输出 [batch_size, sum(output_dims)]。
        """
        x = self.embedding_layer(batch)
        x = self.backbone(x)
        return self.head(x)




