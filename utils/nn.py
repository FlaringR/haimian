import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from omegaconf import DictConfig
import math 

def _initialize_layers_base(layers: nn.Module) -> None:
    """初始化网络层的权重和偏置。

    参数:
        layers (nn.Module): 要初始化的网络层，可以是单个层或包含多个层的模块。

    返回:
        None: 该函数直接修改输入层的权重和偏置。
    """
    for m in layers.modules():  # 遍历所有子模块
        if isinstance(m, nn.Linear):  # 如果是线性层
            # 使用 Kaiming Uniform 初始化权重
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:  # 如果有偏置
                # 计算 fan_in，用于偏置的均匀初始化
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):  # 如果是 BatchNorm 层
            # BatchNorm 的权重初始化为 1，偏置初始化为 0
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _initialize_layers(activation, initialization, layers):
    """对所有可能的激活层，初始化层等做归一化。

    """
    if type(layers) is nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                # logger.warning("Kaiming initialization is only recommended for ReLU and" " LeakyReLU.")
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=(nn.init.calculate_gain(nonlinearity) if activation in ["ReLU", "LeakyReLU"] else 1),
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)


def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        from pytorch_tabular.models.common.layers.batch_norm import BatchNorm1d

        layers.append(BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers