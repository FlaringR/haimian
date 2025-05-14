import torch
import torch.nn as nn
from typing import Dict, Any
from omegaconf import DictConfig
from utils.nn import _initialize_layers, _initialize_layers_base
from models.base_model import BaseModel
from models.common.heads import MLPMultiHead
class DLinearBackbone(nn.Module):
    def __init__(self, config: DictConfig):
        """DLinear 主干网络，仅处理连续特征的时序数据。

        参数:
            config (DictConfig): 配置，包括输入维度、窗口长度、预测长度等。
        """
        super().__init__()
        self.config = config
        self.window_size = config.window_len
        self.feature_dim = config.continuous_dim  # 仅连续特征
        self.pred_len = config.get("pred_len", 1)  # 预测长度，默认为单步预测
        self.kernel_size = config.get("kernel_size", 3)  # 移动平均核大小
        self.individual = config.get("individual", True)  # 是否为每个特征单独建模

        self._build_network()

    def _build_network(self):
        """构建 DLinear 主干网络，使用线性层进行趋势和季节性分解。"""
        if self.individual:
            # 为每个特征单独建模
            self.trend_layers = nn.ModuleList([
                nn.Linear(self.window_size, self.pred_len) for _ in range(self.feature_dim)
            ])
            self.seasonal_layers = nn.ModuleList([
                nn.Linear(self.window_size, self.pred_len) for _ in range(self.feature_dim)
            ])
        else:
            # 统一建模
            self.trend_layer = nn.Linear(self.window_size, self.pred_len)
            self.seasonal_layer = nn.Linear(self.window_size, self.pred_len)

        # 移动平均用于趋势提取
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)

        # 初始化权重
        layers = self.trend_layers + self.seasonal_layers if self.individual else [self.trend_layer, self.seasonal_layer]
        for layer in layers:
            _initialize_layers(self.config.activation, self.config.initialization, layer)

        # 输出维度：预测长度 * 特征维度
        self.output_dim = self.pred_len * self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，处理时序数据。

        参数:
            x (torch.Tensor): 输入张量 [batch_size, window_size, continuous_dim]。

        返回:
            torch.Tensor: 主干网络输出 [batch_size, pred_len, continuous_dim]。
        """
        # 移动平均提取趋势
        trend_init = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, window_size, feature_dim]
        seasonal_init = x - trend_init  # 季节性 = 原始 - 趋势

        if self.individual:
            trend_out = []
            seasonal_out = []
            for i in range(self.feature_dim):
                trend = self.trend_layers[i](trend_init[:, :, i])  # [batch_size, pred_len]
                seasonal = self.seasonal_layers[i](seasonal_init[:, :, i])  # [batch_size, pred_len]
                trend_out.append(trend)
                seasonal_out.append(seasonal)
            trend_out = torch.stack(trend_out, dim=-1)  # [batch_size, pred_len, feature_dim]
            seasonal_out = torch.stack(seasonal_out, dim=-1)  # [batch_size, pred_len, feature_dim]
        else:
            trend_out = self.trend_layer(trend_init.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, pred_len, feature_dim]
            seasonal_out = self.seasonal_layer(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, pred_len, feature_dim]

        # 融合趋势和季节性
        output = trend_out + seasonal_out  # [batch_size, pred_len, feature_dim]
        return output



class DLinear(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        """DLinear 模型，继承自 BaseModel，仅处理连续特征，支持时序预测和多任务学习。

        参数:
            config (DictConfig): 配置，包括输入维度、窗口长度、预测长度、任务类型等。
            **kwargs: 传递给 BaseModel 的额外参数。
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._build_network()

    def _build_network(self):
        """构建网络，仅包括主干网络和头部（无嵌入层）。"""
        # 主干网络：DLinear 处理时序数据
        self._backbone = DLinearBackbone(self.config)
        # 头部：多任务输出
        self._head = MLPMultiHead(input_dim=self._backbone.output_dim * self._backbone.pred_len, output_dims=self.config.output_dims, layers=self.config.head_layers)

    @property
    def backbone(self):
        return self._backbone

    @property
    def head(self):
        return self._head

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向传播。

        参数:
            batch (Dict[str, Any]): 包含连续特征的批次数据。
                - continuous: [batch_size, window_size, continuous_dim]

        返回:
            Dict[str, torch.Tensor]: 多任务预测字典。
        """
        # 直接取连续特征, 不做embedding层
        x = batch["continuous"]  # [batch_size, window_size, continuous_dim]
        # 主干网络处理时序数据
        x = self.backbone(x)  # [batch_size, pred_len, continuous_dim]
        x = x.view(x.size(0), -1)
        # 头部生成多任务输出
        return self.head(x)