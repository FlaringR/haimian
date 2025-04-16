import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from utils.nn import _linear_dropout_bn, _initialize_layers
from models.common.heads import LinearHead
from omegaconf import DictConfig
from models.base_model import BaseModel

class DeepFMEmbeddingLayer(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Optional[List[Tuple[int, int]]],
        embedding_dropout: float,
        fm_embedding_dim: int,
        batch_norm_continuous_input: bool
    ):
        """DeepFM 嵌入层，FM 和 DNN 共享离散特征嵌入

        参数:
            continuous_dim: 连续特征维度
            categorical_embedding_dims: 分类特征嵌入配置，例如 [(5, 8), (10, 8)]
            fm_embedding_dim: FM 和 DNN 共享的嵌入维度
            embedding_dropout: Dropout 比率
            batch_norm_continuous_input: 是否对连续特征应用 BatchNorm
        """
        super().__init__()
        self.continuous_dim = continuous_dim

        # 共享嵌入层（FM 二阶和 DNN）
        self.shared_embedding = nn.ModuleList()
        self.fm_first_order = nn.ModuleList()  # 一阶嵌入仍独立
        total_embed_dim = continuous_dim
        self.fm_embedding_dim = fm_embedding_dim

        # 连续特征：仅一阶
        self.fm_continuous_linear = nn.Linear(continuous_dim, 1) if continuous_dim > 0 else None

        # 离散特征：共享二阶和 DNN 嵌入
        if categorical_embedding_dims:
            for n_categories, _ in categorical_embedding_dims:
                self.fm_first_order.append(nn.Embedding(n_categories, 1))  # 一阶：标量
                self.shared_embedding.append(nn.Embedding(n_categories, fm_embedding_dim))  # 共享：向量
                total_embed_dim += fm_embedding_dim

        self.dropout = nn.Dropout(embedding_dropout) if embedding_dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(continuous_dim) if batch_norm_continuous_input and continuous_dim > 0 else None
        self.total_embed_dim = total_embed_dim

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向传播，返回 FM 和 DNN 嵌入

        参数:
            batch: 包含 'continuous' 和 'categorical' 的字典

        返回:
            字典，包含：
                - 'dnn': DNN 输入 [batch_size, continuous_dim + embedded_cat_dim]
                - 'fm_first': FM 一阶项 [batch_size, num_features]
                - 'fm_second': FM 二阶嵌入 [batch_size, num_features, fm_embedding_dim]
        """
        continuous = batch.get("continuous", torch.tensor([]))
        categorical = batch.get("categorical", torch.tensor([]))
        
        # DNN 嵌入
        dnn_inputs = []
        if continuous.size(-1) > 0:
            dnn_cont = continuous
            if self.batch_norm:
                dnn_cont = self.batch_norm(dnn_cont)
            dnn_inputs.append(dnn_cont)
        
        # FM 嵌入
        fm_first = []
        fm_second = []

        # 连续特征：一阶
        if continuous.size(-1) > 0 and self.fm_continuous_linear:
            fm_first.append(self.fm_continuous_linear(continuous))
        
        # 离散特征：共享嵌入
        if categorical.size(-1) > 0 and len(self.fm_first_order) > 0:
            for i, emb_first in enumerate(self.fm_first_order):
                emb_shared = self.shared_embedding[i]
                cat_feature = categorical[:, i]
                fm_first.append(emb_first(cat_feature))  # 一阶：标量
                fm_second.append(emb_shared(cat_feature))  # 二阶：向量
                dnn_inputs.append(emb_shared(cat_feature))  # DNN：向量
        
        # DNN 输入：拼接连续和离散特征
        dnn_input = torch.cat(dnn_inputs, dim=-1) if dnn_inputs else torch.tensor([], device=continuous.device)
        
        # FM 输入：拼接一阶和二阶
        fm_first = torch.cat(fm_first, dim=-1) if fm_first else torch.tensor([], device=continuous.device)
        fm_second = torch.stack(fm_second, dim=1) if fm_second else torch.tensor([], device=continuous.device)
        
        # Dropout
        if self.dropout:
            dnn_input = self.dropout(dnn_input)
            fm_first = self.dropout(fm_first)
            fm_second = self.dropout(fm_second)
        
        return {
            "dnn": dnn_input,
            "fm_first": fm_first,
            "fm_second": fm_second
        }

class DeepFMBackbone(nn.Module):
    def __init__(self, config: DictConfig):
        """DeepFM 主干网络，结合 FM 和 DNN 组件

        参数:
            config: DeepFMConfig 配置对象
        """
        super().__init__()
        self.config = config
        self._build_network()

    def _build_network(self):
        """构建 FM 和 DNN 组件"""
        layers = []
        # DNN 输入维度：连续特征 + 嵌入后的分类特征
        input_dim = self.config.continuous_dim + self.config.embedded_cat_dim
        curr_units = input_dim

        # 构建 DNN 层
        for units in self.config.layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.config.activation,
                    self.config.initialization,
                    self.config.use_batch_norm,
                    curr_units,
                    int(units),
                    self.config.dropout
                )
            )
            curr_units = int(units)
        
        self.dnn = nn.Sequential(*layers) if layers else nn.Identity()
        self.fm_dropout = nn.Dropout(self.config.fm_dropout) if self.config.fm_dropout > 0 else None
        # 输出维度：DNN 输出 + FM 输出（标量）
        self.output_dim = curr_units + 1

        # 初始化权重
        _initialize_layers(self.config.activation, self.config.initialization, self.dnn)

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播，结合 FM 和 DNN 输出

        参数:
            embeddings: DeepFMEmbeddingLayer 输出的字典，包含 'dnn', 'fm_first', 'fm_second'

        返回:
            torch.Tensor: 拼接后的输出 [batch_size, dnn_output_dim + 1]
        """
        # DNN 前向
        dnn_output = self.dnn(embeddings["dnn"]) if self.dnn else embeddings["dnn"]
        
        # FM 前向
        fm_first = embeddings["fm_first"]  # (batch_size, num_features)
        fm_second = embeddings["fm_second"]  # (batch_size, num_features, fm_embedding_dim)
        
        # 一阶项：直接求和
        fm_first_sum = fm_first.sum(dim=-1, keepdim=True) if fm_first.size(-1) > 0 else torch.zeros_like(dnn_output[:, :1])
        
        # 二阶项：sum(square(x)) - square(sum(x))
        if fm_second is not None and fm_second.ndimension() > 1 and fm_second.size(1) > 0:
            sum_emb = fm_second.sum(dim=1)  # (batch_size, fm_embedding_dim)
            sum_square = (fm_second ** 2).sum(dim=1)  # (batch_size, fm_embedding_dim)
            fm_second_term = 0.5 * (sum_emb ** 2 - sum_square).sum(dim=-1, keepdim=True)
        else:
            fm_second_term = torch.zeros_like(fm_first_sum)
        
        fm_output = fm_first_sum + fm_second_term
        
        # 应用 FM Dropout
        if self.fm_dropout:
            fm_output = self.fm_dropout(fm_output)
        
        # 拼接 DNN 和 FM 输出
        return torch.cat([dnn_output, fm_output], dim=-1)

class DeepFM(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        """DeepFM 模型，集成嵌入层、主干网络和头部

        参数:
            config: DeepFMConfig 配置对象
            **kwargs: 传递给 BaseModel 的额外参数
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._build_network()

    def _build_network(self):
        """构建网络，包括嵌入层、主干网络和头部"""
        # 为 FM 重写 embedding_dims，确保与 fm_embedding_dim 一致
        if self.config.embedding_dims:
            self.config.embedding_dims = [(n, self.config.fm_embedding_dim) for n, _ in self.config.embedding_dims]
            self.config.embedded_cat_dim = sum(self.config.fm_embedding_dim for _, _ in self.config.embedding_dims)
        
        # 嵌入层
        self._embedding_layer = DeepFMEmbeddingLayer(
            continuous_dim=self.config.continuous_dim,
            categorical_embedding_dims=self.config.embedding_dims,
            fm_embedding_dim=self.config.fm_embedding_dim,
            embedding_dropout=self.config.embedding_dropout,
            batch_norm_continuous_input=self.config.batch_norm_continuous_input
        )
        # 主干网络
        self._backbone = DeepFMBackbone(self.config)
        # 头部
        self._head = LinearHead(self._backbone.output_dim, self.config.output_dims)

    @property
    def backbone(self):
        """获取主干网络"""
        return self._backbone

    @property
    def embedding_layer(self):
        """获取嵌入层"""
        return self._embedding_layer

    @property
    def head(self):
        """获取头部"""
        return self._head

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向传播

        参数:
            batch: 包含 'continuous' 和 'categorical' 张量的字典

        返回:
            Dict[str, torch.Tensor]: 多任务预测结果
        """
        embeddings = self.embedding_layer(batch)
        x = self.backbone(embeddings)
        return self.head(x)



