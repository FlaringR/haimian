import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Any
from models import BaseModel
from models.common.heads import MLPMultiHead
from models.common.embeddings import NumericalCategoricalEmbeddingLayer
import torch.nn.functional as F  

class TransformerEncoderLayerNoNorm(nn.Module):  
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):  
        super().__init__()  
        self.batch_first = batch_first  
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)  
        
        # 前馈网络部分  
        self.linear1 = nn.Linear(d_model, dim_feedforward)  
        self.dropout = nn.Dropout(dropout)  
        self.linear2 = nn.Linear(dim_feedforward, d_model)  
        
        if activation == "relu":  
            self.activation = F.relu  
        elif activation == "gelu":  
            self.activation = F.gelu  
        else:  
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")  

        # 这里不再定义 LayerNorm  
        
        self.dropout1 = nn.Dropout(dropout)  
        self.dropout2 = nn.Dropout(dropout)  

    def forward(self, src, src_mask=None, src_key_padding_mask=None,**kwargs):  
        # multihead attention  
        src2, _ = self.self_attn(src, src, src,  
                                 attn_mask=src_mask,  
                                 key_padding_mask=src_key_padding_mask)  
        src = src + self.dropout1(src2)  # 残差连接，没有norm  

        # feedforward  
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  
        src = src + self.dropout2(src2)  # 残差连接，没有norm  
        return src  


# 封装成 TransformerEncoder，保持接口不变，传入上面自定义的层  
def build_transformer_encoder_no_norm(config):  
    encoder_layer = TransformerEncoderLayerNoNorm(  
        d_model=config.embedding_dim,  
        nhead=config.transformer_nhead,  
        dim_feedforward=config.transformer_dim_feedforward,
        dropout=config.transformer_dropout,  
        activation=config.get("transformer_activation", "relu"),  
        batch_first=True  
    )  
    transformer = nn.TransformerEncoder(  
        encoder_layer,  
        num_layers=config.transformer_num_layers,  
        norm=None  # 最外层不加norm  
    )  
    return transformer  

class TransformerBackbone(nn.Module):
    def __init__(self, config: DictConfig):
        """Transformer 主干网络

        参数:
            config (DictConfig): 配置，包括嵌入维度、Transformer 参数等
        """
        super().__init__()
        self.config = config
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=config.embedding_dim,
        #         nhead=config.transformer_nhead,
        #         dim_feedforward=config.embedding_dim * 4,
        #         dropout=config.transformer_dropout,
        #         activation="relu",
        #         batch_first=True
        #     ),
        #     num_layers=config.transformer_num_layers,
        #     norm=None
        # )

        # 这里就没有使用norm层， 使用的自定义的层, 和官方实现的区别是不增加layernorm。 个人感觉数据量小，规律性弱（没有显著的batch_norm或者layer_norm的统计量）的数据和层数不深的网络没有必要使用这个。
        self.transformer = build_transformer_encoder_no_norm(config)
        # 输出维度：使用平均池化，保持 embedding_dim
        self.output_dim = config.embedding_dim
        # 替代方案：拼接（取消注释）
        # self.output_dim = config.embedding_dim * (config.continuous_dim + len(config.embedding_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, embedding_dim]

        返回:
            torch.Tensor: 输出张量 [batch_size, embedding_dim]
        """
        x = self.transformer(x)  # [batch_size, seq_len, embedding_dim]
        # 平均池化：聚合所有位置的嵌入
        x = torch.mean(x, dim=1)  # [batch_size, embedding_dim]
        # 替代方案：拼接
        # x = x.flatten(start_dim=1)  # [batch_size, seq_len * embedding_dim]
        return x

class TransformerModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        """Transformer 模型，继承自 BaseModel，实现解耦的嵌入层、主干网络和头部

        参数:
            config (DictConfig): 配置，包括输入维度、Transformer 参数、任务类型等
            **kwargs: 传递给 BaseModel 的额外参数
        """
        super().__init__(config, **kwargs)
        self.config = config
        self._build_network()

    def _build_network(self):
        """构建网络，包括嵌入层、主干网络和头部"""
        # 嵌入层
        self._embedding_layer = NumericalCategoricalEmbeddingLayer(
            continuous_dim=self.config.continuous_dim,
            categorical_embedding_dims=self.config.embedding_dims,
            embedding_dim=self.config.embedding_dim,
            embedding_dropout=self.config.embedding_dropout,
            batch_norm_continuous_input=self.config.batch_norm_continuous_input
        )
        # 主干网络
        self._backbone = TransformerBackbone(self.config)
        # 头部
        self._head = MLPMultiHead(input_dim=self._backbone.output_dim, output_dims=self.config.output_dims, layers=self.config.head_layers)

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

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """前向传播

        参数:
            batch: 包含 'continuous' 和 'categorical' 的字典

        返回:
            torch.Tensor: 模型输出 [batch_size, sum(output_dims)]
        """
        x = self.embedding_layer(batch)  # [batch_size, no_cat + no_num, embedding_dim]
        x = self.backbone(x)             # [batch_size, embedding_dim]
        return self.head(x)              # [batch_size, sum(output_dims)]