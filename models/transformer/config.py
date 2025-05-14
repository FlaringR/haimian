from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from config import ModelConfig

@dataclass
class TransformerConfig(ModelConfig):
    """Transformer 模型配置类，继承 ModelConfig，添加 Transformer 独有参数"""
    head_layers: str = "32"                # 分类头，例如 "64-32" 表示两层，分别为 64 和 32 单元
    embedding_dim: int = 8                    # 嵌入维度（与 DNN 分类特征共享, 这样确保离散值嵌入的维度是一样的，可以二阶特征交互）

    transformer_nhead: int = 1                # 多头注意力机制的头数
    transformer_num_layers: int = 2           # transformer 层数
    transformer_dropout: float = 0.1          # transformer dropout
    transformer_dim_feedforward: int = 64     # transformer 前馈神经网络的维度
    _module_src: str = "models.transformer"   # 模块来源
    _model_name: str = "TransformerModel"     # 模型名称
    
    @classmethod
    def from_file(cls, file_path: str):
        """从配置文件加载参数并构建 TransformerConfig"""
        config_dict = OmegaConf.load(file_path)
        return cls(**config_dict)