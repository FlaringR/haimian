from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from config import ModelConfig

@dataclass
class DeepFMConfig(ModelConfig):
    """DeepFM 模型配置类，继承 ModelConfig，添加 DeepFM 独有参数"""
    layers: str = "64-32"                # DNN 主干网络层结构，例如 "64-32" 表示两层，分别为 64 和 32 单元
    fm_embedding_dim: int = 8            # FM 组件的嵌入维度（与 DNN 分类特征共享, 这样确保离散值嵌入的维度是一样的，可以二阶特征交互）
    fm_dropout: float = 0.0              # FM 输出层的 Dropout 比率

    _module_src: str = "models.deepfm"   # 模块来源
    _model_name: str = "DeepFM"     # 模型名称
    
    @classmethod
    def from_file(cls, file_path: str):
        """从配置文件加载参数并构建 DeepFMConfig"""
        config_dict = OmegaConf.load(file_path)
        return cls(**config_dict)