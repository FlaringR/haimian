from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from config import ModelConfig

@dataclass
class MLPConfig(ModelConfig):
    """MLP 模型配置类，继承 ModelConfig，添加 MLP 独有参数"""
    layers: str = "64-32"                # MLP 主干网络的层结构，例如 "64-32" 表示两层，分别为 64 和 32 单元

    _module_src: str = "models.mlp"
    _model_name: str = "MLPModel"
    
    @classmethod
    def from_file(cls, file_path: str):
        """从参数文件加载配置并构建 MLPConfig"""
        config_dict = OmegaConf.load(file_path)  # 加载文件内容为 DictConfig
        return cls(**config_dict)  # 转换为 MLPConfig 对象