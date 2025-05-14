from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from config import ModelConfig

@dataclass
class DLinearConfig(ModelConfig):
    """DLinear 模型配置类，继承 ModelConfig，添加 DLinear 独有参数"""
    kernel_size: int = 3                # 移动平均核大小，用于趋势提取， 3s一个tick做平滑
    individual: bool = True             # 是否为每个特征单独建模
    pred_len: int = 1                    # 预测长度，默认为单步预测, 有这个参数是为了和原始论文保持一致，确保为1就好。
    head_layers : str = ""
    _module_src: str = "models.dlinear"  # 模块路径
    _model_name: str = "DLinear"    # 模型类名

    @classmethod
    def from_file(cls, file_path: str):
        """从参数文件加载配置并构建 DLinearConfig"""
        config_dict = OmegaConf.load(file_path)  # 加载文件内容为 DictConfig
        return cls(**config_dict)  # 转换为 DLinearConfig 对象