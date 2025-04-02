from dataclasses import dataclass
from typing import List, Dict, Optional
from omegaconf import OmegaConf

@dataclass
class DataConfig:
    """数据配置类，用于管理 StockDataModule 的参数"""
    continuous_cols: List[str]           # 连续特征列名
    target_cols: List[str]               # 目标列名
    task_types: Dict[str, str]           # 任务类型字典
    categorical_cols: Optional[List[str]] = None  # 分类特征列名，可选
    category_col: str = "factor_0"        # 类别列名, 表示股票的分类信息的列，默认为factor_0
    target_category: int = 7              # 目标类别, 默认为第7类
    window_len: int = 1                   # 窗口长度
    padding_value: float = 0.0            # 填充值
    batch_size: int = 32                  # 批次大小
    split_ratio: float = 0.1              # 训练/验证分割比例（改为 0.2 表示 20% 验证集）
    split_type: str = "time"              # 分割类型："time" , "random", "random_time"
    split_start: float = 0.8              # 时间分割的起始点（当 split_type="time" 时使用）

    @classmethod
    def from_file(cls, file_path: str):
        """从参数文件加载配置并构建 DataConfig"""
        config_dict = OmegaConf.load(file_path)  # 加载文件内容为 DictConfig
        return cls(**config_dict)  # 转换为 DataConfig 对象

@dataclass
class TrainConfig:
    """训练配置类，用于管理 Trainer 的参数"""
    epochs: int = 1
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    @classmethod
    def from_file(cls, file_path: str):
        """从参数文件加载配置并构建 TrainConfig"""
        config_dict = OmegaConf.load(file_path)  # 加载文件内容为 DictConfig
        return cls(**config_dict)
    




