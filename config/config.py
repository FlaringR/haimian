from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from omegaconf import OmegaConf

@dataclass
class DataConfig:
    """数据配置类，用于管理 StockDataModule 的参数"""
    continuous_cols: List[str]           # 连续特征列名
    target_cols: List[str]               # 目标列名
    task_types: Dict[str, str]           # 任务类型字典
    metrics_target_cols: List[str]       # 评估指标的目标列名
    categorical_cols: Optional[List[str]] = None  # 分类特征列名，可选
    category_col: str = "factor_0"        # 类别列名, 表示股票的分类信息的列，默认为factor_0
    target_category: int = 7              # 目标类别, 默认为第7类
    time_col: str = "index"               # 时间列名, 表示数据中的时间列
    window_len: int = 1                   # 窗口长度
    padding_value: float = 0.0            # 填充值
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
    """训练配置类，用于管理Trainer的参数"""
    max_epochs: int = 20
    min_epochs: int = 2
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 1024
    patience: int = 3

   
    @classmethod
    def from_file(cls, file_path: str):
        """从参数文件加载配置并构建 TrainConfig"""
        config_dict = OmegaConf.load(file_path)  # 加载文件内容为 DictConfig
        return cls(**config_dict)
    
@dataclass
class ModelConfig:
    """模型配置类，用于管理base模型的公共参数, 包括训练和评估的管理"""
    embedding_dims: Optional[List] = None  # 分类特征的嵌入维度，例如 [(5, 3)] 表示 5 个类别嵌入为 3 维
    embedded_cat_dim: int = 0            # 嵌入后的分类特征总维度，默认为 0（无分类特征时）, 后续从InferredConfig中推断获取。
    embedding_dropout: float = 0.0       # 嵌入层的 Dropout 比率
    batch_norm_continuous_input: bool = False  # 是否对连续特征输入应用 BatchNorm
    learning_rate: float = 1e-3          # 学习率
    optimizer: str = "Adam"              # 优化器类型，例如 "Adam", "SGD"
    use_batch_norm: bool = False          # 是否在网络中使用 BatchNorm
    dropout: float = 0.1                 # Dropout 比率
    activation: str = "ReLU"             # 激活函数类型，例如 "ReLU", "LeakyReLU"
    initialization: str = "kaiming"      # 初始化方法，例如 "kaiming", "xavier"

    @classmethod
    def from_file(cls, file_path: str):
        """从参数文件加载配置并构建 ModelConfig"""
        config_dict = OmegaConf.load(file_path)  # 加载文件内容为 DictConfig
        return cls(**config_dict)  # 转换为 ModelConfig 对象

@dataclass
class InferredConfig:
    """从模型中推断出来的参数"""
    categorical_dim: int                   # 分类型特征维度
    continuous_dim: int                    # 连续特征维度
    output_dims: Dict[str, int]            # 输出维度 , 例如 {"y60_duo": 2}
    embedding_dims: Optional[List] = None  # 分类特征的嵌入维度，例如 [(5, 3)] 表示 5 个类别嵌入为 3 维
    embedded_cat_dim: int = 0              # 嵌入后的分类特征总维度，默认为 0（无分类特征时）


@dataclass
class ExperimentConfig:
    """实验配置类，用于管理实验对象的参数"""
    train_path: str = "/data/home/lichengzhang/zhoujun/HaimianData/20250325_split/train74_1_300750.SZ"
    test_path: str = "/data/home/lichengzhang/zhoujun/HaimianData/20250325_split/test74_1_300750.SZ"
    log_dir: str = "./logs"
    start_date: str = "20201230"
    end_date: str = "20230730"
    gpus: List[int] = field(default_factory=lambda: [0,1,2,3]) # dataclass中不支持可变参数， 用filed每次新开一个， 确保每个实例都有自己独立的列表。
    processes_per_gpu: int = 36
    seed: int = 42

