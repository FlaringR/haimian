import numpy as np
import torch
import pandas as pd 
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pandas import DataFrame
from typing import List, Dict, Optional, Union, Tuple
from omegaconf import DictConfig
import pytorch_lightning as pl
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from config import InferredConfig

class StockDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        continuous_cols: List[str],    # 连续特征列名
        target_cols: List[str],        # 目标列名, 例如[y60_duo]
        task_types: Dict[str, str],    # 任务类型字典，例如 {"y60_duo": "regression", "y60_duo": "classification"}
        metrics_target_cols: Optional[List[str]] = None,  # 用于统计标签的列名
        categorical_cols: Optional[List[str]] = None,  # 分类特征列名
        category_col: str = "factor_0",      # 类别列名
        target_category: int = 7,            # 目标类别（第7类）
        time_col: str = "index",             # 时间列名
        window_len: int = 1,                 # 窗口长度
        padding_value: float = 0.0,          # 不足窗口长度时的填充值
        weight_scheme: str = 'exp',         # 样本权重计算方式
        anchor_indices: Optional[List[int]] = None,     # 用于训练集和验证集的锚点。（因为是对部分数据做预测，例如第7类数据）
    ):
        """用于加载股票数据的 Dataset，支持时序、多任务学习和分类特征。

        参数:
            data (pd.DataFrame): 输入的 pandas DataFrame 数据，按时间顺序排列
            continuous_cols (List[str]): 连续特征列名列表
            target_cols (List[str]): 目标列名列表
            task_types (Dict[str, str]): 每个目标的任务类型
            mertrics_target_cols (List[str], optional): 用于统计标签的列名列表（）
            categorical_cols (List[str], optional): 分类特征列名列表（需预先序数编码）
            category_col (str): 类别列名，默认为 "category"
            target_category (int): 目标类别，默认为 7
            window_len (int): 时序窗口长度，默认为 1（单行）
            padding_value (float): 窗口不足时的填充值，默认为 0.0
            weight_scheme (str): 样本权重计算方式, 可选项为 "exp"、"linear"、"equal
        """
        self.data = data
        self.continuous_cols = continuous_cols if continuous_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.target_cols = target_cols
        self.task_types = task_types
        self.category_col = category_col
        self.target_category = target_category
        self.window_len = window_len
        self.padding_value = padding_value
        self.weight_scheme = weight_scheme
        
        # 指定可见的样本索引valid_indices，如果有外部指定，则直接使用，可用来划分训练和验证集。 
        if anchor_indices is not None:
            self.valid_indices = anchor_indices
        else:
            # 过滤第7类数据， valid_indices 
            self.valid_indices = self.data.index[self.data[category_col] == target_category].tolist()
        self.n = len(self.valid_indices)


        # 提取特征值
        self.continuous_X = self._get_feature_values(self.continuous_cols, np.float32)
        self.categorical_X = self._get_feature_values(self.categorical_cols, np.int64)

        # 提取训练目标, 设置为一个字典
        self.targets = {}
        for target_name in target_cols:
            target_data = self.data[target_name].astype(np.float32).values
            if self.task_types[target_name] == "classification":
                target_data = target_data.astype(np.int64)
            self.targets[target_name] = target_data 
        
        # 提取评估目标
        self.metrics_targets = {}
        for metrics_target_name in metrics_target_cols:
            self.metrics_targets[metrics_target_name] = self.data[metrics_target_name].astype(np.float32).values
        
        # 提取时间列时刻
        self.time_index = self.data.loc[self.valid_indices, time_col]

        # 计算样本权重
        self.weights = self.compute_weights()

    def _get_feature_values(self, cols: List[str], dtype: type) -> np.ndarray:
        """提取指定列的特征值。

        参数:
            cols (List[str]): 特征列名列表
            dtype (type): 数据类型（如 np.float32 或 np.int64）
        返回:
            np.ndarray: 特征值数组，若列为空则返回空数组
        """
        if cols:
            return self.data[cols].astype(dtype).values
        return np.array([])

    def compute_weights(self) -> np.ndarray:
        """根据不同的方案计算样本权重"""
        if self.weight_scheme == 'exp':
            weights = np.exp(np.linspace(-1, 0, self.n))
        elif self.weight_scheme == 'linear':
            weights = np.arange(1, self.n + 1)  # 线性递增，越新权重越高
        elif self.weight_scheme == 'equal':
            weights = np.ones(self.n)
        else:
            raise ValueError(f"Unsupported weight scheme: {self.weight_scheme}")
        # 归一化
        weights = weights / np.sum(weights)
        return weights

    def get_weights(self) -> np.ndarray:
        """返回样本权重"""
        return self.weights

    def _get_window_features(self, feature_data: np.ndarray, window_indices: List[int], feature_dim: int, dtype: torch.dtype) -> torch.Tensor:
        """提取窗口特征并转换为张量。

        参数:
            feature_data (np.ndarray): 特征数据数组
            window_indices (List[int]): 窗口索引列表
            feature_dim (int): 特征维度（列数）
            dtype (torch.dtype): 输出张量的数据类型（如 torch.float32 或 torch.long）
        返回:
            torch.Tensor: 窗口特征张量，若无特征则返回空张量
        """
        if feature_data.size == 0:  # 无特征时返回空张量
            return torch.tensor([])

        window_size = len(window_indices)
        if window_size < self.window_len:
            padding = np.full(
                (self.window_len - window_size, feature_dim),
                self.padding_value,
                dtype=feature_data.dtype,
            )
            window_data = np.vstack([padding, feature_data[window_indices]])
        else:
            window_data = feature_data[window_indices]

        tensor = torch.from_numpy(window_data).to(dtype)
        if self.window_len == 1:
            tensor = tensor.squeeze(0)  # 如果窗口长度为1，去掉时间维度
        return tensor

    def __len__(self) -> int:
        """返回第7类数据的样本总数"""
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个样本，支持时序窗口，连续特征和分类特征均应用窗口"""

        current_idx = self.valid_indices[idx]  # 当前第7类数据的索引

        # 计算窗口索引
        start_idx = max(0, current_idx - self.window_len + 1)  # 防止越界
        window_indices = list(range(start_idx, current_idx + 1))

        # 提取连续特征和分类特征的窗口
        continuous = self._get_window_features(
            self.continuous_X, window_indices, len(self.continuous_cols), torch.float32
        )
        categorical = self._get_window_features(
            self.categorical_X, window_indices, len(self.categorical_cols), torch.long
        )

        # 提取目标（仅当前行）
        targets = {}
        for target_name, target_data in self.targets.items():
            dtype = torch.long if self.task_types[target_name] == "classification" else torch.float32
            targets[target_name] = torch.tensor(target_data[current_idx], dtype=dtype)
        
        # 提取评估目标（仅当前行）
        metrics_targets = {}
        for mertrics_target_name, metrics_target_data in self.metrics_targets.items():
            metrics_targets[mertrics_target_name] = torch.tensor(metrics_target_data[current_idx], dtype=torch.float32)

        return {
            "continuous": continuous,    # (window_len, feature_dim) 或 (feature_dim) 或空张量
            "categorical": categorical,  # (window_len, categorical_dim) 或 (categorical_dim) 或空张量
            "targets": targets,           # 目标张量的字典
            "metrics_targets": metrics_targets,#用于评估张量的字典
        }

    def get_weights(self) -> np.ndarray:
        """返回样本权重"""
        return self.weights


class StockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: DataFrame,
        config: DictConfig,  # 接受 DictConfig
        validation: DataFrame = None,
        train_sampler: bool = True,
        copy_data: bool = True,
        verbose: bool = True,
    ):
        """股票数据的 PyTorch Lightning 数据模块。

        参数:
            train (DataFrame): 训练数据集
            config (DictConfig): 配置参数（OmegaConf 格式）
            validation (DataFrame, optional): 验证数据集，若为空则从 train 中分割
            train_sampler (True): 是否使用训练集采样器，默认使用, 可以获得线性权重等。
            copy_data (bool): 是否复制数据
            verbose (bool): 是否打印详细信息
        """
        super().__init__()
        self.train_data = train.copy() if copy_data else train
        self.val_data = validation.copy() if validation is not None and copy_data else validation
        self.config = config
        self.train_sampler = train_sampler
        self.verbose = verbose

        # 从 config 中提取参数
        self.continuous_cols = self.config.continuous_cols
        self.target_cols = self.config.target_cols
        self.task_types = self.config.task_types
        self.metrics_target_cols = self.config.metrics_target_cols
        self.categorical_cols = self.config.categorical_cols
        self.category_col = self.config.category_col
        self.target_category = self.config.target_category
        self.time_col = self.config.time_col
        self.window_len = self.config.window_len
        self.padding_value = self.config.padding_value
        self.batch_size = self.config.batch_size
        self.split_ratio = self.config.split_ratio
        self.split_type = self.config.split_type
        self.split_start = self.config.split_start # 当split_type为time的时候，必须要提供。

        if self.verbose:
            print(f"初始化 StockDataModule: split_type={self.split_type}, batch_size={self.batch_size}")

    def _split_train_val_indices(self, train: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """根据指定逻辑切分训练集和验证集的第7类样本锚点索引"""
        # 获取所有第7类样本的索引
        category_7_indices = train.index[train[self.category_col] == self.target_category].tolist()
        n_category_7 = len(category_7_indices)

        if self.split_type == "random":
            # 随机切分：在第7类样本中随机选择验证集锚点
            val_size = int(self.split_ratio * n_category_7)
            val_anchor_indices = np.random.choice(category_7_indices, size=val_size, replace=False).tolist()
            train_anchor_indices = [idx for idx in category_7_indices if idx not in val_anchor_indices]

        elif self.split_type == "time":
            # 时间顺序切分：从指定比例开始，选择验证集长度的样本
            if self.split_start is None:
                raise ValueError("split_type='time' 时必须提供 split_start 参数")
            if not 0 <= self.split_start <= 1:
                raise ValueError("split_start 必须在 0 到 1 之间")
            
            val_size = int(self.split_ratio * n_category_7)
            start_idx = int(self.split_start * n_category_7)  # 起始点
            end_idx = min(start_idx + val_size, n_category_7)  # 结束点不超过总数
            val_anchor_indices = category_7_indices[start_idx:end_idx]
            train_anchor_indices = [idx for idx in category_7_indices if idx not in val_anchor_indices]

        elif self.split_type == "random_time":
            # 随机时间切分：随机选择起始点，然后取验证集长度的样本
            val_size = int(self.split_ratio * n_category_7)
            max_start_idx = max(0, n_category_7 - val_size)  # 确保不超过边界
            start_idx = np.random.randint(0, max_start_idx + 1)  # 随机起始点
            val_anchor_indices = category_7_indices[start_idx:start_idx + val_size]
            train_anchor_indices = [idx for idx in category_7_indices if idx not in val_anchor_indices]

        else:
            raise ValueError(f"不支持的切分类型: {self.split_type}")

        return train_anchor_indices, val_anchor_indices

    def _split_train_val(self, train: pd.DataFrame) -> Tuple[StockDataset, StockDataset]:
        """切分训练集和验证集"""
        # 调用类内的切分方法
        train_anchor_indices, val_anchor_indices = self._split_train_val_indices(train)

        # 创建训练集和验证集的 StockDataset
        train_dataset = StockDataset(
            data=train,
            continuous_cols=self.continuous_cols,
            target_cols=self.target_cols,
            task_types=self.task_types,
            metrics_target_cols=self.metrics_target_cols,
            categorical_cols=self.categorical_cols,
            category_col=self.category_col,
            target_category=self.target_category,
            window_len=self.window_len,
            padding_value=self.padding_value,
            anchor_indices=train_anchor_indices,
        )
        val_dataset = StockDataset(
            data=train,
            continuous_cols=self.continuous_cols,
            target_cols=self.target_cols,
            task_types=self.task_types,
            metrics_target_cols=self.metrics_target_cols,
            categorical_cols=self.categorical_cols,
            category_col=self.category_col,
            target_category=self.target_category,
            window_len=self.window_len,
            padding_value=self.padding_value,
            anchor_indices=val_anchor_indices,
        )

        return train_dataset, val_dataset

    def setup(self, stage: Optional[str] = None):
        """划分数据集并创建训练和验证集"""
        if self.val_data is None:
            self.train_dataset, self.val_dataset = self._split_train_val(self.train_data)
        else:
            self.train_dataset = StockDataset(
                data=self.train_data,
                continuous_cols=self.continuous_cols,
                target_cols=self.target_cols,
                task_types=self.task_types,
                categorical_cols=self.categorical_cols,
                category_col=self.category_col,
                target_category=self.target_category,
                window_len=self.window_len,
                padding_value=self.padding_value,
            )
            self.val_dataset = StockDataset(
                data=self.val_data,
                continuous_cols=self.continuous_cols,
                target_cols=self.target_cols,
                task_types=self.task_types,
                categorical_cols=self.categorical_cols,
                category_col=self.category_col,
                target_category=self.target_category,
                window_len=self.window_len,
                padding_value=self.padding_value,
            )

        if self.verbose:
            print(f"训练集大小: {len(self.train_dataset)}, 验证集大小: {len(self.val_dataset)}")

    def train_dataloader(self):
        sampler, shuffle = self._get_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=0,
        )

    def _get_sampler(self):
        """创建采样器"""
        if self.train_sampler:
            weights = torch.from_numpy(self.train_dataset.get_weights()).float()
            sampler = WeightedRandomSampler(weights, len(self.train_dataset), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return sampler, shuffle
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def infer_config(self, config):
        #TODO:  增加连续特征的嵌入
        categorical_dim = len(config.categorical_cols) if config.categorical_cols else 0
        continuous_dim = len(config.continuous_cols)  if config.continuous_cols else 0 
        # 输出维度的字典， 例如{y60_duo, 2}
        output_dims = {}

        for target_name in config.target_cols:
            if config.task_types[target_name] == "regression":
                output_dims[target_name] = 1
            else:
                # classification, 找到各个类别的数量
                class_count = self.train_data[target_name].fillna("NA").nunique()   
                output_dims[target_name] = class_count
        
        # categorical_cardinality 各个分类型变量的类别数量
        categorical_cardinality = [(self.train_data[col].nunique())for col in config.categorical_cols]
        # embedding_dims
        if getattr(config, "embedding_dims", None) is not None:
            embedding_dims = config.embedding_dims
        else:
            embedding_dims = [(x, min(8, (x + 1) // 2)) for x in categorical_cardinality]
        # embedded_cat_dim 经过embedding层后的维度 
        embedded_cat_dim = sum(dim for _, dim in embedding_dims)

        return InferredConfig(
            categorical_dim = categorical_dim,
            continuous_dim = continuous_dim,
            output_dims=output_dims,
            embedding_dims = embedding_dims,
            embedded_cat_dim=embedded_cat_dim
        )

    def prepare_inference_dataloader(self, test: pd.DataFrame) -> DataLoader:
        """为推理准备 DataLoader。不依赖setup实现，防止数据泄露

        参数:
            test (pd.DataFrame): 测试数据集

        返回:
            DataLoader: 用于推理的 DataLoader
        """
        if self.verbose:
            print(f"Preparing inference DataLoader for test data with {len(test)} samples...")

        # 创建测试数据集，只包含第 7 类样本
        test_dataset = StockDataset(
            data=test,
            continuous_cols=self.continuous_cols,
            target_cols=self.target_cols,
            task_types=self.task_types,
            metrics_target_cols=self.metrics_target_cols,
            categorical_cols=self.categorical_cols,
            category_col=self.category_col,
            target_category=self.target_category,
            time_col=self.time_col,
            window_len=self.window_len,
            padding_value=self.padding_value,
            weight_scheme="equal",  # 推理时不需要加权，设为 equal
        )

        # 创建 DataLoader
        inference_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 推理时不打乱顺序
            num_workers=0,
        )

        # 测试样本的时间索引
        test_index = test_dataset.time_index

        if self.verbose:
            print(f"Inference DataLoader prepared with {len(test_dataset)} samples (category {self.target_category}).")
        return inference_loader, test_index