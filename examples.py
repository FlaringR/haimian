#  测试通过案例
import pandas as pd
train = pd.read_feather('/data/home/lichengzhang/zhoujun/HaimianData/20250325_split/train74_5_300750.SZ/20200427.ftr')
test = pd.read_feather('/data/home/lichengzhang/zhoujun/HaimianData/20250325_split/test74_5_300750.SZ/20200427.ftr')

# Data Pre
cat_cols = []
num_cols = [f'factor_{i}' for i in range(1,113)]
train['y'] = train['y60_duo'].apply(lambda x: 1 if x > 0.0022 else 0)
test['y'] = test['y60_duo'].apply(lambda x: 1 if x > 0.0022 else 0)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Dict 
# 简单模型定义
class StockModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dims: Dict[str, int] = None):
        super().__init__()
        self.save_hyperparameters()
        
        # 定义网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 假设所有任务输出拼接
        )
        self.output_dims = output_dims
        self.loss_fn = nn.CrossEntropyLoss() # 假设分类任务

    def forward(self, continuous, categorical=None):
        # 假设只用连续特征
        return self.network(continuous)

    def training_step(self, batch, batch_idx):
        continuous = batch["continuous"]
        targets = batch["targets"]
        pred = self(continuous)
        loss = self.loss_fn(pred, targets["y"])  # 假设目标是 y60_duo

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        continuous = batch["continuous"]
        targets = batch["targets"]
        pred = self(continuous)
        loss = self.loss_fn(pred, targets["y"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

from config import DataConfig, TrainConfig
from data import StockDataModule
from omegaconf import OmegaConf 
from models.mlp import MLPModel, MLPConfig

config = DataConfig(
        categorical_cols=["factor_0"],
        continuous_cols=[ "factor_5", "factor_6"],
        target_cols=["y", "y60_duo"],
        task_types={"y": "classification", "y60_duo": "regression"},
        metrics_target_cols = ["y60_duo", "y120_duo", "y180_duo"],
        category_col="factor_0",
        target_category=7,
        window_len=1,
        padding_value=0.0,
        split_ratio=0.1,
        split_type="time",
        split_start=0.9
    )
trainer_config = TrainConfig(
    batch_size=256,
    max_epochs=10

)
models_config = MLPConfig(
    layers = "32-32"
)
config = OmegaConf.structured(config)
models_config = OmegaConf.structured(models_config)
trainer_config = OmegaConf.structured(trainer_config)

# 合并所有参数
config = OmegaConf.merge(
    OmegaConf.to_container(config), 
    OmegaConf.to_container(models_config),
    OmegaConf.to_container(trainer_config),
    )

# 数据模块
data_module = StockDataModule(
    train = train, 
    config = config, 
    validation = None, # 空的话从训练集中切分
    train_sampler = True, #使用
    verbose = True,
)
infer_config = data_module.infer_config(config)
infer_config = OmegaConf.structured(infer_config)
config = OmegaConf.merge(config, OmegaConf.to_container(infer_config))
data_module.setup() 
traindataset = data_module.train_dataset


print(traindataset[1])

from utils.loss import MultiTaskLoss
task_weights = {"y": 1.0, "y60_duo": 1.0}
# 自定义的损失函数
loss_fn = MultiTaskLoss(task_types=config.task_types, task_weights=task_weights)

# 模型模块 
model = MLPModel(config=config, custom_loss=loss_fn)

trainer = pl.Trainer(max_epochs=config.max_epochs, devices = [0], log_every_n_steps=1000)
trainer.fit(model, data_module)