#  测试通过案例
import pandas as pd
from utils.logger import HaimianLogger, initialize_logger
from utils.trainer import get_filelist, set_seed, generate_log_dir
import os
train = pd.read_feather('/data/home/lichengzhang/zhoujun/HaimianData/20250325_split/train74_1_300750.SZ/20200427.ftr')
test = pd.read_feather('/data/home/lichengzhang/zhoujun/HaimianData/20250325_split/test74_1_300750.SZ/20200427.ftr')
# Data Pre
cat_cols = []
num_cols = [f'factor_{i}' for i in range(1,113)]
train['y'] = train['y60_duo'].apply(lambda x: 1 if x > 0.0020 else 0)
test['y'] = test['y60_duo'].apply(lambda x: 1 if x > 0.0020 else 0)

from config import DataConfig, TrainConfig, ExperimentConfig
from data import StockDataModule
from omegaconf import OmegaConf 
from models.mlp import MLPModel, MLPConfig
from models.deepfm import DeepFM, DeepFMConfig
from utils.loss import MultiTaskLoss
from haimian_model import HaimianModel

data_config = DataConfig(
        categorical_cols=[],
        continuous_cols=[f"factor_{i}" for i in range(1,57)],
        target_cols=["y"],
        task_types={"y": "classification"},
        metrics_target_cols = ["y60_duo", "y120_duo", "y180_duo"],
        category_col="factor_0",
        target_category=7,
        window_len=1,
        padding_value=0.0,
        split_ratio=0.1,
        split_type="random",
        split_start=0.9
    )

model_config = DeepFMConfig(
    layers = "32-32"
)

trainer_config = TrainConfig(
    batch_size=256,
    max_epochs=20

)
exp_config = ExperimentConfig()
#自定义的损失函数
task_weights = {"y": 1.0, "y60_duo": 1.0}
# loss_fn = MultiTaskLoss(task_types=data_config.task_types, task_weights=task_weights)
train_file_name = "20200427"
model_type = model_config._model_name
log_save_dir = generate_log_dir(exp_config.log_dir, model_type=model_type)
logger = initialize_logger(os.path.join(log_save_dir, train_file_name))


haimian_model = HaimianModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        log_dir=log_save_dir,
        verbose=True
    )

haimian_model.fit(train=train, loss_fn=None, train_file_name=train_file_name)
haimian_model.predict(test, train_file_name=train_file_name)