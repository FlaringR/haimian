import os
from typing import Optional, Dict, Union, List
from pathlib import Path
import pytorch_lightning as pl
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from config import DataConfig
from utils.logger import get_logger
from data import StockDataModule
from models import BaseModel
import pandas 
from pandas import DataFrame
from utils.py_utils import getattr_nested
from utils.feature import rankic, rankic_v2
import inspect
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging


class HaimianModel:
    """系统的核心模型，负责整合数据、模型、训练器，提供训练、评估、预测和保存功能。

    参数:
        config (Optional[DictConfig]): 全局配置，包含所有子配置。
        data_config (Optional[Union[DataConfig, str]]): 数据配置对象或配置文件路径。
        model_config (Optional[Union[DictConfig, str]]): 模型配置对象或配置文件路径。
        trainer_config (Optional[Union[DictConfig, str]]): 训练器配置对象或配置文件路径。
        log_dir (str): 日志保存目录。
        model_callable (Optional[callable]): 自定义模型类，覆盖默认模型。
        model_state_dict_path (Optional[Union[str, Path]]): 预训练模型权重路径。
        verbose (bool): 是否打印详细信息。
    """
    def __init__(
        self,
        config: Optional[DictConfig] = None,
        data_config: Optional[Union[DataConfig, str]] = None,
        model_config: Optional[Union[DictConfig, str]] = None,
        trainer_config: Optional[Union[DictConfig, str]] = None,
        log_dir: str = "logs",
        model_callable: Optional[callable] = None,
        model_state_dict_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> None:
        
        self.verbose = verbose
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 加载配置
        self.config = self._load_config(config, data_config, model_config, trainer_config)
        print(self.config)

        # 初始化模型
        self._init_model_callabel(model_callable)

        if self.verbose:
            print(f"Log directory set to: {self.log_dir}")

        self.logger = get_logger()
        
    def _load_config(
        self,
        config: Optional[DictConfig],
        data_config: Optional[Union[DataConfig, str]],
        model_config: Optional[Union[DictConfig, str]],
        trainer_config: Optional[Union[DictConfig, str]]
    ) -> DictConfig:
        """加载或合并配置。"""
        if config is not None:
            return OmegaConf.load(config) if isinstance(config, str) else config
        
        data_config = OmegaConf.structured(data_config)
        model_config = OmegaConf.structured(model_config)
        trainer_config = OmegaConf.structured(trainer_config)

        merged_config = OmegaConf.create()
        configs = {"data_config": data_config, "model_config": model_config, "trainer_config": trainer_config}
        merged_configs = [OmegaConf.to_container(cfg) for cfg in configs.values() if cfg is not None]
        merged_config = OmegaConf.merge(*merged_configs)
        return merged_config

    def prepare_datamodule(self, train: DataFrame) -> StockDataModule:
        """给训练和验证准备数据模块"""

        if self.config.select_features:
            # 默认用第一个target进行筛选， 后续可以修改逻辑
            feature_selected = self.select_factors(train, self.config.target_cols[0])
            self.logger.info(f"Feature selected: {feature_selected}")
            self.config.continuous_col = feature_selected


        datamodule = StockDataModule(
            train=train,
            config=self.config,
            
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        return datamodule

    
    def prepare_model(self, datamodule: StockDataModule, loss_fn: Optional[torch.nn.Module] = None) -> BaseModel:
        """准备模型模块"""

        model = self.model_callable(config=self.config, datamodule=datamodule, custom_loss=loss_fn)
        return model  
    
    def prepare_trainer(self, save_dir) -> pl.Trainer:
        """准备训练器"""
        # 动态获取要传递给Trainer的参数
        trainer_sig = inspect.signature(pl.Trainer.__init__)
        trainer_args = [p for p in trainer_sig.parameters.keys() if p != "self"]
        trainer_args_config = {k: v for k, v in self.config.items() if k in trainer_args}

        # 添加 ModelCheckpoint 回调，保存最优模型
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.monitor,  # 监控验证损失
            dirpath=save_dir,        # 稍后在 fit() 中动态设置
            filename="best_model_epoch{epoch}",
            save_top_k=1,        # 只保存最好的 1 个模型
            mode=self.config.mode,          # 验证损失越小越好
            save_weights_only=True,  #只保存权重
        )
        # # 添加 EarlyStopping 回调，设置早停策略
        # early_stopping_callback = EarlyStopping(
        #         monitor="val_loss",      # 监控验证损失
        #         min_delta=0.00,          # 改进的最小阈值
        #         patience=self.config.patience,              # 在验证损失停止改善后等待的轮数
        #         verbose=True,            # 是否打印早停信息
        #         mode="min"               # 验证损失越小越好
        #     )
        # 使用profit
        early_stopping_callback = EarlyStopping(
                monitor=self.config.monitor,      # 监控验证损失
                min_delta=0.00,          # 改进的最小阈值
                patience=self.config.patience,              # 在验证损失停止改善后等待的轮数
                verbose=True,            # 是否打印早停信息
                mode=self.config.mode               # 验证损失越小越好
            )
        
        swa_callback = StochasticWeightAveraging(
                swa_epoch_start=0.8,  # 从 80% 的 epoch 开始平均
                swa_lrs=0.01,         # SWA 阶段的学习率
                annealing_epochs=5,   # 退火阶段
            )

        return pl.Trainer(
            devices=[0],
            callbacks=[checkpoint_callback, early_stopping_callback, ],
            default_root_dir=save_dir,
            **trainer_args_config)
    
    def _init_model_callabel(self, model_callable: Optional[callable]) -> BaseModel:
        """初始化模型， 获得所使用模型"""
        if model_callable is None:
            self.model_callable = getattr_nested(self.config._module_src, self.config._model_name)
        else:
            self.model_callable = model_callable

        return self.model_callable

    def fit(
        self, 
        train: pd.DataFrame, 
        validation: Optional[pd.DataFrame] = None, 
        loss_fn: Optional[torch.nn.Module] = None, 
        train_file_name: str = "train_data"):
        """训练模型。

        参数:
            train (pd.DataFrame): 训练数据。
            validation (Optional[pd.DataFrame]): 验证数据，默认为 None。
            loss_fn (Optional[torch.nn.Module]): 损失函数，覆盖配置中的值。
            train_file_name (str): 训练数据文件名。用于生成保存目录
        """
        if self.verbose:
            print("Preparing data and training...")
        
        # 准备数据
        self.datamodule = self.prepare_datamodule(train)

        # 更新参数, 这是一部分可以推断出来的参数，用于训练和模型的构建。
        inferred_config = self.datamodule.infer_config(self.config)
        inferred_config = OmegaConf.structured(inferred_config)
        self.config = OmegaConf.merge(self.config, OmegaConf.to_container(inferred_config))

        # 准备模型
        self.model = self.prepare_model(datamodule=self.datamodule, loss_fn=loss_fn)

        # 准备保存目录
        save_dir = os.path.join(self.log_dir, train_file_name)
        os.makedirs(save_dir, exist_ok=True)
        self.trainer = self.prepare_trainer(save_dir)
        

        # 设置检查点保存路径
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = save_dir
                callback.filename = "best_model"  # 固定文件名，避免版本号
                break

        # 训练
        # self.model.train()
        self.trainer.fit(self.model, self.datamodule)

        # 获取最优模型路径, 需要遍历是因为[<TQDMProgressBar>, <ModelCheckpoint>], pytorch lighting 可能会添加一个进度条的回调。
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                self.best_model_path = callback.best_model_path
                break
        
        self.logger.log_metrics("train_end", {"message": "Training completed"})
        if self.verbose:
            print("Training completed")

        # 保存所有
        self.save_model(save_dir)
        self.save_config(save_dir)
        self._save_logger(save_dir)
        if self.verbose:
            print(f"Training completed. Artifacts saved to {save_dir}")

    def evaluate(self, test: Optional[pd.DataFrame] = None) -> List[Dict]:
        """评估模型。返回各个指标的评估结果

        参数:
            test (Optional[pd.DataFrame]): 测试数据，默认为 None（使用训练时的测试集）。
        
        返回:
            List[Dict]: 评估结果。
        """
        if self.verbose:
            print("Evaluating model with best checkpoint...")
        
        if self.best_model_path:
            self.load_model_weights(self.best_model_path, continue_training=False)
        else:
            if self.verbose:
                print("Warning: No best model path found, using final trained model.")

        if test is not None:
            test_loader, time_index= self.datamodule.prepare_inference_dataloader(test)
        else:
            test_loader = self.datamodule.test_dataloader()
        
        results = self.trainer.test(self.model, dataloaders=test_loader, verbose=False)

        print(results)
        self.logger.log_metrics("eval", results[0])
        return results

    def predict(self, test: pd.DataFrame, train_file_name: str = "train_data") -> pd.DataFrame:
        if self.verbose:
            print("Generating predictions with best checkpoint...")
        
        if self.best_model_path:
            self.load_model_weights(self.best_model_path, continue_training=False)
        else:
            if self.verbose:
                print("Warning: No best model path found, using final trained model.")

        inference_dataloader, time_index = self.datamodule.prepare_inference_dataloader(test)
        predictions = self.trainer.predict(self.model, inference_dataloader)
        
        # predictions 是 List[Dict[str, torch.Tensor]]，处理成张量和动作
        pred_dict = {target_name: [] for target_name in self.config.target_cols}
        action_dict = {target_name: [] for target_name in self.config.target_cols if self.config.task_types[target_name] == "classification"}

        for batch_pred in predictions:  # 遍历每个批次的预测字典
            for target_name, pred_tensor in batch_pred.items():
                pred_dict[target_name].append(pred_tensor.cpu())
                if self.config.task_types[target_name] == "classification":
                    action = self.chooseaction_class(pred_tensor).cpu()
                    action_dict[target_name].append(action)

        # 拼接预测张量
        pred_tensors = {name: torch.cat(tensors, dim=0) for name, tensors in pred_dict.items()}
        action_tensors = {name: torch.cat(tensors, dim=0) for name, tensors in action_dict.items()}

        # 生成列名和数据
        output_data = []
        output_columns = []
        for target_name in self.config.target_cols:
            tensor = pred_tensors[target_name]
            if self.config.task_types[target_name] == "classification":
                # 分类任务：输出每个类别的概率
                num_classes = self.config.output_dims[target_name]
                for i in range(num_classes):
                    output_data.append(tensor[:, i].numpy())
                    output_columns.append(f"{target_name}_prob_{i}")
                # 添加动作列
                output_data.append(action_tensors[target_name].numpy())
                output_columns.append(f"{target_name}_action")
            else:
                # 回归任务：直接输出值
                output_data.append(tensor.numpy().flatten())
                output_columns.append(f"{target_name}_prediction")

        # 增加time_index索引列
        output_data.append(time_index)
        output_columns.append("Time")

        # 创建 DataFrame
        pred_df = pd.DataFrame(
            data=dict(zip(output_columns, output_data))
        )

        save_dir = os.path.join(self.log_dir, train_file_name)
        if save_dir:
            pred_file = os.path.join(save_dir, "predictions.csv")
            pred_df.to_csv(pred_file, index=False)
            self.logger.log_metrics("predict", {"message": "Prediction completed", "pred_file": pred_file})
            if self.verbose:
                print(f"Predictions saved to {pred_file}")
        return pred_df

    def chooseaction_class(self, probs: torch.Tensor) -> torch.Tensor:
        """根据分类任务的概率选择动作。

        参数:
            probs (torch.Tensor): 分类任务的概率张量，形状 [batch_size, num_classes]

        返回:
            torch.Tensor: 选择的动作（类别索引），形状 [batch_size]
        """
        return torch.argmax(probs, dim=-1)  # 默认选择概率最大的类别作为动作
    
    def load_model_weights(self, path: str, continue_training: bool = False) -> None:
        """加载模型权重或检查点。

        参数:
            path (str): 模型权重文件路径（.ckpt 或 .pt）
            continue_training (bool): 是否加载检查点并准备继续训练，默认为 False
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        file_ext = os.path.splitext(path)[1].lower()
        if self.trainer and self.trainer.strategy:
            device = self.trainer.strategy.root_device  # 动态获取 PL 的设备（例如 cuda:0 或 cpu）
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if file_ext == ".ckpt":
            # 加载 PyTorch Lightning 检查点
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            if "state_dict" not in checkpoint:
                raise ValueError(f"Checkpoint file {path} does not contain 'state_dict'")
            state_dict = checkpoint["state_dict"]
            if self.verbose:
                print(f"Loaded checkpoint from {path}. Keys: {list(state_dict.keys())[:5]}...")

            if continue_training:
                # 恢复完整检查点状态以继续训练
                self.trainer = pl.Trainer.resume_from_checkpoint(
                    path,
                    devices=[0],
                    default_root_dir=self.log_dir,
                    logger=self.trainer.logger,
                    callbacks=self.trainer.callbacks
                )
                self.best_model_path = path
                if self.verbose:
                    print(f"Resumed training state from checkpoint {path}. Epoch: {checkpoint.get('epoch', 0)}")
            else:
                # 仅加载模型权重
                try:
                    self.model.load_state_dict(state_dict)
                    if self.verbose:
                        print(f"Model weights loaded successfully from checkpoint {path}")
                except RuntimeError as e:
                    print(f"Error loading state_dict from checkpoint: {e}")
                    raise
        
        elif file_ext == ".pt":
            # 加载纯 state_dict
            state_dict = torch.load(path, map_location=device)
            if not isinstance(state_dict, dict):
                raise ValueError(f"File {path} does not contain a valid state_dict")
            if self.verbose:
                print(f"Loaded state_dict from {path}. Keys: {list(state_dict.keys())[:5]}...")

            try:
                self.model.load_state_dict(state_dict)
                if self.verbose:
                    print(f"Model weights loaded successfully from {path}")
            except RuntimeError as e:
                print(f"Error loading state_dict: {e}")
                raise

        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Expected .ckpt or .pt")
                             
    def save_model(self, dir: str) -> None:
        """保存模型到指定目录。

        参数:
            dir (str): 保存路径。
        """
        # 保存训练结束时的模型（可选，最优模型已由 checkpoint 保存）
        torch.save(self.model.state_dict(), os.path.join(dir, "final_model.pt"))
        if self.verbose:
            print(f"Final model saved to {os.path.join(dir, 'final_model.pt')}")
        self.logger.log_metrics("save", {"model_path": os.path.join(dir, "final_model.pt")})

    def save_config(self, dir: str) -> None:
        """保存配置文件到指定目录。

        参数:
            dir (str): 保存路径。
        """
        OmegaConf.save(self.config, os.path.join(dir, "config.yaml"))
        if self.verbose:
            print(f"Config saved to {os.path.join(dir, 'config.yaml')}")
        self.logger.log_metrics("save", {"config_path": os.path.join(dir, "config.yaml")})

    def _save_logger(self, dir: str) -> None:
        """保存日志文件"""
        # 假设 logger 是 HaimianLogger 实例，访问其内部的 logging.Logger
        if hasattr(self.logger, 'logger'):  # 检查是否是 HaimianLogger
            log_file = self.logger.logger.handlers[0].baseFilename if self.logger.logger.handlers else os.path.join(dir, "training.log")
        else:  # 如果直接是 logging.Logger
            log_file = self.logger.handlers[0].baseFilename if self.logger.handlers else os.path.join(dir, "training.log")
        
        # 把log_file转移到save_dir
        os.rename(log_file, os.path.join(dir, "training.log"))
        if self.verbose:
            print(f"Logger saved to {os.path.join(dir, 'training.log')}")

    def select_factors(self, train: pd.DataFrame, target_col: str) -> List[str]:
        """因子筛选逻辑，只对连续特征进行筛选。

        参数:
            train (pd.DataFrame): 训练数据，包含因子和目标变量。
            target_col (str): 目标变量列名。

        返回:
            List[str]: 筛选后的因子列名列表。
        """
        # 只对 continuous_cols 进行筛选
        factor_cols = [col for col in self.config.continuous_cols if col in train.columns]
        corr_threshold_lev1 = self.config.get('corr_threshold_lev1', 0.05)
        corr_threshold_lev2 = self.config.get('corr_threshold_lev2', 0.0)

        # 一级筛选：基于 rankic
        corr = train[factor_cols].apply(rankic, y_f=train[target_col])
        factor_cols_select = corr[corr > corr_threshold_lev1].sort_values(ascending=False).index.tolist()

        # 二级筛选：基于 rankic_v2
        corr2 = train[factor_cols].apply(rankic_v2, y_f=train[target_col])
        factor_cols_select2 = corr2[corr2 > corr_threshold_lev2].sort_values(ascending=False).index.tolist()

        # 合并筛选结果
        factor_cols_select = [col for col in factor_cols_select if col in factor_cols_select2]

        if not factor_cols_select:
            if self.verbose:
                print(f"Warning: No factors selected after screening. Check correlation thresholds or data.")
            return []

        if self.verbose:
            print(f"Selected factors: {factor_cols_select}")
        return factor_cols_select
