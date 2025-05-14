import pandas as pd
from functools import partial
import multiprocessing as mp
from config import DataConfig, TrainConfig, ExperimentConfig, ModelConfig
from typing import List, Any, Dict, Optional
from models.mlp import MLPModel, MLPConfig
from models.deepfm import DeepFM, DeepFMConfig
from models.transformer import TransformerConfig
from models.dlinear import DLinear, DLinearConfig
from utils.loss import MultiTaskLoss
from haimian_model import HaimianModel
from utils.trainer import get_filelist, set_seed, generate_log_dir
from utils.logger import HaimianLogger, initialize_logger
from utils.data_preprocess import preprocess_classification_labels
import os
import argparse


def train_and_predict_on_file(
    train_file: str,
    test_path: str,
    log_save_dir: str,
    data_config: DataConfig,
    model_config: ModelConfig,
    trainer_config: TrainConfig,
    seed: int
):
    """单个文件的训练和预测函数"""
    try:
        # 获得训练文件名
        train_file_name = os.path.basename(train_file).replace(".ftr", "")

        # 设置随机种子（每个进程独立设置）
        set_seed(seed)

        # 每个进程独立创建logger
        logger = initialize_logger(os.path.join(log_save_dir, train_file_name))

        test_file = os.path.join(test_path, os.path.basename(train_file))
        if not os.path.exists(test_file):
            return

        # 加载数据
        train = pd.read_feather(train_file)
        test = pd.read_feather(test_file)

        # 数据预处理
        train = preprocess_classification_labels(train, data_config.threshold_map)
        test = preprocess_classification_labels(test, data_config.threshold_map)
        # 创建模型
        haimian_model = HaimianModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            verbose=True,
            log_dir=log_save_dir,  # 统一日志目录
        )

        # 自定义损失函数
        task_weights = {"y60_duo_class": 1.0, "y60_duo": 1.0}
        loss_fn = MultiTaskLoss(task_types=data_config.task_types, task_weights=task_weights)

        # 训练和预测
        haimian_model.fit(train=train, loss_fn=loss_fn, train_file_name=train_file_name)
        # 评估
        haimian_model.evaluate(test=test)
        # 保存
        haimian_model.predict(test, train_file_name=train_file_name)
        
    except Exception as e:
        print(e)

def train_files_on_gpu(
    file_subset: List[str],
    gpu_id: int,
    log_save_dir: str,
    exp_config: ExperimentConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    trainer_config: TrainConfig,
):
    """在单个 GPU 上并行处理文件子集"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_files = len(file_subset)
    if num_files == 0:
        return

    processes_per_gpu = min(exp_config.processes_per_gpu, num_files)
    
    with mp.Pool(processes=processes_per_gpu) as pool:
        from functools import partial
        train_model_partial = partial(
            train_and_predict_on_file,
            test_path=exp_config.test_path,
            log_save_dir=log_save_dir,
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            seed=exp_config.seed  # 传递种子
        )
        # 使用 imap 保证文件顺序
        pool.imap(train_model_partial, file_subset)
        pool.close()
        pool.join()

def run_training(
    exp_config: ExperimentConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    trainer_config: TrainConfig, 
):
    """运行多 GPU 并行训练"""
    # 设置全局种子（主进程）
    set_seed(exp_config.seed)
    
    # 获得需要处理的文件
    filelist = get_filelist(exp_config.train_path, exp_config.start_date, exp_config.end_date)
    num_files = len(filelist)
    print(f"Found {num_files} files to process")

    # 给每个GPU分配文件
    num_gpus = len(exp_config.gpus)
    filelist_split = [filelist[i::num_gpus] for i in range(num_gpus)]

    for i, sublist in enumerate(filelist_split):
        print(f"GPU {exp_config.gpus[i]} will process {len(sublist)} files")
    
    # 创建日志目录，每次运行单独创建
    model_type = model_config._model_name
    log_save_dir = generate_log_dir(exp_config.log_dir, model_type=model_type)
    print("log_save_dir: ", log_save_dir)
    mp.set_start_method("spawn", force=True)
    processes = []

    for i, gpu_id in enumerate(exp_config.gpus):
        process = mp.Process(
            target=train_files_on_gpu,
            args=(
                filelist_split[i],
                gpu_id,
                log_save_dir, 
                exp_config,
                data_config,
                model_config,
                trainer_config
            )
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All training and prediction tasks completed")

def main():
    """主函数，支持参数遍历"""
    # 定义默认配置
    exp_config = ExperimentConfig(
        seed=42,
        gpus=[0,1,2,3],
        processes_per_gpu=32,
    )  

    data_config = DataConfig(
        categorical_cols=[],
        continuous_cols=[f"factor_{i}" for i in range(1,57)],
        # continuous_cols=[f'factor_{i}' for i in range(1, 33)]+[f'factor_{i}' for i in range(41, 57)],
        target_cols=["y60_duo_class"],
        task_types={"y60_duo_class": "classification"},
        metrics_target_cols = ["y60_duo", "y120_duo", "y180_duo"],
        category_col="factor_0", 
        threshold_map={"y60_duo":0.0016},
        target_category=7,
        window_len=1,
        padding_value=0.0,
        split_ratio=0.05,
        split_type="random",
        split_start=0.9,
        select_features= False
    )
    model_config = TransformerConfig(
    )

    trainer_config = TrainConfig(
    batch_size = 512,
    max_epochs = 30,
    min_epochs= 15, 
    )

    run_training(exp_config, data_config, model_config, trainer_config)

if __name__ == "__main__":
    main()
