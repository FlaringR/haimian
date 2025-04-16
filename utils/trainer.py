from pathlib import Path
from datetime import datetime
import os
import torch
import numpy as np
import random
import os

def get_filelist(train_dir: str, start_date: str = None, end_date: str = None) -> list:
    """获取指定日期范围内的训练文件列表"""
    files = sorted(Path(train_dir).glob("*.ftr"))
    if not start_date and not end_date:
        return files
    
    start_date = datetime.strptime(start_date, "%Y%m%d") if start_date else datetime.min
    end_date = datetime.strptime(end_date, "%Y%m%d") if end_date else datetime.max
    
    return [str(f) for f in files if start_date <= datetime.strptime(f.stem, "%Y%m%d") <= end_date]

def set_seed(seed: int):
    """设置全局随机种子，确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

def generate_log_dir(base_dir:str, model_type: str) -> str:
    """生成动态 log_dir"""
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"{model_type}_")]
    if not existing_dirs:
        version = 1
    else:
        versions = [int(d.split("_")[-1]) for d in existing_dirs]
        version = max(versions) + 1
    return os.path.join(base_dir, f"{model_type}_{version}")
