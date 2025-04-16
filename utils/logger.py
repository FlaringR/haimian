# utils/logger.py
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

class HaimianLogger:
    """自定义日志记录器，用于记录训练和验证的指标到独立 .log 文件"""
    def __init__(self, log_dir: str = "logs", gpu_id: Optional[int] = None):
        """初始化日志记录器

        参数:
            log_dir (str): 日志保存目录。
            gpu_id (int, optional): GPU ID，用于区分日志文件。
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 生成唯一日志文件名，包含 GPU ID 和进程 ID
        pid = os.getpid()
        gpu_suffix = f"_GPU{gpu_id}" if gpu_id is not None else ""
        self.log_path = os.path.join(
            self.log_dir, 
            f"haimian_{datetime.now().strftime('%Y%m%d_%H%M%S')}{gpu_suffix}_PID{pid}.log"
        )

        # 配置 logging
        self.logger = logging.getLogger(f"HaimianLogger_{pid}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # 清空旧处理器
        handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def log_metrics(self, stage: str, metrics: Dict[str, Any], step: Optional[int] = None, epoch: Optional[int] = None):
        """记录指标"""
        prefix = f"[{stage}]"
        if epoch is not None:
            prefix += f" Epoch {epoch}"
        if step is not None:
            prefix += f" Step {step}"
        metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items())
        self.logger.info(f"{prefix} - {metrics_str}")

    def info(self, message: str):
        """记录普通信息"""
        self.logger.info(message)

# 进程内的全局 logger，初始为 None
_current_logger = None

def get_logger() -> HaimianLogger:
    """获取当前进程的 logger，如果未初始化则抛出异常"""
    if _current_logger is None:
        raise RuntimeError("Logger not initialized. Call initialize_logger() first.")
    return _current_logger

def initialize_logger(log_dir: str, gpu_id: Optional[int] = None) -> HaimianLogger:
    """初始化当前进程的 logger"""
    global _current_logger
    _current_logger = HaimianLogger(log_dir=log_dir, gpu_id=gpu_id)
    return _current_logger