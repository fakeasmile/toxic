# log_config.py
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(experiment_path):
    """
    统一配置日志：同时输出到控制台和文件
    只需在程序入口调用一次
    """
    # 1. 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 设置级别

    logging_base_path = Path(experiment_path) / datetime.now().strftime("%Y%m%d-%H%M%S")

    # 2. 配置 "hyperopt_train" 专用 Logger
    hyperopt_train_logger = logging.getLogger("hyperopt_train")
    hyperopt_train_logger.setLevel(logging.INFO)
    hyperopt_train_logger.propagate = False

    if not hyperopt_train_logger.handlers:
        os.makedirs(logging_base_path, exist_ok=True)
        fh = logging.FileHandler(logging_base_path / "hyperopt_train.log", mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S'))
        hyperopt_train_logger.addHandler(fh)

    # 3. 配置 "final_train" 专用 Logger
    final_train_logger = logging.getLogger("final_train")
    final_train_logger.setLevel(logging.INFO)
    final_train_logger.propagate = False

    if not final_train_logger.handlers:
        os.makedirs(logging_base_path, exist_ok=True)
        fh = logging.FileHandler(logging_base_path / "final_train.log", mode='a' ,encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S'))
        final_train_logger.addHandler(fh)

    open(logging_base_path / "point.txt", 'w').close()

    logger = {
        "root": root_logger,
        "hyperopt_train": hyperopt_train_logger,
        "final_train": final_train_logger
    }

    print(">>> 日志系统初始化完成")
    return logger, logging_base_path