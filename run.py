import random
import numpy as np
import torch
import os

from configs.base_config import BaseConfig
from configs.hyperopt_config import HyperOptConfig
from configs.log_config import setup_logger
from utils.hyperopt import run_hyperopt
from utils.train import final_train

def save_result(base_config, best_params, train_losses, dev_losses, dev_accuracies, fnn_state_dict, bert_state_dict, save_path, descriptions):
    print(f"{'='*10}保存结果{'='*10}")

    # 配置
    config = {
        "base_config": base_config,
    }

    # 超参数搜索到的最优超参数
    best_search_hyperparameters = {
        "best_params": best_params,
    }

    hyperparameters = {  # 为了适配之前的代码
        "lr": best_params["lr"],
        "batch_size": best_params["batch_size"],
        "max_len": best_params["max_len"],
        "weight_decay": best_params["weight_decay"],
        "max_epochs": base_config.epochs,
        "seed": base_config.seed
    }

    # 在final_train中每个epoch训练集上的平均损失，验证集上的平均损失和准确率
    loss_history = {
        "train": train_losses,
        "dev": dev_losses,
        "dev_accuracies": dev_accuracies
    }

    # 全连接层参数
    model = {
        "fnn": fnn_state_dict,
        "bert": bert_state_dict
    }

    # 额外记录的信息
    extra = {
        "descriptions": descriptions
    }

    checkpoint = {
        "config": config,
        "best_search_hyperparameters": best_search_hyperparameters,
        "hyperparameters": hyperparameters,
        "loss_history": loss_history,
        "model": model,
        "extra": extra
    }

    path = save_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    print(f"=>训练结果保存到：{path}")

if __name__ == '__main__':
    # -----初始化配置---------
    base_config = BaseConfig()  # 基础配置
    hyperopt_config = HyperOptConfig()  # 超参数配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, logging_path = setup_logger(base_config.experiment_path)  # 日志配置

    # -----设置随机种子---------
    if base_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(base_config)
        print(">>> 已启用确定性模式 (Reproducibility Enabled)")
    else:
        print(">>> 已禁用确定性模式 (Randomness Enabled)，结果将不可复现")

    # -----超参数搜索---------
    # best_params = run_hyperopt(base_config, hyperopt_config)

    # best_params = {'lr': 1.3381786793705053e-05, 'batch_size': 8, 'weight_decay': 0.00035205298032095185, 'max_len': 96}
    best_params = {'lr': 1e-5, 'batch_size': 64, 'weight_decay': 0.00035205298032095185, 'max_len': 80}

    # -----用最优参数进行训练---------
    train_losses, dev_losses, dev_accuracies, fnn_state_dict, bert_state_dict = final_train(base_config, best_params)


    description = ""
    save_result(base_config, best_params, train_losses, dev_losses, dev_accuracies, fnn_state_dict, bert_state_dict, logging_path, description)



