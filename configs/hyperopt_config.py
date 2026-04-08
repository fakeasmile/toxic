from pathlib import Path


class HyperOptConfig:
    def __init__(self):
        # Optuna基础参数
        self.n_trials = 10  # 搜索轮数

        # 超参数搜索空间（key对应训练参数，value为搜索范围/类型）
        self.search_space = {
            "lr": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-4,
                "log": True
            },
            "batch_size": {
                "type": "categorical",
                "choices": [8, 16]
            },
            "weight_decay": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-3,
                "log": True
            },
            "max_len": {
                "type": "categorical",
                "choices": [80, 96, 112]
            }
        }