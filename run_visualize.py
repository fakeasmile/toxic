from pathlib import Path

from configs.base_config import BaseConfig
from utils.test import test
from scripts.visualize import visualize_train_dev_losses

if __name__ == '__main__':
    """
    对模型的训练结果可视化
    """
    base_config = BaseConfig()

    # visualize_train_dev_losses(Path(__file__).parent / "experiments"  / "20260315-035855.pth", if_save=False)
    visualize_train_dev_losses(Path(__file__).parent / "experiments" / "20260316-20-38-41" / "checkpoint.pth", if_save=True)