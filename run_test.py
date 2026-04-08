from pathlib import Path

from configs.base_config import BaseConfig
from utils.test import test

if __name__ == '__main__':
    """
    加载模型，在测试集上测试
    """
    base_config = BaseConfig()

    test(base_config, Path(__file__).parent / "experiments" / "20260407-140745" / "checkpoint.pth")