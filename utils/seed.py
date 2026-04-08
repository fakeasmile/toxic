import random
import numpy as np
import torch
import os


def set_reproducibility(config):
    # ===================== 🔥 修复报错：必须放在最顶部 =====================
    if config.use_deterministic:
        # 开启确定性时，强制设置CuBLAS环境变量（解决报错）
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # 关闭确定性时，删除该环境变量 = 完全原生随机
        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
    # ==================================================================

    if config.use_deterministic:
        # 固定所有随机种子
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        os.environ["PYTHONHASHSEED"] = str(config.seed)

        # CUDA 确定性配置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # 恢复原生随机行为
        if "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        print(">>> 已关闭确定性模式 (Random Mode Enabled)")