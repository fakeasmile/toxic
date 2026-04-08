from configs.base_config import BaseConfig
import json
import torch
from models.mlp import MLP

def init():
    base_config = BaseConfig()  # 基础配置

    # -----设置随机种子---------
    if base_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(base_config)
        print(">>> 已启用确定性模式 (Reproducibility Enabled)")
    else:
        print(">>> 已禁用确定性模式 (Randomness Enabled)，结果将不可复现")

    return base_config


def load_data(base_config):
    with open(base_config.processed_path / "train_with_concepts.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    concepts_matrix = []
    for sample_data in raw_data:
        concepts_matrix.append(sample_data["concept"])
    concepts_tensor = torch.tensor(concepts_matrix, dtype=torch.long)

    return concepts_tensor




if __name__ == '__main__':
    # 先用这个形容词概念向量训练试一试
    base_config = init()
    concepts_tensor = load_data(base_config)  # 形容词概念向量，形状[N,A]

