import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from configs.base_config import BaseConfig
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
    # 根据 generate_adjective_c_r.py 生成的文件名加载数据
    # 注意：generate 生成的是 with_concepts2.json，此处根据您的 load_data 逻辑调整
    path = base_config.processed_path / "train_with_concepts.json"
    with open(path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    with open(base_config.train_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)


    concepts_matrix = []
    labels = []

    for i in range(0, len(raw_concept_data)):
        # 特征：形容词概念向量
        concepts_matrix.append(raw_concept_data[i]["concept"])
        # 标签：假设原始数据中包含 "label" 字段，若没有需根据实际字段名修改
        labels.append(raw_data[i]["toxic"])

    concepts_tensor = torch.tensor(concepts_matrix, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return concepts_tensor, labels_tensor


def train(features, labels):
    # 1. 准备数据迭代器
    dataset = TensorDataset(features, labels)

    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型
    # in_features 即为形容词的数量 (b)
    model = MLP(in_features=features.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 4. 训练循环
    epochs = 20  # 可根据需要调整
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(batch_x)
            # 计算损失
            loss = criterion(outputs, batch_y)
            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # 5. 保存模型
    torch.save(model.state_dict(), base_config.experiment_path / "mlp_model.pth")
    print(">>> 训练完成，模型已保存。")


if __name__ == '__main__':
    # 1. 环境初始化
    base_config = init()
    # 2. 加载数据（特征和标签）
    concepts_tensor, labels_tensor = load_data(base_config)
    # 3. 开始训练
    train(concepts_tensor, labels_tensor)