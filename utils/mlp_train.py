from configs.base_config import BaseConfig
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from models.mlp import MLP


def init():
    base_config = BaseConfig()  # 基础配置
    if base_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(base_config)
        print(">>> 已启用确定性模式 (Reproducibility Enabled)")
    else:
        print(">>> 已禁用确定性模式 (Randomness Enabled)，结果将不可复现")
    return base_config


def load_data(base_config, file_name):
    """加载指定文件的概念向量和标签"""

    # 加载概念向量
    concept_path = base_config.processed_path / file_name
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    # 从源数据集中加载标签
    if file_name.startswith("train"):
        label_path = base_config.train_path
    elif file_name.startswith("test"):
        label_path = base_config.test_path
    else:
        raise RuntimeError("in load_data, file_name must be start with 'train' or 'test'")

    with open(label_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    concepts, labels = [], []

    assert len(raw_data) == len(raw_concept_data)

    for i in range(0, len(raw_concept_data)):
        # 特征：形容词概念向量
        concepts.append(raw_concept_data[i]["concept"])
        labels.append(raw_data[i]["toxic"])

    return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def train(base_config, train_data, test_data):
    # 解构训练集与测试集的特征和标签
    train_x, train_y = train_data
    test_x, test_y = test_data
    # 构造训练数据加载器，设置Batch大小并打乱
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    # 构造测试数据加载器，用于模型验证
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)
    # 初始化MLP模型，输入维度为形容词数量
    model = MLP(in_features=train_x.shape[1])
    # 自动选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到计算设备
    model.to(device)
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义Adam优化器，设置学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # 初始化历史最高F1分数
    best_f1 = 0.0

    # 开始训练循环
    for epoch in range(50):
        # 开启训练模式
        model.train()
        # 遍历训练批次
        for batch_x, batch_y in train_loader:
            # 数据搬运至设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向计算得到预测输出
            outputs = model(batch_x)
            # 计算当前批次损失
            loss = criterion(outputs, batch_y)
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

        # 开启评估模式
        model.eval()
        # 初始化预测列表与真实列表
        all_preds, all_labels = [], []
        # 禁用梯度计算以节省显存
        with torch.no_grad():
            # 遍历测试集进行验证
            for val_x, val_y in test_loader:
                # 验证数据搬运
                val_x, val_y = val_x.to(device), val_y.to(device)
                # 获取模型原始输出
                val_outputs = model(val_x)
                # 取最大概率对应的索引作为预测类
                preds = torch.argmax(val_outputs, dim=1)
                # 收集预测结果
                all_preds.extend(preds.cpu().numpy())
                # 收集真实标签
                all_labels.extend(val_y.cpu().numpy())

        # 计算当前Epoch在验证集上的宏观F1分数
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        # 打印当前训练进度与性能
        print(f"Epoch {epoch + 1}: Test F1 = {current_f1:.4f}")

        # 如果当前性能优于历史最好水平
        if current_f1 > best_f1:
            # 更新最高分记录
            best_f1 = current_f1
            # 保存性能最佳的模型参数
            torch.save(model.state_dict(), base_config.processed_path / "best_mlp_model.pth")
            # 打印保存提示
            print(f">>> 发现更优模型，已保存至 {base_config.processed_path}")


if __name__ == '__main__':
    base_config = init()
    # 加载训练集
    train_data = load_data(base_config, "train_with_concepts.json")
    # 加载测试集
    test_data = load_data(base_config, "test_with_concepts.json")
    # 执行训练与验证过程
    train(base_config, train_data, test_data)