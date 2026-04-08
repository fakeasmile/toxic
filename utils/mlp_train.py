from configs.base_config import BaseConfig
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
from models.mlp import MLP
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


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
    concept_path = base_config.processed_path / file_name
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

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
        concepts.append(raw_concept_data[i]["concept"])
        labels.append(raw_data[i]["toxic"])

    return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def plot_metrics(base_config, epochs, losses, f1_scores, precisions, recalls):
    """绘制损失与各项评价指标曲线图"""
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制验证集损失 (左轴)
    lns1 = ax1.plot(epochs, losses, color='tab:red', marker='o', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # [修改] 绘制验证集 F1, Precision, Recall (右轴)
    lns2 = ax2.plot(epochs, f1_scores, color='tab:blue', marker='s', label='Test F1')
    lns3 = ax2.plot(epochs, precisions, color='tab:green', marker='^', linestyle='--', label='Test Precision')
    lns4 = ax2.plot(epochs, recalls, color='tab:orange', marker='v', linestyle=':', label='Test Recall')

    ax2.set_ylabel('Score', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('MLP Training Metrics with Gating Mechanism')

    # 合并图例
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')

    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = base_config.experiment_path / "training_metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train(base_config, train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    model = MLP(in_features=train_x.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_f1 = 0.0
    epoch_list = []
    loss_history = []
    f1_history = []
    # [新增] 记录精确率和召回率的历史
    precision_history = []
    recall_history = []

    for epoch in range(50):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                v_loss = criterion(val_outputs, val_y)
                total_val_loss += v_loss.item()

                preds = torch.argmax(val_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)

        # [修改] 计算三个指标
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        current_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        current_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        epoch_list.append(epoch + 1)
        loss_history.append(avg_val_loss)
        f1_history.append(current_f1)
        precision_history.append(current_precision)
        recall_history.append(current_recall)

        print(
            f"Epoch {epoch + 1}: Loss = {avg_val_loss:.4f}, F1 = {current_f1:.4f}, P = {current_precision:.4f}, R = {current_recall:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), base_config.experiment_path / "best_mlp_model.pth")
            print(f">>> 发现更优模型 (F1: {best_f1:.4f})，已保存")

    # [修改] 调用更新后的绘图函数
    plot_metrics(base_config, epoch_list, loss_history, f1_history, precision_history, recall_history)


if __name__ == '__main__':
    base_config = init()
    train_data = load_data(base_config, "train_with_concepts.json")
    test_data = load_data(base_config, "test_with_concepts.json")
    train(base_config, train_data, test_data)