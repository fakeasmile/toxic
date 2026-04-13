from configs.MLP_config import MLPConfig
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from models.mlp import MLP
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def init():
    mlp_config = MLPConfig()  # MLP 配置
    
    # 生成时间戳并创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = mlp_config.experiment_path / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    mlp_config.experiment_path = experiment_dir
    
    if mlp_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(mlp_config)
        print(">>> 已启用确定性模式 (Reproducibility Enabled)")
    else:
        print(">>> 已禁用确定性模式 (Randomness Enabled)，结果将不可复现")
    
    # 打印关键配置参数到控制台
    print("\n" + "="*60)
    print("MLP 训练配置参数")
    print("="*60)
    print(f"实验目录: {mlp_config.experiment_path}")
    print(f"时间戳: {timestamp}")
    print("\n--- 数据集配置 ---")
    print(f"数据集名称: {mlp_config.dataset_name}")
    print(f"模型名称: {mlp_config.model_name}")
    print(f"训练集路径: {mlp_config.train_path}")
    print(f"测试集路径: {mlp_config.test_path}")
    print("\n--- 训练超参数 ---")
    print(f"批次大小 (batch_size): {mlp_config.batch_size}")
    print(f"训练轮数 (epochs): {mlp_config.epochs}")
    print(f"峰值学习率 (max_lr): {mlp_config.max_lr}")
    print(f"Warmup比例 (pct_start): {mlp_config.pct_start}")
    print(f"初始学习率除数 (div_factor): {mlp_config.div_factor}")
    print(f"最终学习率除数 (final_div_factor): {mlp_config.final_div_factor}")
    print(f"衰减策略 (anneal_strategy): {mlp_config.anneal_strategy}")
    print("\n--- 模型结构参数 ---")
    print(f"Dropout比率: {mlp_config.dropout_rate}")
    print(f"隐藏层维度: {mlp_config.hidden_features}")
    print("\n--- 随机种子配置 ---")
    print(f"随机种子 (seed): {mlp_config.seed}")
    print(f"确定性模式: {mlp_config.use_deterministic}")
    print("="*60 + "\n")
    
    # 保存完整配置到JSON文件
    config_dict = {
        "timestamp": timestamp,
        "experiment_path": str(mlp_config.experiment_path),
        "dataset_name": mlp_config.dataset_name,
        "model_name": mlp_config.model_name,
        "train_path": str(mlp_config.train_path),
        "test_path": str(mlp_config.test_path),
        "processed_path": str(mlp_config.processed_path),
        "seed": mlp_config.seed,
        "use_deterministic": mlp_config.use_deterministic,
        "batch_size": mlp_config.batch_size,
        "epochs": mlp_config.epochs,
        "max_lr": mlp_config.max_lr,
        "pct_start": mlp_config.pct_start,
        "div_factor": mlp_config.div_factor,
        "final_div_factor": mlp_config.final_div_factor,
        "anneal_strategy": mlp_config.anneal_strategy,
        "dropout_rate": mlp_config.dropout_rate,
        "hidden_features": mlp_config.hidden_features
    }
    
    config_file = mlp_config.experiment_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f">>> 配置文件已保存至: {config_file}\n")
    
    return mlp_config


def load_data(mlp_config, file_name):
    """加载指定文件的概念向量和标签"""
    concept_path = mlp_config.processed_path / file_name
    # 加载概念向量文件
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    if file_name.startswith("train"):
        label_path = mlp_config.train_path
    elif file_name.startswith("test"):
        label_path = mlp_config.test_path
    else:
        raise RuntimeError("in load_data, file_name must be start with 'train' or 'test'")

    # 加载原始数据集
    with open(label_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    concepts, labels = [], []
    assert len(raw_data) == len(raw_concept_data)

    # 提取形容词概念向量和对应的标签
    for i in range(0, len(raw_concept_data)):
        concepts.append(raw_concept_data[i]["concept"])
        labels.append(raw_data[i]["toxic"])

    return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def plot_metrics(mlp_config, epochs, losses, f1_scores, precisions, recalls):
    """绘制损失与各项评价指标曲线图"""
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制验证集损失 (左轴)
    lns1 = ax1.plot(epochs, losses, color='tab:red', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 绘制验证集 F1, Precision, Recall (右轴)
    lns2 = ax2.plot(epochs, f1_scores, color='tab:blue', label='Val F1')
    lns3 = ax2.plot(epochs, precisions, color='tab:green', linestyle='--', label='Val Precision')
    lns4 = ax2.plot(epochs, recalls, color='tab:orange', linestyle=':', label='Val Recall')

    ax2.set_ylabel('Score', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('MLP Training Metrics with Gating Mechanism')

    # 合并图例
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')

    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = mlp_config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train(mlp_config, train_data):

    batch_size = mlp_config.batch_size
    epochs = mlp_config.epochs

    # 从训练集中按9:1比例划分验证集（分层抽样）
    full_train_x, full_train_y = train_data
    train_x_np, val_x_np, train_y_np, val_y_np = train_test_split(
        full_train_x.numpy(), full_train_y.numpy(),
        test_size=0.1,
        stratify=full_train_y.numpy(),
        random_state=mlp_config.seed
    )
    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    val_x = torch.tensor(val_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.long)
    val_y = torch.tensor(val_y_np, dtype=torch.long)

    print(f">>> 训练集样本数: {len(train_x)}, 验证集样本数: {len(val_x)}")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_features=train_x.shape[1], dropout_rate=mlp_config.dropout_rate, 
                hidden_features=mlp_config.hidden_features).to(device)

    # 损失函数，优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=mlp_config.max_lr / mlp_config.div_factor)

    max_lr = mlp_config.max_lr  # 峰值学习率
    pct_start = mlp_config.pct_start  # Warmup 占总步数的比例
    total_steps = len(train_loader) * epochs  # 总训练步数

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=mlp_config.anneal_strategy,  # Warmup 后余弦衰减
        div_factor=mlp_config.div_factor,  # 初始学习率 = max_lr / div_factor
        final_div_factor=mlp_config.final_div_factor,  # 最终学习率 = max_lr / final_div_factor
        three_phase=False
    )

    # 指标
    best_f1 = 0.0
    best_mlp_status_dict = None
    # 验证集上的损失历史，F1历史...
    epoch_list, loss_history, f1_history, precision_history, recall_history = [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            scheduler.step()

        # 验证集上验证
        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)

                val_outputs = model(val_x)

                v_loss = criterion(val_outputs, val_y)
                total_val_loss += v_loss.item()

                preds = torch.softmax(val_outputs, dim=1)
                preds = torch.argmax(preds, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # 计算指标
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        current_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        current_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # 保存结果
        epoch_list.append(epoch + 1)
        loss_history.append(avg_val_loss)
        f1_history.append(current_f1)
        precision_history.append(current_precision)
        recall_history.append(current_recall)

        print(f"Epoch {epoch + 1}: \n>>>Loss = {avg_val_loss:.4f}, \n>>>F1 = {current_f1:.4f}, \n>>>P = {current_precision:.4f}, "
              f"\n>>>R = {current_recall:.4f}")

        if current_f1 > best_f1:
            print(f">>> 发现更优模型 (F1: {current_f1:.4f})，提升：{current_f1 - best_f1:.4f}")
            best_f1 = current_f1
            best_mlp_status_dict = model.state_dict()

    # 保存模型
    if best_mlp_status_dict is not None:
        model_save_path = mlp_config.experiment_path / "best_model.pth"
        torch.save(best_mlp_status_dict, model_save_path)
        print(f">>> 最佳模型已保存至: {model_save_path}")
    # 调用绘图函数
    plot_metrics(mlp_config, epoch_list, loss_history, f1_history, precision_history, recall_history)


if __name__ == '__main__':
    mlp_config = init()
    train_data = load_data(mlp_config, f"train_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json")
    train(mlp_config, train_data)