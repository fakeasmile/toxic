"""GraphCBM 训练与测试流水线

使用方法：
    # 训练+测试（TOXICN，形容词概念）
    python utils/graph_cbm_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-GPTQ-Int8

    # 仅测试
    python utils/graph_cbm_pipeline.py --mode test --timestamp 20260609-120000
"""

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig
from models.graph_cbm import GraphCBM, build_adjacency_matrix

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    parser = argparse.ArgumentParser(description="GraphCBM 训练与测试流水线")

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None, help='测试模式时的实验时间戳')

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct-GPTQ-Int8')
    parser.add_argument('--concept_type', type=str, default='adjective', choices=['adjective', 'reasoning'])

    # 随机种子
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_lr', type=float, default=None)
    parser.add_argument('--pct_start', type=float, default=None)
    parser.add_argument('--div_factor', type=float, default=None)
    parser.add_argument('--final_div_factor', type=float, default=None)
    parser.add_argument('--anneal_strategy', type=str, default=None)

    # 模型结构参数
    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--mlp_hidden', type=int, default=None, help='MLP隐藏层维度')
    parser.add_argument('--gcn_hidden', type=int, default=None, help='GCN隐藏层维度')
    parser.add_argument('--num_gcn_layers', type=int, default=None, help='GCN层数')
    parser.add_argument('--patience', type=int, default=None)

    # 图结构参数
    parser.add_argument('--intra_weight', type=float, default=1.0, help='同类别内边权重')
    parser.add_argument('--inter_weight', type=float, default=0.3, help='相邻类别间边权重')

    return parser.parse_args()


def build_config(args):
    """构建配置对象"""
    config = MLPConfig()
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    config.concept_type = args.concept_type

    # 概念向量路径
    if config.concept_type == "reasoning":
        concept_subdir = config.processed_path / config.dataset_name / config.model_name / "reasoning_patterns"
    else:
        concept_subdir = config.processed_path / config.dataset_name / config.model_name

    config.train_concept_path = concept_subdir / "concept_train.json"
    config.test_concept_path = concept_subdir / "concept_test.json"

    # 随机种子
    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    # 训练超参数
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_lr is not None:
        config.max_lr = args.max_lr
    if args.pct_start is not None:
        config.pct_start = args.pct_start
    if args.div_factor is not None:
        config.div_factor = args.div_factor
    if args.final_div_factor is not None:
        config.final_div_factor = args.final_div_factor
    if args.anneal_strategy is not None:
        config.anneal_strategy = args.anneal_strategy
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    if args.patience is not None:
        config.patience = args.patience

    # GraphCBM 特有参数
    config.mlp_hidden = args.mlp_hidden if args.mlp_hidden is not None else 96
    config.gcn_hidden = args.gcn_hidden if args.gcn_hidden is not None else 32
    config.num_gcn_layers = args.num_gcn_layers if args.num_gcn_layers is not None else 2
    config.intra_weight = args.intra_weight
    config.inter_weight = args.inter_weight

    return config


def load_data(config, mode):
    """加载概念向量和标签"""
    if mode == "train":
        concept_path = config.train_concept_path
    elif mode == "test":
        concept_path = config.test_concept_path
    else:
        raise ValueError("mode must be 'train' or 'test'")

    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    concepts, labels = [], []
    for item in raw_concept_data:
        concepts.append(item["concept"])
        labels.append(item["toxic"])

    return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def plot_metrics(config, epochs, val_losses, val_f1_scores, test_f1_scores):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('GraphCBM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='-.', label='Test F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train(config, train_dataset, val_dataset, test_dataset, adj_norm):
    """训练GraphCBM模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    num_concepts = train_dataset[0][0].shape[0]
    adj_norm = adj_norm.to(device)

    model = GraphCBM(
        num_concepts=num_concepts,
        adj_norm=adj_norm,
        gcn_hidden=config.gcn_hidden,
        mlp_hidden=config.mlp_hidden,
        dropout_rate=config.dropout_rate,
        num_gcn_layers=config.num_gcn_layers
    ).to(device)

    print(f">>> 模型参数量: {sum(p.numel() for p in model.parameters())}")
    print(f">>> GCN隐藏维度: {config.gcn_hidden}, GCN层数: {config.num_gcn_layers}")
    print(f">>> 图结构: intra_weight={config.intra_weight}, inter_weight={config.inter_weight}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr / config.div_factor)

    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr, total_steps=total_steps,
        pct_start=config.pct_start, anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor, final_div_factor=config.final_div_factor,
        three_phase=False
    )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list, val_loss_history, val_f1_history, test_f1_history = [], [], [], []

    for epoch in range(config.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 验证集评估
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                v_loss = criterion(val_outputs, val_y)
                total_val_loss += v_loss.item()
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels_list.extend(val_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        # 测试集评估（仅观察）
        test_preds, test_labels_list = [], []
        with torch.no_grad():
            for tx, ty in test_loader:
                tx = tx.to(device)
                t_outputs = model(tx)
                test_preds.extend(torch.argmax(t_outputs, dim=1).cpu().numpy())
                test_labels_list.extend(ty.numpy())

        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        test_f1_history.append(test_f1)

        print(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}, Test F1={test_f1:.4f}")

        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = model.state_dict()
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> 发现更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停触发: 验证集F1已连续 {config.patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return epoch_list, val_loss_history, val_f1_history, test_f1_history


def evaluate(config, timestamp):
    """评估指定实验"""
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_x, test_y = load_data(saved_config, "test")
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=int(saved_config.batch_size), shuffle=False)

    with open(saved_config.test_concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)
    contents = [item["content"] for item in raw_concept_data]

    # 构建邻接矩阵
    adj_norm = build_adjacency_matrix(
        saved_config.category_path,
        num_adjectives=test_x.shape[1],
        intra_weight=saved_config.intra_weight,
        inter_weight=saved_config.inter_weight
    )

    model = GraphCBM(
        num_concepts=test_x.shape[1],
        adj_norm=adj_norm,
        gcn_hidden=saved_config.gcn_hidden,
        mlp_hidden=saved_config.mlp_hidden,
        dropout_rate=saved_config.dropout_rate,
        num_gcn_layers=saved_config.num_gcn_layers
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      GraphCBM 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("GraphCBM 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    label_names = ["Non-Toxic", "Toxic"]
    predictions = []
    for i in range(len(all_preds)):
        predictions.append({
            "index": i,
            "content": contents[i],
            "true_label": int(all_labels[i]),
            "true_label_name": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_label_name": label_names[int(all_preds[i])],
            "correct": bool(all_preds[i] == all_labels[i])
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = build_config(args)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 构建邻接矩阵
        category_path = config.reasoning_pattern_path if config.concept_type == "reasoning" else config.adjective_path
        category_json_path = config.raw_data_path / "adjective" / "adjective_categories.json"

        adj_norm = build_adjacency_matrix(
            category_json_path,
            intra_weight=config.intra_weight,
            inter_weight=config.inter_weight
        )
        print(f">>> 邻接矩阵形状: {adj_norm.shape}")
        print(f">>> 非零元素比例: {(adj_norm > 0).float().mean().item():.4f}")

        # 保存配置
        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "model_type": "GraphCBM",
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "concept_type": config.concept_type,
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "category_path": str(category_json_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "max_lr": config.max_lr,
            "pct_start": config.pct_start,
            "div_factor": config.div_factor,
            "final_div_factor": config.final_div_factor,
            "anneal_strategy": config.anneal_strategy,
            "dropout_rate": config.dropout_rate,
            "mlp_hidden": config.mlp_hidden,
            "gcn_hidden": config.gcn_hidden,
            "num_gcn_layers": config.num_gcn_layers,
            "patience": config.patience,
            "intra_weight": config.intra_weight,
            "inter_weight": config.inter_weight,
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # 加载数据
        train_x, train_y = load_data(config, "train")
        test_x, test_y = load_data(config, "test")

        train_x_np, val_x_np, train_y_np, val_y_np = train_test_split(
            train_x.numpy(), train_y.numpy(),
            test_size=0.1, stratify=train_y.numpy(), random_state=config.seed
        )
        train_dataset = TensorDataset(
            torch.tensor(train_x_np, dtype=torch.float32),
            torch.tensor(train_y_np, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_x_np, dtype=torch.float32),
            torch.tensor(val_y_np, dtype=torch.long)
        )
        test_dataset = TensorDataset(test_x, test_y)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        # 训练
        metrics = train(config, train_dataset, val_dataset, test_dataset, adj_norm)
        plot_metrics(config, *metrics)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = MLPConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
