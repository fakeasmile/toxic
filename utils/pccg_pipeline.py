"""PCCG训练与测试流水线

整合训练和测试功能，实现训练完成后自动测试的流水线。
支持命令行参数配置，确保训练-测试配置一致性。

使用示例：
    # 1. 训练+测试
    python utils/pccg_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ

    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/pccg_pipeline.py --mode test --timestamp 20260611-120000

前置条件：
    1. 已生成Likert概念向量: data/processed/<dataset>/<model>/concept_train.json
    2. 已生成BGE嵌入: data/processed/<dataset>/<model>/bge_reasoning_train.pt
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.PCCG_config import PCCGConfig
from models.pccg import PCCG

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCCG 训练与测试流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None, help='测试模式时的实验时间戳')

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct-AWQ')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)

    # GNN配置
    parser.add_argument('--gnn_hidden_dim', type=int, default=None)
    parser.add_argument('--gnn_num_heads', type=int, default=None)
    parser.add_argument('--gnn_num_layers', type=int, default=None)

    # IB配置
    parser.add_argument('--ib_beta_target', type=float, default=None)
    parser.add_argument('--alpha_ib', type=float, default=None)

    # 分类器配置
    parser.add_argument('--hidden_features', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)

    # 损失权重
    parser.add_argument('--gamma_sparse', type=float, default=None)
    parser.add_argument('--gamma_consist', type=float, default=None)

    return parser.parse_args()


def update_config(args):
    """基于PCCGConfig，根据命令行参数更新配置"""
    config = PCCGConfig()

    # 数据集配置
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name

    # 随机种子
    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    # GNN配置
    if args.gnn_hidden_dim is not None:
        config.gnn_hidden_dim = args.gnn_hidden_dim
    if args.gnn_num_heads is not None:
        config.gnn_num_heads = args.gnn_num_heads
    if args.gnn_num_layers is not None:
        config.gnn_num_layers = args.gnn_num_layers

    # IB配置
    if args.ib_beta_target is not None:
        config.ib_beta_target = args.ib_beta_target
    if args.alpha_ib is not None:
        config.alpha_ib = args.alpha_ib

    # 分类器配置
    if args.hidden_features is not None:
        config.hidden_features = args.hidden_features
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate

    # 训练超参数
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_lr is not None:
        config.max_lr = args.max_lr
    if args.patience is not None:
        config.patience = args.patience

    # 损失权重
    if args.gamma_sparse is not None:
        config.gamma_sparse = args.gamma_sparse
    if args.gamma_consist is not None:
        config.gamma_consist = args.gamma_consist

    # 动态生成路径
    processed_dir = config.processed_path / config.dataset_name / config.model_name
    config.train_likert_path = processed_dir / "concept_train.json"
    config.test_likert_path = processed_dir / "concept_test.json"
    config.train_bge_path = processed_dir / "bge_reasoning_train.pt"
    config.test_bge_path = processed_dir / "bge_reasoning_test.pt"

    return config


def load_data(config, mode):
    """加载Likert概念向量、BGE嵌入和标签

    Returns:
        likert: [N, num_likert_concepts] Tensor
        bge: [N, 7, 768] Tensor
        labels: [N] Tensor
    """
    if mode == "train":
        likert_path = config.train_likert_path
        bge_path = config.train_bge_path
    elif mode == "test":
        likert_path = config.test_likert_path
        bge_path = config.test_bge_path
    else:
        raise ValueError(f"mode must be 'train' or 'test', got {mode}")

    # 加载Likert概念向量
    with open(likert_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    likert_vectors = []
    labels = []
    for item in raw_data:
        likert_vectors.append(item["concept"])
        labels.append(item["toxic"])

    likert = torch.tensor(likert_vectors, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 加载BGE嵌入
    if not bge_path.exists():
        print(f"警告: BGE嵌入文件不存在: {bge_path}")
        print("将使用零向量替代，请运行 encode_reasoning_bge.py 生成嵌入")
        bge = torch.zeros(len(labels), 7, config.bge_dim, dtype=torch.float32)
    else:
        bge = torch.load(bge_path, map_location="cpu", weights_only=True)
        if bge.dim() == 2:
            # 如果只有1个维度的嵌入，扩展为7个
            bge = bge.unsqueeze(1).expand(-1, 7, -1)

    # 维度校验
    assert likert.shape[0] == labels.shape[0] == bge.shape[0], \
        f"数据维度不匹配: likert={likert.shape[0]}, labels={labels.shape[0]}, bge={bge.shape[0]}"

    return likert, bge, labels


def build_edge_index(config, device):
    """构建因果图边索引张量（含自环）

    Returns:
        edge_index: [2, E] LongTensor on device
    """
    edge_index = torch.tensor(config.causal_edges_with_self, dtype=torch.long).t().contiguous()
    return edge_index.to(device)


def compute_consistency_loss(concept_scores, likert_scores, dim_concept_counts, node_names):
    """计算一致性约束损失

    因果概念得分应与对应维度的Likert均值一致。
    Likert向量按CSV行序排列，与node_names顺序一致。
    正面概念不归入任何因果图节点，跳过。

    Args:
        concept_scores: [B, 7] 因果概念得分
        likert_scores: [B, num_likert_concepts] Likert标量
        dim_concept_counts: dict 维度→概念数
        node_names: list 维度名列表（顺序与Likert向量中维度排列顺序一致）
    """
    # 计算每个维度的Likert均值
    dim_means = []
    offset = 0
    for name in node_names:
        count = dim_concept_counts[name]
        dim_likert = likert_scores[:, offset:offset + count]
        dim_mean = dim_likert.mean(dim=1)  # [B]
        dim_means.append(dim_mean)
        offset += count
    # offset之后的概念是正面概念，不归入任何因果图节点，跳过

    dim_means = torch.stack(dim_means, dim=1)  # [B, 7]

    # 一致性损失：因果概念得分与Likert维度均值的MSE
    loss = nn.functional.mse_loss(concept_scores, dim_means)
    return loss


def train(config, train_dataset, val_dataset, test_dataset, edge_index):
    """训练PCCG模型

    Args:
        config: PCCGConfig
        train_dataset: (likert, bge, labels) 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        edge_index: [2, E] 因果边

    Returns:
        训练指标历史
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型
    model = PCCG(
        num_nodes=7,
        bge_dim=config.bge_dim,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_heads=config.gnn_num_heads,
        gnn_num_layers=config.gnn_num_layers,
        ib_beta=config.ib_beta_min,  # 初始IB强度
        num_likert_concepts=train_dataset[0][0].shape[0],
        hidden_features=config.hidden_features,
        dropout_rate=config.dropout_rate,
    ).to(device)

    # 损失函数、优化器、学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr / config.div_factor)

    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        total_steps=total_steps,
        pct_start=config.pct_start,
        anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        three_phase=False,
    )

    # IB退火参数
    ib_warmup_steps = int(config.ib_warmup_ratio * total_steps)
    global_step = 0

    # 训练状态
    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    # 指标记录
    epoch_list = []
    val_loss_history = []
    val_f1_history = []
    val_precision_history = []
    val_recall_history = []
    test_f1_history = []
    test_loss_history = []

    for epoch in range(config.epochs):
        # ========== 训练阶段 ==========
        model.train()
        for batch_likert, batch_bge, batch_labels in train_loader:
            batch_likert = batch_likert.to(device)
            batch_bge = batch_bge.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits, gate_weights, ib_loss, concept_scores = model(
                batch_bge, batch_likert, edge_index
            )

            # 分类损失
            L_cls = criterion(logits, batch_labels)

            # 稀疏门控损失
            L_sparse = gate_weights.abs().sum()

            # 一致性约束损失
            L_consist = compute_consistency_loss(
                concept_scores, batch_likert,
                config.dim_concept_counts, config.node_names
            )

            # 总损失
            loss = L_cls + config.alpha_ib * ib_loss + config.gamma_sparse * L_sparse + config.gamma_consist * L_consist

            loss.backward()
            optimizer.step()
            scheduler.step()

            # IB退火
            global_step += 1
            if global_step < ib_warmup_steps:
                beta = config.ib_beta_min + (config.ib_beta_target - config.ib_beta_min) * (global_step / ib_warmup_steps)
            else:
                beta = config.ib_beta_target
            model.grouped_ib.ib_beta = beta

        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for val_likert, val_bge, val_labels in val_loader:
                val_likert = val_likert.to(device)
                val_bge = val_bge.to(device)
                val_labels = val_labels.to(device)

                logits, gate_weights, ib_loss, concept_scores = model(
                    val_bge, val_likert, edge_index
                )
                v_loss = criterion(logits, val_labels)
                total_val_loss += v_loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(val_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # ========== 测试集评估（仅观察）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for t_likert, t_bge, t_labels in test_loader:
                t_likert = t_likert.to(device)
                t_bge = t_bge.to(device)
                t_labels = t_labels.to(device)

                t_logits, _, _, _ = model(t_bge, t_likert, edge_index)
                t_loss = criterion(t_logits, t_labels)
                total_test_loss += t_loss.item()
                test_preds.extend(torch.argmax(t_logits, dim=1).cpu().numpy())
                test_labels_list.extend(t_labels.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        # 记录指标
        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        val_precision_history.append(val_p)
        val_recall_history.append(val_r)
        test_f1_history.append(test_f1)
        test_loss_history.append(avg_test_loss)

        current_beta = model.grouped_ib.ib_beta
        print(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}, "
              f"Val P={val_p:.4f}, Val R={val_r:.4f}, "
              f"Test F1={test_f1:.4f}, IB_beta={current_beta:.4f}")

        # ========== 最佳模型选择与早停 ==========
        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> 发现更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停触发: 验证集F1已连续 {config.patience} 个epoch未提升")
            break

    # 保存最佳模型
    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    """绘制训练曲线图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('PCCG Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, val_precisions, color='tab:green', linestyle='--', label='Val Precision')
    ax2.plot(epochs, val_recalls, color='tab:orange', linestyle=':', label='Val Recall')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='-.', label='Test F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def evaluate(config, timestamp):
    """评估指定实验的最佳模型在测试集上的表现"""
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_likert, test_bge, test_labels = load_data(saved_config, "test")
    test_dataset = TensorDataset(test_likert, test_bge, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

    # 加载原始文本
    with open(saved_config.test_likert_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    contents = [item["content"] for item in raw_data]

    # 构建因果边（从保存的配置中恢复）
    saved_edges = getattr(saved_config, 'causal_edges', None)
    if saved_edges is not None:
        # 添加自环
        edges_with_self = saved_edges + [(i, i) for i in range(7)]
        edge_index = torch.tensor(edges_with_self, dtype=torch.long).t().contiguous().to(device)
    else:
        # 回退到默认配置
        default_config = PCCGConfig()
        edge_index = build_edge_index(default_config, device)

    # 加载模型
    model = PCCG(
        num_nodes=7,
        bge_dim=getattr(saved_config, 'bge_dim', 768),
        gnn_hidden_dim=getattr(saved_config, 'gnn_hidden_dim', 256),
        gnn_num_heads=getattr(saved_config, 'gnn_num_heads', 4),
        gnn_num_layers=getattr(saved_config, 'gnn_num_layers', 2),
        ib_beta=getattr(saved_config, 'ib_beta_target', 1.0),
        num_likert_concepts=test_likert.shape[1],
        hidden_features=getattr(saved_config, 'hidden_features', 128),
        dropout_rate=getattr(saved_config, 'dropout_rate', 0.3),
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_likert, batch_bge, batch_labels in test_loader:
            batch_likert = batch_likert.to(device)
            batch_bge = batch_bge.to(device)

            logits, _, _, _ = model(batch_bge, batch_likert, edge_index)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      PCCG 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    # 保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("PCCG 测试集评估结果\n")
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
        config = update_config(args)

        # 生成时间戳并创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存配置
        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "train_likert_path": str(config.train_likert_path),
            "test_likert_path": str(config.test_likert_path),
            "train_bge_path": str(config.train_bge_path),
            "test_bge_path": str(config.test_bge_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "bge_dim": config.bge_dim,
            "gnn_hidden_dim": config.gnn_hidden_dim,
            "gnn_num_heads": config.gnn_num_heads,
            "gnn_num_layers": config.gnn_num_layers,
            "ib_beta_min": config.ib_beta_min,
            "ib_beta_target": config.ib_beta_target,
            "ib_warmup_ratio": config.ib_warmup_ratio,
            "alpha_ib": config.alpha_ib,
            "gamma_sparse": config.gamma_sparse,
            "gamma_consist": config.gamma_consist,
            "hidden_features": config.hidden_features,
            "dropout_rate": config.dropout_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "patience": config.patience,
            "max_lr": config.max_lr,
            "pct_start": config.pct_start,
            "div_factor": config.div_factor,
            "final_div_factor": config.final_div_factor,
            "anneal_strategy": config.anneal_strategy,
            "causal_edges": config.causal_edges,
            "dim_concept_counts": config.dim_concept_counts,
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        # 确定性模式
        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")
        else:
            print(">>> 已禁用确定性模式，结果将不可复现")

        # 加载数据
        train_likert, train_bge, train_labels = load_data(config, "train")
        test_likert, test_bge, test_labels = load_data(config, "test")

        # 划分验证集
        indices = list(range(len(train_labels)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.1,
            stratify=train_labels.numpy(),
            random_state=config.seed
        )

        train_dataset = TensorDataset(
            train_likert[train_idx], train_bge[train_idx], train_labels[train_idx]
        )
        val_dataset = TensorDataset(
            train_likert[val_idx], train_bge[val_idx], train_labels[val_idx]
        )
        test_dataset = TensorDataset(test_likert, test_bge, test_labels)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
        print(f">>> Likert维度: {train_likert.shape[1]}, BGE形状: {train_bge.shape}")

        # 构建因果边
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_index = build_edge_index(config, device)

        # 训练
        metrics = train(config, train_dataset, val_dataset, test_dataset, edge_index)

        # 绘制训练曲线
        plot_metrics(config, *metrics)

        # all模式下执行测试
        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = PCCGConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
