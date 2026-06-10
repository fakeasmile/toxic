"""ICB-CBM训练与测试流水线

Information-Compressed Bottleneck Concept Bottleneck Model

整合训练和测试功能，实现训练完成后自动测试的流水线。
支持命令行参数配置，确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试
    python utils/icb_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct

    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/icb_pipeline.py --mode test --timestamp 20260609-120000

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳

    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)
        --model_name        LLM模型名称 (默认: Qwen2.5-7B-Instruct)

    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (默认: False)

    ICB-CBM模型参数:
        --dense_dim         每个概念的稠密向量维度 (默认: 64)
        --num_residual      残差概念数量 (默认: 32)
        --layer             LLM提取hidden state的层号 (默认: 16)
        --alpha             IB损失权重 (默认: 0.01)
        --gamma             稀疏损失权重 (默认: 0.001)
        --icc_threshold     ICC概念选择相关性阈值 (默认: 0.05)

    训练超参数:
        --batch_size        批次大小 (默认: 32)
        --epochs            训练轮数 (默认: 200)
        --max_lr            峰值学习率 (默认: 1e-3)
        --patience          早停耐心值 (默认: 30)
        --dropout_rate      Dropout比率 (默认: 0.3)
        --hidden_features   分类器隐藏层维度 (默认: 128)

参数优先级:
    - 训练模式: 命令行参数 > ICB_config.py
    - 测试模式: 强制使用实验目录的 config.json

输出文件:
    实验目录结构 (experiments/<timestamp>-icb-cbm/):
        ├── config.json              # 实验配置快照
        ├── best_model.pth           # 最佳模型权重
        ├── metrics.png              # 训练曲线图
        └── test_results/
            ├── metrics.json         # 测试集评估指标
            ├── classification_report.txt
            └── predictions.json     # 逐条预测结果
"""

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.ICB_config import ICBConfig
from models.icb_cbm import ICB_CBM

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ICB-CBM 训练与测试流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'],
                        default='all', help='运行模式')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='测试模式时的实验时间戳')

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='TOXICN', help='数据集名称')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct',
                        help='LLM模型名称')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False,
                        help='启用确定性模式')

    # ICB-CBM模型参数
    parser.add_argument('--dense_dim', type=int, default=None,
                        help='每个概念的稠密向量维度')
    parser.add_argument('--num_residual', type=int, default=None,
                        help='残差概念数量')
    parser.add_argument('--layer', type=int, default=None,
                        help='LLM提取hidden state的层号')
    parser.add_argument('--alpha', type=float, default=None,
                        help='IB损失权重')
    parser.add_argument('--gamma', type=float, default=None,
                        help='稀疏损失权重')
    parser.add_argument('--icc_threshold', type=float, default=None,
                        help='ICC概念选择相关性阈值')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_lr', type=float, default=None, help='峰值学习率')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值')
    parser.add_argument('--dropout_rate', type=float, default=None, help='Dropout比率')
    parser.add_argument('--hidden_features', type=int, default=None,
                        help='分类器隐藏层维度')

    # OneCycleLR参数
    parser.add_argument('--pct_start', type=float, default=None, help='Warmup比例')
    parser.add_argument('--div_factor', type=float, default=None, help='初始学习率除数')
    parser.add_argument('--final_div_factor', type=float, default=None,
                        help='最终学习率除数')
    parser.add_argument('--anneal_strategy', type=str, default=None,
                        help='衰减策略 (cos/linear)')

    return parser.parse_args()


def update_ICBConfig(args):
    """基于ICBConfig默认值，根据命令行参数更新配置对象

    优先级: 命令行参数 > ICBConfig默认值
    """
    config = ICBConfig()

    # 数据集配置
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name

    # 动态生成路径
    config.train_concept_path = (config.processed_path / config.dataset_name
                                  / config.model_name / "concept_train.json")
    config.test_concept_path = (config.processed_path / config.dataset_name
                                 / config.model_name / "concept_test.json")
    config.train_hidden_path = (config.processed_path / config.dataset_name
                                 / config.model_name / "hidden_train.pt")
    config.test_hidden_path = (config.processed_path / config.dataset_name
                                / config.model_name / "hidden_test.pt")

    # 随机种子
    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    # ICB-CBM模型参数
    if args.dense_dim is not None:
        config.dense_dim = args.dense_dim
    if args.num_residual is not None:
        config.num_residual = args.num_residual
    if args.layer is not None:
        config.layer = args.layer
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.icc_threshold is not None:
        config.icc_threshold = args.icc_threshold

    # 训练超参数
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_lr is not None:
        config.max_lr = args.max_lr
    if args.patience is not None:
        config.patience = args.patience
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    if args.hidden_features is not None:
        config.hidden_features = args.hidden_features

    # OneCycleLR参数
    if args.pct_start is not None:
        config.pct_start = args.pct_start
    if args.div_factor is not None:
        config.div_factor = args.div_factor
    if args.final_div_factor is not None:
        config.final_div_factor = args.final_div_factor
    if args.anneal_strategy is not None:
        config.anneal_strategy = args.anneal_strategy

    return config


def load_data(config, mode):
    """加载Likert概念向量、LLM hidden state和标签

    Args:
        config: ICBConfig配置对象
        mode: "train" 或 "test"

    Returns:
        likert_scores: [N, K] Likert标量概念向量
        hidden_states: [N, hidden_dim] LLM hidden state
        labels: [N] 标签
        contents: [N] 文本内容列表
    """
    if mode == "train":
        concept_path = config.train_concept_path
        hidden_path = config.train_hidden_path
    elif mode == "test":
        concept_path = config.test_concept_path
        hidden_path = config.test_hidden_path
    else:
        raise ValueError(f"mode must be 'train' or 'test', got {mode}")

    # 检查文件存在性
    concept_path = Path(concept_path) if not isinstance(concept_path, Path) else concept_path
    hidden_path = Path(hidden_path) if not isinstance(hidden_path, Path) else hidden_path

    if not concept_path.exists():
        raise FileNotFoundError(f"概念向量文件不存在: {concept_path}\n"
                                f"请先运行: python scripts/generate_adjective_c_r_vllm.py")
    if not hidden_path.exists():
        raise FileNotFoundError(f"Hidden state文件不存在: {hidden_path}\n"
                                f"请先运行: python scripts/extract_hidden_states.py")

    # 加载Likert概念向量
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    concepts, labels, contents = [], [], []
    for item in raw_concept_data:
        concepts.append(item["concept"])
        labels.append(item["toxic"])
        contents.append(item["content"])

    likert_scores = torch.tensor(concepts, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 加载LLM hidden state
    hidden_states = torch.load(hidden_path, map_location="cpu", weights_only=True)

    # 校验数据一致性
    if likert_scores.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"数据量不一致: Likert {likert_scores.shape[0]} vs Hidden {hidden_states.shape[0]}"
        )

    print(f"  加载 {mode} 数据: {likert_scores.shape[0]} 条, "
          f"Likert维度: {likert_scores.shape[1]}, "
          f"Hidden维度: {hidden_states.shape[1]}")

    return likert_scores, hidden_states, labels_tensor, contents


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses, ib_losses):
    """绘制训练曲线图（三个子图）

    上图: Loss曲线（Val Loss + Test Loss）
    中图: Score曲线（Val F1, Test F1）
    下图: IB损失曲线
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 上图: Loss
    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('ICB-CBM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 中图: Score
    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, val_precisions, color='tab:green', linestyle='--', label='Val Precision')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='-.', label='Test F1')
    ax2.set_ylabel('Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 下图: IB Loss
    ax3.plot(epochs, ib_losses, color='tab:purple', label='IB Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('IB Loss')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train(config, train_dataset, val_dataset, test_dataset, num_concepts, hidden_dim):
    """训练ICB-CBM模型

    基于验证集F1进行早停和最佳模型选择，同时观察测试集F1但不参与模型筛选。

    Args:
        config: ICBConfig配置对象
        train_dataset: 训练集 (likert, hidden, labels)
        val_dataset: 验证集
        test_dataset: 测试集
        num_concepts: 概念数量K
        hidden_dim: LLM hidden state维度

    Returns:
        训练指标历史
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型
    model = ICB_CBM(
        num_concepts=num_concepts,
        hidden_dim=hidden_dim,
        dense_dim=config.dense_dim,
        num_residual=config.num_residual,
        hidden_features=config.hidden_features,
        dropout_rate=config.dropout_rate,
        alpha=config.alpha,
        gamma=config.gamma,
    ).to(device)

    # 计算ICC概念选择掩码
    # 使用训练集的Likert标量和标签
    train_likert = train_dataset.tensors[0]  # [N_train, K]
    train_labels = train_dataset.tensors[2]  # [N_train]
    model.compute_concept_mask(train_likert, train_labels, threshold=config.icc_threshold)

    # 优化器和学习率调度器
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
        three_phase=False
    )

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
    ib_loss_history = []

    for epoch in range(config.epochs):
        # ========== 训练阶段 ==========
        model.train()
        epoch_ib_loss = 0.0
        epoch_train_count = 0

        for likert_batch, hidden_batch, label_batch in train_loader:
            likert_batch = likert_batch.to(device)
            hidden_batch = hidden_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits, gate_weights, mu, logvar = model(likert_batch, hidden_batch)
            loss, loss_dict = model.compute_loss(logits, label_batch, mu, logvar, gate_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_ib_loss += loss_dict["L_IB"]
            epoch_train_count += 1

        avg_ib_loss = epoch_ib_loss / max(epoch_train_count, 1)

        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for likert_batch, hidden_batch, label_batch in val_loader:
                likert_batch = likert_batch.to(device)
                hidden_batch = hidden_batch.to(device)
                label_batch = label_batch.to(device)

                logits, gate_weights, mu, logvar = model(likert_batch, hidden_batch)
                loss, loss_dict = model.compute_loss(logits, label_batch, mu, logvar, gate_weights)
                total_val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(label_batch.cpu().numpy())
                val_count += 1

        avg_val_loss = total_val_loss / max(val_count, 1)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # ========== 测试集评估（仅观察）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        test_count = 0

        with torch.no_grad():
            for likert_batch, hidden_batch, label_batch in test_loader:
                likert_batch = likert_batch.to(device)
                hidden_batch = hidden_batch.to(device)
                label_batch = label_batch.to(device)

                logits, gate_weights, mu, logvar = model(likert_batch, hidden_batch)
                loss, _ = model.compute_loss(logits, label_batch, mu, logvar, gate_weights)
                total_test_loss += loss.item()
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels_list.extend(label_batch.cpu().numpy())
                test_count += 1

        avg_test_loss = total_test_loss / max(test_count, 1)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        # 记录指标
        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        val_precision_history.append(val_p)
        val_recall_history.append(val_r)
        test_f1_history.append(test_f1)
        test_loss_history.append(avg_test_loss)
        ib_loss_history.append(avg_ib_loss)

        print(f"Epoch {epoch + 1}: \n"
              f"  Val Loss = {avg_val_loss:.4f}, Val F1 = {val_f1:.4f}, "
              f"Val P = {val_p:.4f}, Val R = {val_r:.4f}\n"
              f"  Test Loss = {avg_test_loss:.4f}, Test F1 = {test_f1:.4f}\n"
              f"  IB Loss = {avg_ib_loss:.4f}")

        # ========== 最佳模型选择与早停 ==========
        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"  >>> 发现更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
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
            val_recall_history, test_f1_history, test_loss_history, ib_loss_history)


def evaluate(config, timestamp):
    """评估指定实验的最佳模型在测试集上的表现

    Args:
        config: ICBConfig配置对象
        timestamp: 实验时间戳
    """
    experiment_dir = Path(config.base_path) / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    # 从实验目录加载训练时保存的配置
    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_likert, test_hidden, test_labels, contents = load_data(saved_config, "test")

    # 构建测试数据集
    test_dataset = TensorDataset(test_likert, test_hidden, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size),
                              shuffle=False)

    # 初始化模型
    model = ICB_CBM(
        num_concepts=test_likert.shape[1],
        hidden_dim=test_hidden.shape[1],
        dense_dim=int(saved_config.dense_dim),
        num_residual=int(saved_config.num_residual),
        hidden_features=int(saved_config.hidden_features),
        dropout_rate=float(saved_config.dropout_rate),
        alpha=float(saved_config.alpha),
        gamma=float(saved_config.gamma),
    )

    model.load_state_dict(
        torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False)
    )
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    all_gate_weights = []

    with torch.no_grad():
        for likert_batch, hidden_batch, label_batch in test_loader:
            likert_batch = likert_batch.to(device)
            hidden_batch = hidden_batch.to(device)

            logits, gate_weights, mu, logvar = model(likert_batch, hidden_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.numpy())
            all_gate_weights.append(gate_weights.cpu())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 计算稀疏性指标
    all_gate = torch.cat(all_gate_weights, dim=0)  # [N_test, total_dim]
    avg_active_concepts = (all_gate > 0.5).float().mean(dim=0).sum().item()

    # 输出到控制台
    print("\n" + "=" * 30)
    print("      ICB-CBM 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print(f"平均激活概念数: {avg_active_concepts:.1f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    # 持久化保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # 保存评估指标
    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
            "avg_active_concepts": round(avg_active_concepts, 1),
        }, f, indent=2, ensure_ascii=False)

    # 保存分类报告
    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("ICB-CBM 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write(f"平均激活概念数: {avg_active_concepts:.1f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    # 保存逐条预测结果
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
        # 获取完整参数配置
        config = update_ICBConfig(args)

        # 生成时间戳并创建实验目录（带分支名标识）
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-icb-cbm"
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存完整配置到config.json
        config_dict = {
            "timestamp": timestamp,
            "branch": "feature/icb-cbm",
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "train_hidden_path": str(config.train_hidden_path),
            "test_hidden_path": str(config.test_hidden_path),
            "processed_path": str(config.processed_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "dense_dim": config.dense_dim,
            "num_residual": config.num_residual,
            "hidden_features": config.hidden_features,
            "dropout_rate": config.dropout_rate,
            "layer": config.layer,
            "alpha": config.alpha,
            "gamma": config.gamma,
            "icc_threshold": config.icc_threshold,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "max_lr": config.max_lr,
            "pct_start": config.pct_start,
            "div_factor": config.div_factor,
            "final_div_factor": config.final_div_factor,
            "anneal_strategy": config.anneal_strategy,
            "patience": config.patience,
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        # 是否启用确定性模式
        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")
        else:
            print(">>> 已禁用确定性模式")

        # 加载数据
        print("\n>>> 加载数据...")
        train_likert, train_hidden, train_labels, _ = load_data(config, "train")
        test_likert, test_hidden, test_labels, _ = load_data(config, "test")

        num_concepts = train_likert.shape[1]
        hidden_dim = train_hidden.shape[1]

        print(f"\n>>> 数据概览:")
        print(f"  概念数量 K = {num_concepts}")
        print(f"  Hidden dim = {hidden_dim}")
        print(f"  稠密向量维度 d = {config.dense_dim}")
        print(f"  残差概念数量 K_r = {config.num_residual}")
        print(f"  总概念表示维度 = {num_concepts * config.dense_dim + config.num_residual}")

        # 从训练集中按9:1比例划分验证集（分层抽样）
        train_likert_np = train_likert.numpy()
        train_hidden_np = train_hidden.numpy()
        train_labels_np = train_labels.numpy()

        (train_likert_np, val_likert_np,
         train_hidden_np, val_hidden_np,
         train_labels_np, val_labels_np) = train_test_split(
            train_likert_np, train_hidden_np, train_labels_np,
            test_size=0.1, stratify=train_labels_np, random_state=config.seed
        )

        train_dataset = TensorDataset(
            torch.tensor(train_likert_np, dtype=torch.float32),
            torch.tensor(train_hidden_np, dtype=torch.float32),
            torch.tensor(train_labels_np, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(val_likert_np, dtype=torch.float32),
            torch.tensor(val_hidden_np, dtype=torch.float32),
            torch.tensor(val_labels_np, dtype=torch.long),
        )
        test_dataset = TensorDataset(test_likert, test_hidden, test_labels)

        print(f"\n>>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, "
              f"测试集: {len(test_dataset)}")

        # 训练
        print("\n>>> 开始训练...")
        metrics = train(config, train_dataset, val_dataset, test_dataset,
                       num_concepts, hidden_dim)

        # 绘制训练曲线图
        plot_metrics(config, *metrics)

        # all模式下执行测试
        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = ICBConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
