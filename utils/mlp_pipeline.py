"""MLP训练与测试。

整合训练和测试功能,实现训练完成后自动测试的流水线。
支持命令行参数配置,确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试
    python utils/mlp_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-GPTQ-Int8 --epochs 500 --patience 20
    
    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/mlp_pipeline.py --mode test --timestamp 20260415-085433

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳 (如: 20260415-085433)
    
    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)
        --model_name        LLM模型名称 (默认: Qwen2.5-1.5B-Instruct)
        --template          提示词模板类型 (已废弃，保留参数仅兼容旧实验)
    
    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (确保实验可复现，默认：False)
    
    训练超参数:
        --batch_size        批次大小 (默认: 16)
        --epochs            训练轮数 (默认: 200)
        --max_lr            峰值学习率 (默认: 1e-3)
        --pct_start         Warmup比例 (默认: 0.2)
        --div_factor        初始学习率除数 (默认: 25.0)
        --final_div_factor  最终学习率除数 (默认: 10000.0)
        --anneal_strategy   衰减策略: cos (余弦) 或 linear (线性), 默认: cos
    
    MLP模型结构参数:
        --dropout_rate      Dropout比率 (默认: 0.3)
        --hidden_features   隐藏层维度 (默认: 96)
        --patience          早停耐心值 (默认: 20)

参数优先级:
    - 训练模式: 命令行参数 > MLP_config.py（命令行参数覆盖MLP_config参数）
    - 测试模式: 强制使用实验目录的 config.json (忽略命令行超参数)

输出文件:
    实验目录结构 (experiments/<timestamp>/):
        ├── config.json              # 实验配置快照
        ├── best_model.pth           # 最佳模型权重
        ├── metrics.png              # 训练曲线图
        └── test_results/            # 测试结果目录 (仅 all/test 模式)
            ├── metrics.json         # 测试集评估指标
            ├── classification_report.txt  # 详细分类报告
            └── predictions.json     # 逐条预测结果

注意事项:
    1. 运行前需确保已生成概念向量文件 (使用scripts/generate_adjective_c_r.py)
    2. 测试模式必须指定有效的实验时间戳
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

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig
from models.mlp import MLP

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MLP 训练与测试统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""""
    )

    # 运行模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'train', 'test'],
        default='all',
        help='运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)'
    )

    # 测试模式必需参数
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='测试模式时的实验时间戳 (如: 20260415-085433)'
    )

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='TOXICN', help='数据集名称 (TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-1.5B-Instruct', help='LLM模型名称')
    parser.add_argument('--template', type=str, default='likert',help='提示词模板类型（已废弃，保留仅兼容旧实验）')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False, help='启用确定性模式')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_lr', type=float, default=None, help='峰值学习率')
    parser.add_argument('--pct_start', type=float, default=None, help='Warmup比例')
    parser.add_argument('--div_factor', type=float, default=None, help='初始学习率除数')
    parser.add_argument('--final_div_factor', type=float, default=None, help='最终学习率除数')
    parser.add_argument('--anneal_strategy', type=str, default=None, help='衰减策略 (cos/linear)')

    # 模型结构参数
    parser.add_argument('--dropout_rate', type=float, default=None, help='Dropout比率')
    parser.add_argument('--hidden_features', type=int, default=None, help='隐藏层维度')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值 (验证集F1连续patience个epoch未提升则停止)')

    return parser.parse_args()


def update_MLPConfig(args):
    """基于MLP_config参数，根据命令行参数更新配置对象

    优先级: 命令行参数 > MLPConfig默认值
    """
    mlp_config = MLPConfig()  # MLP_config.py中的配置对象

    # 数据集与模板配置
    mlp_config.dataset_name = args.dataset_name
    mlp_config.model_name = args.model_name
    mlp_config.template = args.template

    # 动态生成依赖 dataset_name/model_name/template 的路径
    mlp_config.train_concept_path = (mlp_config.processed_path / mlp_config.dataset_name
                                     / mlp_config.model_name / "concept_train.json")
    mlp_config.test_concept_path = (mlp_config.processed_path / mlp_config.dataset_name
                                    / mlp_config.model_name / "concept_test.json")

    # 随机种子
    if args.seed is not None:
        mlp_config.seed = args.seed

    # 确定性模式
    if args.use_deterministic:  # store_true默认为False，只有显式传入才为True
        mlp_config.use_deterministic = True

    # 训练超参数
    if args.batch_size is not None:
        mlp_config.batch_size = args.batch_size
    if args.epochs is not None:
        mlp_config.epochs = args.epochs
    if args.max_lr is not None:
        mlp_config.max_lr = args.max_lr
    if args.pct_start is not None:
        mlp_config.pct_start = args.pct_start
    if args.div_factor is not None:
        mlp_config.div_factor = args.div_factor
    if args.final_div_factor is not None:
        mlp_config.final_div_factor = args.final_div_factor
    if args.anneal_strategy is not None:
        mlp_config.anneal_strategy = args.anneal_strategy

    # 模型结构参数
    if args.dropout_rate is not None:
        mlp_config.dropout_rate = args.dropout_rate
    if args.hidden_features is not None:
        mlp_config.hidden_features = args.hidden_features
    if args.patience is not None:
        mlp_config.patience = args.patience

    return mlp_config


def load_data(config, mode):
    """加载指定训练或测试的概念向量和标签。

    概念向量文件中已包含 toxic 标签字段，无需再加载原始数据集。

    Args:
        config: 配置文件
        mode: train/test，区分加载训练或实验数据集

    Returns:
        tuple: (concepts, labels) 概念向量和标签张量
    """

    if mode == "train":
        concept_path = config.train_concept_path
    elif mode == "test":
        concept_path = config.test_concept_path
    else:
        raise ValueError("in load_data, mode must be 'train' or 'test'")

    # 加载概念向量文件（已包含 content, toxic, concept 字段）
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    concepts, labels = [], []
    for item in raw_concept_data:
        concepts.append(item["concept"])
        labels.append(item["toxic"])

    return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    """绘制训练曲线图（上下双子图）。

    上图: Loss曲线（Val Loss + Test Loss）
    下图: Score曲线（Val F1, Val Precision, Val Recall, Test F1）

    :param config: MLPConfig 配置对象
    :param epochs: 轮次列表
    :param val_losses: 验证集损失列表
    :param val_f1_scores: 验证集F1列表
    :param val_precisions: 验证集精确率列表
    :param val_recalls: 验证集召回率列表
    :param test_f1_scores: 测试集F1列表
    :param test_losses: 测试集损失列表
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图: Loss
    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('MLP Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 下图: Score
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


def train(config, train_dataset, val_dataset, test_dataset):
    """训练MLP模型。

    基于验证集F1进行早停和最佳模型选择，同时观察测试集F1但不参与模型筛选。

    :param config: MLPConfig 配置对象
    :param train_dataset: 训练集 (concepts, labels)
    :param val_dataset: 验证集 (concepts, labels)
    :param test_dataset: 测试集 (concepts, labels)，仅观察F1变化，不参与模型筛选
    :return: (epochs, val_losses, val_f1_scores, val_precisions, val_recalls, test_f1_scores, test_losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型
    model = MLP(
        in_features=train_dataset[0][0].shape[0],
        dropout_rate=config.dropout_rate,
        hidden_features=config.hidden_features
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

    for epoch in range(config.epochs):
        # ========== 训练阶段 ==========
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # ========== 验证集评估 ==========
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
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # ========== 测试集评估（仅观察，不参与模型筛选）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for tx, ty in test_loader:
                tx = tx.to(device)
                t_outputs = model(tx)
                t_loss = criterion(t_outputs, ty.to(device))
                total_test_loss += t_loss.item()
                test_preds.extend(torch.argmax(t_outputs, dim=1).cpu().numpy())
                test_labels_list.extend(ty.numpy())

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

        print(f"Epoch {epoch + 1}: \n>>>Val Loss = {avg_val_loss:.4f}, \n>>>Val F1 = {val_f1:.4f}, "
              f"\n>>>Val P = {val_p:.4f}, \n>>>Val R = {val_r:.4f}, "
              f"\n>>>Test Loss = {avg_test_loss:.4f}, \n>>>Test F1 = {test_f1:.4f}")

        # ========== 最佳模型选择与早停 ==========
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

    # 保存最佳模型
    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history)


def evaluate(config, timestamp):
    """评估指定实验的最佳模型在测试集上的表现。

    从实验目录加载config.json恢复训练配置，加载最佳模型权重，
    在测试集上计算Precision/Recall/F1，并保存结果。

    :param config: MLPConfig 配置对象（用于获取base_path）
    :param timestamp: 实验时间戳
    """
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    # 从实验目录加载训练时保存的配置
    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    # 恢复训练时的随机种子设置，确保可复现性
    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_x, test_y = load_data(saved_config, "test")
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=int(saved_config.batch_size), shuffle=False)

    # 从概念向量文件中加载原始文本内容（逐条保存预测结果）
    with open(saved_config.test_concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)
    contents = [item["content"] for item in raw_concept_data]

    # 加载最佳模型
    model = MLP(
        in_features=test_x.shape[1],
        dropout_rate=saved_config.dropout_rate,
        hidden_features=saved_config.hidden_features
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 输出到控制台
    print("\n" + "=" * 30)
    print("      MLP 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    # 持久化保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # 保存评估指标 JSON
    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }, f, indent=2, ensure_ascii=False)

    # 保存分类报告 TXT
    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("MLP 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    # 保存逐条预测结果 JSON
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
    """
    参数加载逻辑：训练模式基于MLP_config.py，使用命令行参数更新配置，并保存到config.json中
    测试模式从实验目录的config.json中加载参数配置
    """
    args = parse_args()

    if args.mode in ['all', 'train']:
        # 获取完整参数配置
        config = update_MLPConfig(args)

        # 生成时间戳并创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存完整配置到config.json
        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "template": config.template,
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "processed_path": str(config.processed_path),
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
            "hidden_features": config.hidden_features,
            "patience": config.patience
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        # 是否启用确定性模式
        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式 (Reproducibility Enabled)")
        else:
            print(">>> 已禁用确定性模式 (Randomness Enabled), 结果将不可复现")

        # 加载数据
        train_x, train_y = load_data(config, "train")
        test_x, test_y = load_data(config, "test")

        # 从训练集中按9:1比例划分验证集（分层抽样）
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

        # 训练并获取指标
        metrics = train(config, train_dataset, val_dataset, test_dataset)

        # 绘制训练曲线图
        plot_metrics(config, *metrics)

        # all模式下执行测试
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
