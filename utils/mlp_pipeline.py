"""MLP训练与测试。

整合训练和测试功能,实现训练完成后自动测试的流水线。
超参数统一在 MLP_config.py 中配置,命令行仅指定数据集和模型。

使用示例:
    # 1. 训练+测试
    python utils/mlp_pipeline.py --mode all --dataset_name TOXICN --model_name glm-4-9b-chat

    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/mlp_pipeline.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat --timestamp 20260415-085433

命令行参数说明:
    --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
    --timestamp         测试模式时的实验时间戳 (如: 20260415-085433)
    --dataset_name      数据集名称 (TOXICN/COLD), 必填
    --model_name        LLM模型名称, 必填
    --concept_type      概念向量类型: likert (默认) 或 binary

超参数配置:
    所有训练超参数（学习率、批次大小、epoch数、dropout等）均在 MLP_config.py 中配置。
    测试模式强制使用实验目录的 config.json,忽略当前配置。

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
    1. 运行前需确保已生成概念向量文件 (使用scripts/generate_adjective_c_r_vllm.py)
    2. 测试模式必须指定有效的实验时间戳
"""

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
import numpy as np
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
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')

    # 概念向量类型
    parser.add_argument('--concept_type', type=str, default='likert', choices=['likert', 'binary', 'intent', 'dual', 'dual_interact', 'pragmatic'], help='概念向量类型: likert (相关度), binary (是否评分), intent (意图信号), dual (双信号拼接, 354维), dual_interact (双信号+差异+交互特征, 708维), pragmatic (语用轴, 8维设计正交)')

    return parser.parse_args()


def update_MLPConfig(args):
    """基于命令行参数更新配置对象"""
    mlp_config = MLPConfig()  # MLP_config.py中的配置对象

    # 数据集配置
    mlp_config.dataset_name = args.dataset_name
    mlp_config.model_name = args.model_name

    # 动态生成依赖 dataset_name/model_name 的路径
    concept_type = getattr(args, 'concept_type', 'likert')

    # 构建文件名后缀：形容词词典版本 + concept_type后缀
    # 从词典文件名提取版本（如 toxic_adjectives_v1.csv → v1），与generate脚本命名规则一致
    adj_stem = mlp_config.adjective_path.stem  # toxic_adjectives_v1
    adj_version = adj_stem.replace("toxic_adjectives_", "")  # v1
    if concept_type == 'pragmatic':
        # 语用轴概念向量：文件名后缀为_pragmatic，不依赖形容词词典版本
        suffix = '_pragmatic'
    else:
        suffix = f"_{adj_version}"
        if concept_type == 'binary':
            suffix += '_binary'
        elif concept_type == 'intent':
            suffix += '_intent'

    # dual/dual_interact模式：主路径为likert，辅助路径为intent
    # 主路径(train_concept_path/test_concept_path)指向likert文件
    # 辅助路径(train_concept_path_intent/test_concept_path_intent)指向intent文件
    intent_suffix = f"_{adj_version}_intent"

    mlp_config.train_concept_path = (mlp_config.processed_path / mlp_config.dataset_name
                                     / mlp_config.model_name / f"concept_train_{mlp_config.model_name}{suffix}.json")
    mlp_config.test_concept_path = (mlp_config.processed_path / mlp_config.dataset_name
                                    / mlp_config.model_name / f"concept_test_{mlp_config.model_name}{suffix}.json")

    # dual/dual_interact模式下额外保存intent路径
    if concept_type in ('dual', 'dual_interact'):
        mlp_config.train_concept_path_intent = (mlp_config.processed_path / mlp_config.dataset_name
                                                / mlp_config.model_name / f"concept_train_{mlp_config.model_name}{intent_suffix}.json")
        mlp_config.test_concept_path_intent = (mlp_config.processed_path / mlp_config.dataset_name
                                               / mlp_config.model_name / f"concept_test_{mlp_config.model_name}{intent_suffix}.json")

    mlp_config.concept_type = concept_type
    return mlp_config


def load_data(config, mode):
    """加载指定训练或测试的概念向量和标签。

    概念向量文件中已包含 toxic 标签字段，无需再加载原始数据集。
    dual模式下加载likert+intent两个文件并拼接为354维向量。
    启用length_norm时，按文本长度归一化概念向量以消除长文本激活膨胀。

    Args:
        config: 配置文件（含concept_type字段）
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

    # 加载主概念向量文件（likert或intent或binary）
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    # 是否启用长度归一化（消除长文本概念激活膨胀）
    length_norm = getattr(config, 'length_norm', False)

    concepts, labels = [], []
    for item in raw_concept_data:
        vec = item["concept"]
        if length_norm:
            # 归一化到等效长度20字: x_norm = x * log(21) / log(len+1)
            # 长文本(100字) → 概念值缩小约1.3倍；短文本(10字) → 放大约1.4倍
            content_len = len(item["content"])
            norm_factor = np.log(content_len + 1) / np.log(21)
            vec = [v / norm_factor for v in vec]
        concepts.append(vec)
        labels.append(item["toxic"])

    # dual/dual_interact模式：加载intent概念向量
    concept_type = getattr(config, 'concept_type', 'likert')
    if concept_type in ('dual', 'dual_interact'):
        if mode == "train":
            intent_path = config.train_concept_path_intent
        else:
            intent_path = config.test_concept_path_intent

        with open(intent_path, "r", encoding="utf-8") as f:
            intent_data = json.load(f)

        if concept_type == 'dual':
            # 简单拼接：[likert_177维, intent_177维] → 354维
            for i, item in enumerate(intent_data):
                intent_vec = item["concept"]
                if length_norm:
                    content_len = len(item["content"])
                    norm_factor = np.log(content_len + 1) / np.log(21)
                    intent_vec = [v / norm_factor for v in intent_vec]
                concepts[i] = concepts[i] + intent_vec
        else:
            # dual_interact: 差异特征显式编码
            # [likert, intent, likert-intent, likert*intent] → 708维
            # 显式提供差异信号（相关但不表达 vs 表达但不相关）和交互信号（既相关又表达）
            for i, item in enumerate(intent_data):
                likert_vec = np.array(concepts[i], dtype=np.float32)
                intent_vec = np.array(item["concept"], dtype=np.float32)
                if length_norm:
                    content_len = len(item["content"])
                    norm_factor = np.log(content_len + 1) / np.log(21)
                    intent_vec = intent_vec / norm_factor
                diff_vec = likert_vec - intent_vec
                interact_vec = likert_vec * intent_vec
                concepts[i] = np.concatenate([likert_vec, intent_vec, diff_vec, interact_vec]).tolist()

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
    gate_type = getattr(config, 'gate_type', 'matrix')
    gate_init = None

    # 矩阵门控和全局门控：从训练集计算 Cohen's d 作为门控初始化先验
    if gate_type in ('global', 'matrix'):
        train_x_all = train_dataset[:][0].numpy()
        train_y_all = train_dataset[:][1].numpy()
        toxic_mask = train_y_all == 1
        non_mask = train_y_all == 0
        toxic_mean = train_x_all[toxic_mask].mean(axis=0)
        non_mean = train_x_all[non_mask].mean(axis=0)
        toxic_std = train_x_all[toxic_mask].std(axis=0)
        non_std = train_x_all[non_mask].std(axis=0)
        pooled_std = np.sqrt((toxic_std**2 + non_std**2) / 2)
        pooled_std = np.where(pooled_std < 1e-8, 1e-8, pooled_std)
        gate_init = (toxic_mean - non_mean) / pooled_std
        if gate_type == 'global':
            print(f">>> 全局门控初始化: Cohen's d 均值={gate_init.mean():.4f}, d>0概念数={int((gate_init>0).sum())}/{len(gate_init)}")
        else:
            print(f">>> 矩阵门控偏置初始化: Cohen's d 均值={gate_init.mean():.4f}, d>0概念数={int((gate_init>0).sum())}/{len(gate_init)}")

    model = MLP(
        in_features=train_dataset[0][0].shape[0],
        dropout_rate=config.dropout_rate,
        hidden_features=config.hidden_features,
        gate_type=gate_type,
        gate_init=gate_init
    ).to(device)
    gate_l1_lambda = getattr(config, 'gate_l1_lambda', 0.0)
    print(f">>> 使用MLP (gate_type={gate_type}, L1_lambda={gate_l1_lambda})")

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
            # 门控L1正则化：强制稀疏性，防止门控学习噪声模式
            if gate_l1_lambda > 0:
                l1_loss = model.get_gate_l1_loss()
                loss = loss + gate_l1_lambda * l1_loss
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
    # 训练时experiment_path已含时间戳，测试时需要拼接
    if timestamp and config.experiment_path.name != timestamp:
        experiment_dir = config.experiment_path / timestamp
    else:
        experiment_dir = config.experiment_path
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
    concept_vectors = [item["concept"] for item in raw_concept_data]

    # dual/dual_interact模式：加载intent概念向量并构造特征
    concept_type = getattr(saved_config, 'concept_type', 'likert')
    if concept_type in ('dual', 'dual_interact'):
        with open(saved_config.test_concept_path_intent, "r", encoding="utf-8") as f:
            intent_concept_data = json.load(f)

        if concept_type == 'dual':
            # 简单拼接：354维
            for i, item in enumerate(intent_concept_data):
                concept_vectors[i] = concept_vectors[i] + item["concept"]
        else:
            # dual_interact: 差异特征显式编码，708维
            for i, item in enumerate(intent_concept_data):
                likert_vec = np.array(concept_vectors[i], dtype=np.float32)
                intent_vec = np.array(item["concept"], dtype=np.float32)
                diff_vec = likert_vec - intent_vec
                interact_vec = likert_vec * intent_vec
                concept_vectors[i] = np.concatenate([likert_vec, intent_vec, diff_vec, interact_vec]).tolist()

    # 加载概念维度命名（用于结果可解释性）
    import csv
    adjective_names = []
    adjective_chinese = []
    if concept_type == 'pragmatic':
        # 语用轴：从pragmatic_axes.csv加载轴名
        axes_path = Path(saved_config.adjective_path).parent / "pragmatic_axes.csv"
        with open(axes_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                adjective_names.append(row[0])  # axis_name
                adjective_chinese.append(row[1] if len(row) > 1 else row[0])  # axis_chinese
    else:
        with open(saved_config.adjective_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                adjective_names.append(row[0])
                adjective_chinese.append(row[1] if len(row) > 1 else row[0])

    # 加载最佳模型
    model = MLP(
        in_features=test_x.shape[1],
        dropout_rate=saved_config.dropout_rate,
        hidden_features=saved_config.hidden_features,
        gate_type=getattr(saved_config, 'gate_type', 'matrix'),
        gate_init=None  # 测试时权重从checkpoint加载，无需初始化
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理（同时提取门控值和概率）
    all_preds, all_labels, all_probs, all_gates = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs, gate_weights = model(batch_x, return_gate=True)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
            if gate_weights is not None:
                all_gates.extend(gate_weights.cpu().numpy())
            else:
                # 无门控模式：用零向量填充以保持数据结构一致
                all_gates.extend(np.zeros((batch_x.shape[0], test_x.shape[1])))

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

    # 保存逐条预测结果 JSON（含概念向量、概率、门控值，用于错误分析）
    label_names = ["Non-Toxic", "Toxic"]
    predictions = []
    for i in range(len(all_preds)):
        # 构建概念-门控映射（仅保存非零门控，减少文件大小）
        gate_dict = {}
        for j, adj_name in enumerate(adjective_names):
            if all_gates[i][j] > 0.01:  # 只保存门控值 > 0.01 的概念
                gate_dict[f"{adj_name}({adjective_chinese[j]})"] = round(float(all_gates[i][j]), 4)

        predictions.append({
            "index": i,
            "content": contents[i],
            "true_label": int(all_labels[i]),
            "true_label_name": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_label_name": label_names[int(all_preds[i])],
            "correct": bool(all_preds[i] == all_labels[i]),
            "probabilities": {
                "non_toxic": round(float(all_probs[i][0]), 4),
                "toxic": round(float(all_probs[i][1]), 4)
            },
            "concept_vector": [round(float(v), 4) for v in concept_vectors[i]],
            "gate_values": gate_dict
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
            # 实验元信息
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "concept_type": getattr(args, 'concept_type', 'likert'),

            # 数据与词典
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "adjective_path": str(config.adjective_path),
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "concept_dim": config.train_concept_path.stem,  # 概念向量文件名（含维度信息）

            # dual模式下额外保存intent路径（用于测试时加载双信号）
            "train_concept_path_intent": str(getattr(config, 'train_concept_path_intent', '')),
            "test_concept_path_intent": str(getattr(config, 'test_concept_path_intent', '')),

            # 随机种子
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,

            # 训练超参数
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "max_lr": config.max_lr,
            "pct_start": config.pct_start,
            "div_factor": config.div_factor,
            "final_div_factor": config.final_div_factor,
            "anneal_strategy": config.anneal_strategy,

            # 模型结构
            "dropout_rate": config.dropout_rate,
            "hidden_features": config.hidden_features,
            "patience": config.patience,

            # 门控配置
            "gate_type": getattr(config, 'gate_type', 'matrix'),
            "gate_l1_lambda": getattr(config, 'gate_l1_lambda', 0.0),
            "length_norm": getattr(config, 'length_norm', False),
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
