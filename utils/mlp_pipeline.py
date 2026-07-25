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
from models.mlp import MLP, FormConditionedMLP, SimpleMLP, FormConditionedSimpleMLP
from utils.concept_features import extract_concept_features

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
    parser.add_argument('--concept_type', type=str, default='likert', choices=['likert', 'binary'], help='概念向量类型: likert (相关度), binary (是否评分)')

    # 形容词词典
    parser.add_argument('--adjective_name', type=str, default=None, help='形容词词典文件名（如toxic_adjectives_v2.csv），默认使用MLP_config.py中的adjective_path')

    # 概念向量后缀（用于SNR加权等变体）
    parser.add_argument('--concept_suffix', type=str, default=None, help='概念向量文件额外后缀 (如 _snr_weighted)')

    # 概念向量格式
    parser.add_argument('--concept_format', type=str, default='typed', choices=['legacy', 'typed'], help='概念向量格式: legacy (旧格式, item["concept"]), typed (新格式, level_probs+concept_types)')

    # 模型类型
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'simple_mlp', 'form_conditioned_mlp', 'form_conditioned_simple_mlp'], help='模型类型: mlp (门控MLP), simple_mlp (无门控MLP), form_conditioned_mlp (Form条件化门控MLP), form_conditioned_simple_mlp (Form条件化偏置MLP)')

    return parser.parse_args()


def update_MLPConfig(args):
    """基于命令行参数更新配置对象"""
    mlp_config = MLPConfig()

    # 数据集配置
    mlp_config.dataset_name = args.dataset_name
    mlp_config.model_name = args.model_name

    # 动态生成依赖 dataset_name/model_name 的路径
    concept_type = getattr(args, 'concept_type', 'likert')

    # 形容词词典路径：命令行指定 > MLP_config.py默认值
    if getattr(args, 'adjective_name', None) is not None:
        mlp_config.adjective_path = mlp_config.raw_data_path / "adjective" / args.adjective_name

    # 构建文件名后缀：形容词词典版本 + concept_type后缀 + concept_suffix
    adj_stem = mlp_config.adjective_path.stem  # toxic_adjectives_v4
    adj_version = adj_stem.replace("toxic_adjectives_", "")  # v4
    concept_format = getattr(args, 'concept_format', 'legacy')
    if concept_format == 'typed':
        suffix = f"_typed_{adj_version}"
    else:
        suffix = f"_{adj_version}"
    if concept_type == 'binary':
        suffix += '_binary'
    if getattr(args, 'concept_suffix', None):
        suffix += f"_{args.concept_suffix}"  # e.g., _snr_weighted

    mlp_config.train_concept_path = (mlp_config.processed_path / mlp_config.dataset_name
                                     / mlp_config.model_name / f"concept_train_{mlp_config.model_name}{suffix}.json")
    mlp_config.test_concept_path = (mlp_config.processed_path / mlp_config.dataset_name
                                    / mlp_config.model_name / f"concept_test_{mlp_config.model_name}{suffix}.json")

    mlp_config.concept_type = concept_type
    mlp_config.concept_format = concept_format

    # 模型类型
    mlp_config.model_type = args.model_type

    # Form-Conditioned Gate: 自动推导form概念向量路径
    if args.model_type in ('form_conditioned_mlp', 'form_conditioned_simple_mlp'):
        mlp_config.train_form_concept_path = (
            mlp_config.processed_path / mlp_config.dataset_name
            / mlp_config.model_name
            / f"concept_train_{mlp_config.model_name}_text_form.json"
        )
        mlp_config.test_form_concept_path = (
            mlp_config.processed_path / mlp_config.dataset_name
            / mlp_config.model_name
            / f"concept_test_{mlp_config.model_name}_text_form.json"
        )
        # 检查form概念文件是否存在
        if not mlp_config.train_form_concept_path.exists():
            raise FileNotFoundError(f"训练集form概念文件不存在: {mlp_config.train_form_concept_path}")
        if not mlp_config.test_form_concept_path.exists():
            raise FileNotFoundError(f"测试集form概念文件不存在: {mlp_config.test_form_concept_path}")

    return mlp_config


def load_data(config, mode):
    """加载指定训练或测试的概念向量和标签。

    支持两种概念向量JSON格式：
    - legacy格式：扁平列表 [{content, toxic, concept: [float]}, ...]
    - typed格式：嵌套结构 {meta: {concept_types: [...]}, data: [{content, toxic, concept_scores, level_probs}]}

    typed格式使用 concept_features.py 提取特征，支持 single/conditional/all_probs 三种模式。

    Args:
        config: 配置文件（含concept_type, concept_format, concept_feat_mode字段）
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

    with open(concept_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 检测格式：typed格式有meta和data键，legacy格式是扁平列表
    concept_format = getattr(config, 'concept_format', 'legacy')

    if concept_format == 'typed' or (isinstance(raw_data, dict) and 'meta' in raw_data and 'data' in raw_data):
        # typed格式：使用concept_features.py提取特征
        meta = raw_data['meta']
        data = raw_data['data']
        concept_types = meta['concept_types']
        feat_mode = getattr(config, 'concept_feat_mode', 'single')

        X, y, feature_names = extract_concept_features(data, concept_types, mode=feat_mode)
        print(f"  [typed格式] 特征模式={feat_mode}, 特征维度={X.shape[1]} (概念数={len(concept_types)})")
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    else:
        # legacy格式：直接读取concept字段
        concepts, labels = [], []
        for item in raw_data:
            concepts.append(item["concept"])
            labels.append(item["toxic"])
        return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def load_dual_data(config, mode):
    """同时加载语义概念向量和文本形式概念向量。

    两个文件样本顺序完全一致，按索引对齐。

    Args:
        config: 配置文件（含train/test_form_concept_path字段）
        mode: train/test

    Returns:
        tuple: (sem_concepts, form_concepts, labels) 三个张量
    """
    if mode == "train":
        sem_path = config.train_concept_path
        form_path = config.train_form_concept_path
    elif mode == "test":
        sem_path = config.test_concept_path
        form_path = config.test_form_concept_path
    else:
        raise ValueError("mode must be 'train' or 'test'")

    with open(sem_path, "r", encoding="utf-8") as f:
        sem_data = json.load(f)
    with open(form_path, "r", encoding="utf-8") as f:
        form_data = json.load(f)

    assert len(sem_data) == len(form_data), \
        f"语义文件({len(sem_data)})与form文件({len(form_data)})样本数不一致"

    sem_concepts, form_concepts, labels = [], [], []
    for i in range(len(sem_data)):
        assert sem_data[i]["content"] == form_data[i]["content"], \
            f"第{i}条样本content不一致"
        assert sem_data[i]["toxic"] == form_data[i]["toxic"], \
            f"第{i}条样本toxic标签不一致"
        sem_concepts.append(sem_data[i]["concept"])
        form_concepts.append(form_data[i]["concept"])
        labels.append(sem_data[i]["toxic"])

    return (torch.tensor(sem_concepts, dtype=torch.float32),
            torch.tensor(form_concepts, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))


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


def train(config, train_dataset, val_dataset, test_dataset,
          model_type="mlp", form_datasets=None):
    """训练MLP模型。

    基于验证集F1进行早停和最佳模型选择，同时观察测试集F1但不参与模型筛选。

    :param config: MLPConfig 配置对象
    :param train_dataset: 训练集 (sem, labels) 或 (sem, labels)
    :param val_dataset: 验证集 (sem, labels)
    :param test_dataset: 测试集 (sem, labels)，仅观察F1变化，不参与模型筛选
    :param model_type: 模型类型 "mlp"/"simple_mlp"/"form_conditioned_mlp"/"form_conditioned_simple_mlp"
    :param form_datasets: form概念数据字典 {"train": tensor, "val": tensor, "test": tensor}，仅form模型使用
    :return: (epochs, val_losses, val_f1_scores, val_precisions, val_recalls, test_f1_scores, test_losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    in_features = train_dataset[0][0].shape[0]

    # 是否需要form特征
    is_form_model = model_type in ("form_conditioned_mlp", "form_conditioned_simple_mlp")

    # 构建DataLoader
    if is_form_model:
        train_sem, train_y = train_dataset.tensors
        val_sem, val_y = val_dataset.tensors
        test_sem, test_y = test_dataset.tensors

        train_form = form_datasets["train"]
        val_form = form_datasets["val"]
        test_form = form_datasets["test"]

        train_loader = DataLoader(
            TensorDataset(train_sem, train_form, train_y),
            batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_sem, val_form, val_y),
            batch_size=config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(test_sem, test_form, test_y),
            batch_size=config.batch_size, shuffle=False
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型
    if model_type == "form_conditioned_mlp":
        model = FormConditionedMLP(
            in_features=in_features,
            form_dim=config.form_dim,
            dropout_rate=config.dropout_rate,
            hidden_features=config.hidden_features,
        ).to(device)
        print(f">>> 使用FormConditionedMLP (Form条件化门控)")
    elif model_type == "form_conditioned_simple_mlp":
        model = FormConditionedSimpleMLP(
            in_features=in_features,
            form_dim=config.form_dim,
            dropout_rate=config.dropout_rate,
            hidden_features=config.hidden_features,
        ).to(device)
        print(f">>> 使用FormConditionedSimpleMLP (Form条件化偏置)")
    elif model_type == "simple_mlp":
        model = SimpleMLP(
            in_features=in_features,
            dropout_rate=config.dropout_rate,
            hidden_features=config.hidden_features,
        ).to(device)
        print(f">>> 使用SimpleMLP (无门控)")
    else:
        model = MLP(
            in_features=in_features,
            dropout_rate=config.dropout_rate,
            hidden_features=config.hidden_features,
        ).to(device)
        print(f">>> 使用MLP (矩阵门控)")

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
        for batch_data in train_loader:
            if is_form_model:
                batch_sem, batch_form, batch_y = batch_data
                batch_sem, batch_form, batch_y = batch_sem.to(device), batch_form.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_sem, batch_form)
            else:
                batch_x, batch_y = batch_data
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
            for batch_data in val_loader:
                if is_form_model:
                    val_sem, val_form, val_y = batch_data
                    val_sem, val_form, val_y = val_sem.to(device), val_form.to(device), val_y.to(device)
                    val_outputs = model(val_sem, val_form)
                    v_loss = criterion(val_outputs, val_y)
                else:
                    val_x, val_y = batch_data
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
            for batch_data in test_loader:
                if is_form_model:
                    t_sem, t_form, t_y = batch_data
                    t_sem, t_form = t_sem.to(device), t_form.to(device)
                    t_outputs = model(t_sem, t_form)
                    t_loss = criterion(t_outputs, t_y.to(device))
                else:
                    tx, ty = batch_data
                    tx = tx.to(device)
                    t_outputs = model(tx)
                    t_loss = criterion(t_outputs, ty.to(device))
                total_test_loss += t_loss.item()
                test_preds.extend(torch.argmax(t_outputs, dim=1).cpu().numpy())
                if is_form_model:
                    test_labels_list.extend(t_y.numpy())
                else:
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

    # 从保存的配置中获取模型类型（兼容旧实验）
    model_type = getattr(saved_config, 'model_type', 'mlp')
    is_form_model = model_type in ('form_conditioned_mlp', 'form_conditioned_simple_mlp')

    # 加载测试数据
    test_x, test_y = load_data(saved_config, "test")

    # 构建DataLoader和模型
    if is_form_model:
        _, test_form_x, _ = load_dual_data(saved_config, "test")
        test_loader = DataLoader(
            TensorDataset(test_x, test_form_x, test_y),
            batch_size=int(saved_config.batch_size), shuffle=False
        )
    else:
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=int(saved_config.batch_size), shuffle=False)

    # 从概念向量文件中加载原始文本内容（逐条保存预测结果）
    with open(saved_config.test_concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    # 兼容两种格式：typed(嵌套) / legacy(扁平)
    if isinstance(raw_concept_data, dict) and 'data' in raw_concept_data:
        raw_items = raw_concept_data['data']
        concept_vectors = [item["concept_scores"] for item in raw_items]
    else:
        raw_items = raw_concept_data
        concept_vectors = [item["concept"] for item in raw_items]
    contents = [item["content"] for item in raw_items]

    # 加载概念维度命名（用于结果可解释性）
    import csv
    adjective_names = []
    adjective_chinese = []
    with open(saved_config.adjective_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            adjective_names.append(row[0])
            adjective_chinese.append(row[1] if len(row) > 1 else row[0])

    # 加载最佳模型
    if model_type == 'form_conditioned_mlp':
        model = FormConditionedMLP(
            in_features=test_x.shape[1],
            form_dim=getattr(saved_config, 'form_dim', 10),
            dropout_rate=saved_config.dropout_rate,
            hidden_features=saved_config.hidden_features,
        )
    elif model_type == 'form_conditioned_simple_mlp':
        model = FormConditionedSimpleMLP(
            in_features=test_x.shape[1],
            form_dim=getattr(saved_config, 'form_dim', 10),
            dropout_rate=saved_config.dropout_rate,
            hidden_features=saved_config.hidden_features,
        )
    elif model_type == 'simple_mlp':
        model = SimpleMLP(
            in_features=test_x.shape[1],
            dropout_rate=saved_config.dropout_rate,
            hidden_features=saved_config.hidden_features,
        )
    else:
        model = MLP(
            in_features=test_x.shape[1],
            dropout_rate=saved_config.dropout_rate,
            hidden_features=saved_config.hidden_features,
        )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if is_form_model:
                batch_sem, batch_form, batch_y = batch_data
                outputs = model(batch_sem.to(device), batch_form.to(device))
            else:
                batch_x, batch_y = batch_data
                outputs = model(batch_x.to(device))
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

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

    # 保存逐条预测结果 JSON（含概念向量、概率）
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
            "correct": bool(all_preds[i] == all_labels[i]),
            "probabilities": {
                "non_toxic": round(float(all_probs[i][0]), 4),
                "toxic": round(float(all_probs[i][1]), 4)
            },
            "concept_vector": [round(float(v), 4) for v in concept_vectors[i]],
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

        # 是否需要form特征
        is_form_model = config.model_type in ('form_conditioned_mlp', 'form_conditioned_simple_mlp')

        # 保存完整配置到config.json
        config_dict = {
            # 实验元信息
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "concept_type": getattr(args, 'concept_type', 'likert'),
            "concept_format": getattr(args, 'concept_format', 'legacy'),
            "concept_feat_mode": config.concept_feat_mode,
            "concept_suffix": getattr(args, 'concept_suffix', None),
            "model_type": config.model_type,

            # 数据与词典
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "adjective_path": str(config.adjective_path),
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "concept_dim": config.train_concept_path.stem,

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
            "form_dim": config.form_dim,
        }

        # Form-Conditioned额外路径
        if is_form_model:
            config_dict["train_form_concept_path"] = str(config.train_form_concept_path)
            config_dict["test_form_concept_path"] = str(config.test_form_concept_path)
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
        if is_form_model:
            train_sem, train_form, train_y = load_dual_data(config, "train")
            test_sem, test_form, test_y = load_dual_data(config, "test")

            # 用索引切分确保sem和form使用相同train/val划分
            indices = np.arange(len(train_sem))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.1, stratify=train_y.numpy(), random_state=config.seed
            )

            train_dataset = TensorDataset(train_sem[train_idx], train_y[train_idx])
            val_dataset = TensorDataset(train_sem[val_idx], train_y[val_idx])
            test_dataset = TensorDataset(test_sem, test_y)

            form_datasets = {
                "train": train_form[train_idx],
                "val": train_form[val_idx],
                "test": test_form,
            }
        else:
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
            form_datasets = None

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        # 训练并获取指标
        metrics = train(config, train_dataset, val_dataset, test_dataset,
                        model_type=config.model_type, form_datasets=form_datasets)

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
