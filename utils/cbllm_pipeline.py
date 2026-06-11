"""CB-LLM-CN训练与测试流水线

Concept Bottleneck LLM for Chinese Toxic Language Detection

支持三种backbone模式：
1. bge: BGE嵌入(768-dim)作为backbone → CBL → 稀疏预测（推荐，同空间对齐）
2. qwen: Qwen hidden state(3584-dim)作为backbone → CBL → 稀疏预测
3. concept_only: ACS概念评分直接作为MLP输入（无瓶颈基线，验证ACS效果）

使用示例:
    # BGE backbone（推荐）
    python utils/cbllm_pipeline.py --mode all --dataset_name TOXICN --backbone bge

    # Qwen backbone
    python utils/cbllm_pipeline.py --mode all --dataset_name TOXICN --backbone qwen

    # ACS概念评分直接分类（无瓶颈基线）
    python utils/cbllm_pipeline.py --mode all --dataset_name TOXICN --backbone concept_only

    # 仅测试
    python utils/cbllm_pipeline.py --mode test --timestamp 20260609-120000
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

from configs.CBLLM_config import CBLLMConfig
from models.cbllm_cn import CBLLM_CN
from models.mlp import MLP

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CB-LLM-CN 训练与测试流水线",
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

    # Backbone模式
    parser.add_argument('--backbone', type=str,
                        choices=['bge', 'qwen', 'concept_only', 'likert_bge'],
                        default='likert_bge',
                        help='Backbone模式: bge(BGE+ACS), qwen(Qwen+ACS), '
                             'concept_only(ACS直接分类), likert_bge(BGE+Likert)')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False,
                        help='启用确定性模式')

    # CB-LLM-CN模型参数（bge/qwen/likert_bge模式）
    parser.add_argument('--cbl_hidden_dim', type=int, default=None,
                        help='CBL中间层维度')
    parser.add_argument('--cbl_dropout', type=float, default=None,
                        help='CBL Dropout比率')
    parser.add_argument('--sparse_l1_weight', type=float, default=None,
                        help='L1稀疏正则化权重')
    parser.add_argument('--prediction_dropout', type=float, default=None,
                        help='预测层Dropout')
    parser.add_argument('--topk', type=int, default=None,
                        help='TopK稀疏激活数量 (0=不使用)')
    parser.add_argument('--use_residual', action='store_true', default=False,
                        help='启用残差连接 (backbone→预测层)')

    # MLP参数（concept_only模式）
    parser.add_argument('--dropout_rate', type=float, default=None,
                        help='MLP Dropout比率 (concept_only模式)')
    parser.add_argument('--hidden_features', type=int, default=None,
                        help='MLP隐藏层维度 (concept_only模式)')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_lr', type=float, default=None, help='峰值学习率')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值')

    # OneCycleLR参数
    parser.add_argument('--pct_start', type=float, default=None, help='Warmup比例')
    parser.add_argument('--div_factor', type=float, default=None, help='初始学习率除数')
    parser.add_argument('--final_div_factor', type=float, default=None,
                        help='最终学习率除数')
    parser.add_argument('--anneal_strategy', type=str, default=None,
                        help='衰减策略 (cos/linear)')

    return parser.parse_args()


def update_CBLLMConfig(args):
    """基于CBLLMConfig默认值，根据命令行参数更新配置对象"""
    config = CBLLMConfig()

    # 数据集配置
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    config.backbone = args.backbone

    # 动态生成路径
    config.train_concept_path = (config.processed_path / config.dataset_name
                                  / config.model_name / "acs_concept_train.json")
    config.test_concept_path = (config.processed_path / config.dataset_name
                                 / config.model_name / "acs_concept_test.json")
    config.train_bge_path = (config.processed_path / config.dataset_name
                              / config.model_name / "bge_embed_train.pt")
    config.test_bge_path = (config.processed_path / config.dataset_name
                             / config.model_name / "bge_embed_test.pt")
    config.train_hidden_path = (config.processed_path / config.dataset_name
                                 / config.model_name / "hidden_train.pt")
    config.test_hidden_path = (config.processed_path / config.dataset_name
                                / config.model_name / "hidden_test.pt")

    # 随机种子
    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    # CB-LLM-CN模型参数
    if args.cbl_hidden_dim is not None:
        config.cbl_hidden_dim = args.cbl_hidden_dim
    if args.cbl_dropout is not None:
        config.cbl_dropout = args.cbl_dropout
    if args.sparse_l1_weight is not None:
        config.sparse_l1_weight = args.sparse_l1_weight
    if args.prediction_dropout is not None:
        config.prediction_dropout = args.prediction_dropout
    if args.topk is not None:
        config.topk = args.topk
    if args.use_residual:
        config.use_residual = True

    # MLP参数
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    if args.hidden_features is not None:
        config.hidden_features = args.hidden_features

    # 训练超参数
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_lr is not None:
        config.max_lr = args.max_lr
    if args.patience is not None:
        config.patience = args.patience

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
    """加载概念评分和backbone嵌入

    根据config.backbone选择加载哪种数据组合：
    - bge: ACS概念评分 + BGE嵌入
    - qwen: ACS概念评分 + Qwen hidden state
    - concept_only: ACS概念评分（无backbone）
    - likert_bge: Likert概念评分 + BGE嵌入（混合方案）

    Returns:
        concept_scores: [N, K] 概念评分（ACS或Likert）
        backbone_embeds: [N, D] backbone嵌入 (concept_only模式为None)
        labels: [N] 标签
        contents: [N] 文本内容列表
    """
    if mode == "train":
        acs_concept_path = config.train_concept_path
        bge_path = config.train_bge_path
        hidden_path = config.train_hidden_path
    elif mode == "test":
        acs_concept_path = config.test_concept_path
        bge_path = config.test_bge_path
        hidden_path = config.test_hidden_path
    else:
        raise ValueError(f"mode must be 'train' or 'test', got {mode}")

    # 确定概念数据路径
    if config.backbone == "likert_bge":
        # 使用Likert概念评分（而非ACS）
        concept_path = (Path(config.processed_path) / config.dataset_name
                        / config.model_name / f"concept_{mode}.json")
    else:
        concept_path = acs_concept_path

    concept_path = Path(concept_path) if not isinstance(concept_path, Path) else concept_path

    if not concept_path.exists():
        raise FileNotFoundError(
            f"概念评分文件不存在: {concept_path}\n"
            f"请先生成对应的概念评分文件")

    # 加载概念评分
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    concepts, labels, contents = [], [], []
    for item in raw_concept_data:
        concepts.append(item["concept"])
        labels.append(item["toxic"])
        contents.append(item["content"])

    concept_scores = torch.tensor(concepts, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 加载backbone嵌入
    backbone_embeds = None
    if config.backbone in ("bge", "likert_bge"):
        bge_path = Path(bge_path) if not isinstance(bge_path, Path) else bge_path
        if not bge_path.exists():
            raise FileNotFoundError(
                f"BGE嵌入文件不存在: {bge_path}\n"
                f"请先运行: python scripts/generate_acs_concepts.py --mode {mode} --dataset_name {config.dataset_name}")
        backbone_embeds = torch.load(bge_path, map_location="cpu", weights_only=True)
        print(f"  加载 {mode} BGE嵌入: {backbone_embeds.shape}")
    elif config.backbone == "qwen":
        hidden_path = Path(hidden_path) if not isinstance(hidden_path, Path) else hidden_path
        if not hidden_path.exists():
            raise FileNotFoundError(
                f"Hidden state文件不存在: {hidden_path}\n"
                f"请先运行: python scripts/extract_hidden_states.py --mode {mode}")
        backbone_embeds = torch.load(hidden_path, map_location="cpu", weights_only=True)
        print(f"  加载 {mode} Qwen hidden: {backbone_embeds.shape}")

    # 校验数据一致性
    if backbone_embeds is not None and concept_scores.shape[0] != backbone_embeds.shape[0]:
        raise ValueError(
            f"数据量不一致: Concept {concept_scores.shape[0]} vs Backbone {backbone_embeds.shape[0]}"
        )

    concept_type = "Likert" if config.backbone == "likert_bge" else "ACS"
    print(f"  加载 {mode} 数据: {concept_scores.shape[0]} 条, "
          f"概念类型={concept_type}, 概念维度: {concept_scores.shape[1]}"
          + (f", Backbone维度: {backbone_embeds.shape[1]}" if backbone_embeds is not None else ""))

    return concept_scores, backbone_embeds, labels_tensor, contents


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses, extra_losses=None, extra_losses2=None):
    """绘制训练曲线图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图: Loss
    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title(f'CB-LLM-CN Training Metrics (backbone={config.backbone})')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 下图: Score
    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, val_precisions, color='tab:green', linestyle='--', label='Val Precision')
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


def train_cbllm(config, train_dataset, val_dataset, test_dataset, num_concepts, backbone_dim):
    """训练CB-LLM-CN模型（bge/qwen backbone模式）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = CBLLM_CN(
        num_concepts=num_concepts,
        backbone_dim=backbone_dim,
        cbl_hidden_dim=config.cbl_hidden_dim,
        cbl_dropout=config.cbl_dropout,
        sparse_l1_weight=config.sparse_l1_weight,
        prediction_dropout=config.prediction_dropout,
        topk=config.topk,
        use_residual=config.use_residual,
    ).to(device)

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

    epoch_list, val_loss_history, val_f1_history = [], [], []
    val_precision_history, val_recall_history = [], []
    test_f1_history, test_loss_history = [], []
    cbl_loss_history, cls_loss_history = [], []

    for epoch in range(config.epochs):
        model.train()
        epoch_cbl_loss, epoch_cls_loss, epoch_count = 0.0, 0.0, 0

        for concept_batch, backbone_batch, label_batch in train_loader:
            concept_batch = concept_batch.to(device)
            backbone_batch = backbone_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits, concept_activations, l1_loss = model(backbone_batch, concept_batch)
            loss, loss_dict = model.compute_loss(
                logits, label_batch, concept_activations, concept_batch, l1_loss
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_cbl_loss += loss_dict["L_cbl"]
            epoch_cls_loss += loss_dict["L_cls"]
            epoch_count += 1

        avg_cbl_loss = epoch_cbl_loss / max(epoch_count, 1)
        avg_cls_loss = epoch_cls_loss / max(epoch_count, 1)

        # 验证集评估
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss, val_count = 0.0, 0

        with torch.no_grad():
            for concept_batch, backbone_batch, label_batch in val_loader:
                concept_batch = concept_batch.to(device)
                backbone_batch = backbone_batch.to(device)
                label_batch = label_batch.to(device)
                logits, concept_activations, l1_loss = model(backbone_batch, concept_batch)
                loss, _ = model.compute_loss(
                    logits, label_batch, concept_activations, concept_batch, l1_loss
                )
                total_val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(label_batch.cpu().numpy())
                val_count += 1

        avg_val_loss = total_val_loss / max(val_count, 1)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # 测试集评估
        test_preds, test_labels_list = [], []
        total_test_loss, test_count = 0.0, 0

        with torch.no_grad():
            for concept_batch, backbone_batch, label_batch in test_loader:
                concept_batch = concept_batch.to(device)
                backbone_batch = backbone_batch.to(device)
                label_batch = label_batch.to(device)
                logits, concept_activations, l1_loss = model(backbone_batch, concept_batch)
                loss, _ = model.compute_loss(
                    logits, label_batch, concept_activations, concept_batch, l1_loss
                )
                total_test_loss += loss.item()
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels_list.extend(label_batch.cpu().numpy())
                test_count += 1

        avg_test_loss = total_test_loss / max(test_count, 1)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        val_precision_history.append(val_p)
        val_recall_history.append(val_r)
        test_f1_history.append(test_f1)
        test_loss_history.append(avg_test_loss)
        cbl_loss_history.append(avg_cbl_loss)
        cls_loss_history.append(avg_cls_loss)

        print(f"Epoch {epoch + 1}: "
              f"Val F1={val_f1:.4f} P={val_p:.4f} R={val_r:.4f} | "
              f"Test F1={test_f1:.4f} | "
              f"CBL={avg_cbl_loss:.4f} Cls={avg_cls_loss:.4f}")

        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"  >>> 更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停触发: 连续 {config.patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history,
            cbl_loss_history, cls_loss_history)


def train_mlp(config, train_dataset, val_dataset, test_dataset, in_features):
    """训练MLP模型（concept_only模式，ACS概念评分直接分类）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = MLP(
        in_features=in_features,
        dropout_rate=config.dropout_rate,
        hidden_features=config.hidden_features,
    ).to(device)

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

    epoch_list, val_loss_history, val_f1_history = [], [], []
    val_precision_history, val_recall_history = [], []
    test_f1_history, test_loss_history = [], []

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
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # 测试集评估
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

        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        val_precision_history.append(val_p)
        val_recall_history.append(val_r)
        test_f1_history.append(test_f1)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch + 1}: "
              f"Val F1={val_f1:.4f} P={val_p:.4f} R={val_r:.4f} | "
              f"Test F1={test_f1:.4f}")

        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"  >>> 更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停触发: 连续 {config.patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history, [], [])


def evaluate(config, timestamp):
    """评估指定实验的最佳模型在测试集上的表现"""
    experiment_dir = Path(config.base_path) / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_concept, test_backbone, test_labels, contents = load_data(saved_config, "test")

    backbone = getattr(saved_config, 'backbone', 'qwen')

    if backbone == "concept_only":
        # MLP模式
        test_dataset = TensorDataset(test_concept, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

        model = MLP(
            in_features=test_concept.shape[1],
            dropout_rate=float(getattr(saved_config, 'dropout_rate', 0.5)),
            hidden_features=int(getattr(saved_config, 'hidden_features', 96)),
        )
        model.load_state_dict(
            torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False)
        )
        model.to(device).eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(batch_y.numpy())
    else:
        # CB-LLM-CN模式
        if test_backbone is None:
            raise ValueError(f"backbone={backbone} 需要backbone嵌入，但加载为None")

        test_dataset = TensorDataset(test_concept, test_backbone, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

        model = CBLLM_CN(
            num_concepts=test_concept.shape[1],
            backbone_dim=test_backbone.shape[1],
            cbl_hidden_dim=int(saved_config.cbl_hidden_dim),
            cbl_dropout=float(saved_config.cbl_dropout),
            sparse_l1_weight=float(saved_config.sparse_l1_weight),
            prediction_dropout=float(saved_config.prediction_dropout),
            topk=int(getattr(saved_config, 'topk', 0)),
            use_residual=bool(getattr(saved_config, 'use_residual', False)),
        )
        model.load_state_dict(
            torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False)
        )
        model.to(device).eval()

        all_preds, all_labels = [], []
        all_concept_activations = []
        with torch.no_grad():
            for concept_batch, backbone_batch, label_batch in test_loader:
                backbone_batch = backbone_batch.to(device)
                logits, concept_activations, _ = model(backbone_batch)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(label_batch.numpy())
                all_concept_activations.append(concept_activations.cpu())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 概念稀疏性指标（仅CB-LLM-CN模式）
    avg_active_concepts = 0
    avg_concept_value = 0
    if backbone != "concept_only" and len(all_concept_activations) > 0:
        all_concept = torch.cat(all_concept_activations, dim=0)
        avg_active_concepts = (all_concept > 0.01).float().mean(dim=0).sum().item()
        avg_concept_value = all_concept.mean().item()

    # 输出
    print("\n" + "=" * 30)
    print(f"      CB-LLM-CN 测试集评估结果 (backbone={backbone})")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    if backbone != "concept_only":
        print(f"平均激活概念数: {avg_active_concepts:.1f} / {test_concept.shape[1]}")
        print(f"平均概念激活值: {avg_concept_value:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    # 保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "backbone": backbone,
    }
    if backbone != "concept_only":
        metrics_dict["avg_active_concepts"] = round(avg_active_concepts, 1)
        metrics_dict["avg_concept_value"] = round(avg_concept_value, 4)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"CB-LLM-CN 测试集评估结果 (backbone={backbone})\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        if backbone != "concept_only":
            f.write(f"平均激活概念数: {avg_active_concepts:.1f} / {test_concept.shape[1]}\n")
            f.write(f"平均概念激活值: {avg_concept_value:.4f}\n")
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
        config = update_CBLLMConfig(args)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{config.backbone}"
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存配置
        config_dict = {
            "timestamp": timestamp,
            "branch": "feature/cb-llm-cn",
            "backbone": config.backbone,
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "train_bge_path": str(config.train_bge_path),
            "test_bge_path": str(config.test_bge_path),
            "train_hidden_path": str(config.train_hidden_path),
            "test_hidden_path": str(config.test_hidden_path),
            "processed_path": str(config.processed_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "embedding_model_name": config.embedding_model_name,
            "cbl_hidden_dim": config.cbl_hidden_dim,
            "cbl_dropout": config.cbl_dropout,
            "sparse_l1_weight": config.sparse_l1_weight,
            "prediction_dropout": config.prediction_dropout,
            "topk": config.topk,
            "use_residual": config.use_residual,
            "dropout_rate": config.dropout_rate,
            "hidden_features": config.hidden_features,
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

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")

        # 加载数据
        print("\n>>> 加载数据...")
        train_concept, train_backbone, train_labels, _ = load_data(config, "train")
        test_concept, test_backbone, test_labels, _ = load_data(config, "test")

        num_concepts = train_concept.shape[1]
        print(f"\n>>> 数据概览: 概念K={num_concepts}, backbone={config.backbone}")

        if config.backbone == "concept_only":
            # ========== concept_only模式：ACS概念评分直接分类 ==========
            train_x_np = train_concept.numpy()
            train_y_np = train_labels.numpy()
            test_x = test_concept
            test_y = test_labels

            train_x_np, val_x_np, train_y_np, val_y_np = train_test_split(
                train_x_np, train_y_np,
                test_size=0.1, stratify=train_y_np, random_state=config.seed
            )

            train_dataset = TensorDataset(
                torch.tensor(train_x_np, dtype=torch.float32),
                torch.tensor(train_y_np, dtype=torch.long),
            )
            val_dataset = TensorDataset(
                torch.tensor(val_x_np, dtype=torch.float32),
                torch.tensor(val_y_np, dtype=torch.long),
            )
            test_dataset = TensorDataset(test_x, test_y)

            print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, "
                  f"测试集: {len(test_dataset)}")
            print("\n>>> 开始训练 (concept_only / MLP)...")
            metrics = train_mlp(config, train_dataset, val_dataset, test_dataset, num_concepts)

        else:
            # ========== bge/qwen模式：backbone → CBL → 稀疏预测 ==========
            backbone_dim = train_backbone.shape[1]
            print(f"  Backbone维度: {backbone_dim}")

            train_concept_np = train_concept.numpy()
            train_backbone_np = train_backbone.numpy()
            train_labels_np = train_labels.numpy()

            (train_concept_np, val_concept_np,
             train_backbone_np, val_backbone_np,
             train_labels_np, val_labels_np) = train_test_split(
                train_concept_np, train_backbone_np, train_labels_np,
                test_size=0.1, stratify=train_labels_np, random_state=config.seed
            )

            train_dataset = TensorDataset(
                torch.tensor(train_concept_np, dtype=torch.float32),
                torch.tensor(train_backbone_np, dtype=torch.float32),
                torch.tensor(train_labels_np, dtype=torch.long),
            )
            val_dataset = TensorDataset(
                torch.tensor(val_concept_np, dtype=torch.float32),
                torch.tensor(val_backbone_np, dtype=torch.float32),
                torch.tensor(val_labels_np, dtype=torch.long),
            )
            test_dataset = TensorDataset(test_concept, test_backbone, test_labels)

            print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, "
                  f"测试集: {len(test_dataset)}")
            print(f"\n>>> 开始训练 (backbone={config.backbone})...")
            metrics = train_cbllm(config, train_dataset, val_dataset, test_dataset,
                                   num_concepts, backbone_dim)

        # 绘制训练曲线图
        plot_metrics(config, *metrics[:7])

        # all模式下执行测试
        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = CBLLMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
