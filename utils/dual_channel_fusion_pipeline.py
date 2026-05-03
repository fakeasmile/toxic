"""双通道融合模型（BERT + 形容词概念向量）训练与测试。

整合训练和测试功能,实现训练完成后自动测试的流水线。
支持命令行参数配置,确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试
    python utils/dual_channel_fusion_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert

    # 2. 仅训练模式
    python utils/dual_channel_fusion_pipeline.py --mode train --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert

    # 3. 仅测试模式 (必须指定实验时间戳)
    python utils/dual_channel_fusion_pipeline.py --mode test --timestamp 20260415-085433 --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert

    # 4. 自定义数据集和超参数（完整命令）
    python utils/dual_channel_fusion_pipeline.py --mode all
        --dataset_name COLD
        --model_name Qwen2.5-3B-Instruct
        --template binary
        --batch_size 16
        --epochs 5
        --max_seq_length 128
        --dropout_rate 0.3
        --patience 2
        --use_deterministic
        --seed 42
    一般情况下：
    python utils/dual_channel_fusion_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert --epochs 5 --use_deterministic

    # 5. 启用确定性模式 (确保实验可复现)
    python utils/dual_channel_fusion_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert --use_deterministic --seed 42

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳 (如: 20260415-085433)

    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)
        --model_name        LLM模型名称 (默认: Qwen2.5-3B-Instruct)
        --template          提示词模板类型 (binary/likert, 默认: likert)

    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (确保实验可复现，默认：False)

    训练超参数:
        --batch_size        批次大小 (默认: 16)
        --epochs            训练轮数 (默认: 5)
        --max_seq_length    最大序列长度 (默认: 128)
        --patience          早停耐心值 (验证集F1连续patience个epoch未提升则停止, 默认: 2)

    融合模型结构参数:
        --dropout_rate      Dropout比率 (默认: 0.3)

参数优先级:
    - 训练模式: 命令行参数 > DualChannelFusionConfig默认值（命令行参数覆盖DualChannelFusionConfig参数）
    - 测试模式: 强制使用实验目录的 config.json (忽略命令行超参数)

输出文件:
    实验目录结构 (experiments_dual_channel_fusion/<timestamp>/):
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
    3. dataset_name、model_name、template 为训练模式必传参数，用于定位概念向量路径
"""

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.dual_channel_fusion_config import DualChannelFusionConfig
from models.bert import DualChannelFusion

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


class FusionDataset(Dataset):
    """融合模型数据集，封装tokenizer编码、概念向量和标签。"""

    def __init__(self, encodings, concept_vectors, labels):
        """
        :param encodings: tokenizer编码结果（包含input_ids, attention_mask, token_type_ids）
        :param concept_vectors: 概念向量张量
        :param labels: 标签张量
        """
        self.encodings = encodings
        self.concept_vectors = concept_vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['concept_vector'] = self.concept_vectors[idx]
        item['labels'] = self.labels[idx]
        return item


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="双通道融合模型 训练与测试统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流水线 (训练+测试)
  python utils/dual_channel_fusion_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert

  # 仅训练
  python dual_channel_fusion_pipeline.py --mode train --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert

  # 仅测试
  python dual_channel_fusion_pipeline.py --mode test --timestamp 20260415-085433 --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct --template likert

  # 自定义超参数
  python dual_channel_fusion_pipeline.py --mode all --dataset_name COLD --model_name Qwen2.5-3B-Instruct --template binary --epochs 10 --dropout_rate 0.3
        """
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

    # 数据集配置（训练模式必传，用于定位概念向量路径）
    parser.add_argument('--dataset_name', type=str, default=None, help='数据集名称 (TOXICN/COLD)，训练模式必传')
    parser.add_argument('--model_name', type=str, default=None, help='LLM模型名称，训练模式必传')
    parser.add_argument('--template', type=str, choices=['binary', 'likert'], default=None,
                        help='提示词模板类型：binary=二元判断, likert=Likert程度量化，训练模式必传')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False, help='启用确定性模式')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_seq_length', type=int, default=None, help='最大序列长度')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值 (验证集F1连续patience个epoch未提升则停止)')

    # 融合模型结构参数
    parser.add_argument('--dropout_rate', type=float, default=None, help='Dropout比率')

    return parser.parse_args()


def update_DualChannelFusionConfig(args):
    """基于DualChannelFusionConfig默认值，根据命令行参数更新配置对象。

    优先级: 命令行参数 > DualChannelFusionConfig默认值

    :param args: 命令行参数
    :return: 更新后的 DualChannelFusionConfig 对象
    """
    config = DualChannelFusionConfig()

    # 校验训练模式必传参数
    if args.dataset_name is None or args.model_name is None or args.template is None:
        raise ValueError("训练模式必须指定 --dataset_name, --model_name, --template")

    # 数据集与模板配置，动态生成依赖路径
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    config.template = args.template
    config.train_path = config.raw_data_path / config.dataset_name / "train.json"
    config.test_path = config.raw_data_path / config.dataset_name / "test.json"
    config.train_concept_path = (config.processed_path / config.dataset_name
                                 / config.model_name / config.template / "concept_train.json")
    config.test_concept_path = (config.processed_path / config.dataset_name
                                / config.model_name / config.template / "concept_test.json")

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
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length
    if args.patience is not None:
        config.patience = args.patience

    # 融合模型结构参数
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate

    return config


def load_data(config, mode):
    """加载原始文本数据和概念向量，按content对齐。

    :param config: DualChannelFusionConfig 配置对象
    :param mode: "train" 或 "test"
    :return: (texts, labels, concepts) 文本列表、标签列表、概念向量列表
    """
    if mode == "train":
        raw_path = config.train_path
        concept_path = config.train_concept_path
    elif mode == "test":
        raw_path = config.test_path
        concept_path = config.test_concept_path
    else:
        raise ValueError("in load_data, mode must be 'train' or 'test'")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    with open(concept_path, "r", encoding="utf-8") as f:
        concept_data = json.load(f)

    # 构建content→概念向量的映射，用于对齐
    concept_map = {}
    for item in concept_data:
        concept_map[item["content"]] = item["concept"]

    texts, labels, concepts = [], [], []
    for item in raw_data:
        content = item["content"]
        if content in concept_map:
            texts.append(content)
            labels.append(item["toxic"])
            concepts.append(concept_map[content])

    return texts, labels, concepts


def tokenize_and_build_dataset(texts, labels, concepts, tokenizer, max_seq_length):
    """对文本进行tokenize编码，与概念向量一起构建FusionDataset。

    :param texts: 文本列表
    :param labels: 标签列表
    :param concepts: 概念向量列表
    :param tokenizer: BertTokenizer 实例
    :param max_seq_length: 最大序列长度
    :return: FusionDataset 实例
    """
    encodings = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_seq_length, return_tensors='pt'
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    concept_tensor = torch.tensor(concepts, dtype=torch.float32)
    return FusionDataset(encodings, concept_tensor, labels_tensor)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    """绘制训练曲线图（上下双子图）。

    上图: Loss曲线（Val Loss + Test Loss）
    下图: Score曲线（Val F1, Val Precision, Val Recall, Test F1）

    :param config: DualChannelFusionConfig 配置对象
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
    ax1.set_title('Fusion Training Metrics')
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
    """训练双通道融合模型。

    使用AdamW优化器 + linear warmup/decay学习率调度 + 梯度裁剪。
    基于验证集F1进行早停和最佳模型选择，同时观察测试集F1但不参与模型筛选。

    :param config: DualChannelFusionConfig 配置对象
    :param train_dataset: 训练集
    :param val_dataset: 验证集
    :param test_dataset: 测试集（仅观察F1变化，不参与模型筛选）
    :return: (epochs, val_losses, val_f1_scores, val_precisions, val_recalls, test_f1_scores, test_losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型（concept_dim从数据自动推断）
    concept_dim = train_dataset.concept_vectors.shape[1]
    model = DualChannelFusion(
        bert_path=str(config.bert_path),
        concept_dim=concept_dim,
        dropout_rate=config.dropout_rate
    ).to(device)

    # 损失函数
    if config.use_focal_loss:
        from utils.focal_loss import FocalLoss
        criterion = FocalLoss(gamma=config.focal_gamma)
        print(f">>> 使用Focal Loss (gamma={config.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        print(f">>> 使用CrossEntropy + 标签平滑 (smoothing={config.label_smoothing})")

    # 分层学习率优化器
    # BERT层使用较低学习率，投影层和分类头使用较高学习率
    bert_params = list(model.bert.named_parameters())
    projection_params = (
        list(model.concept_proj.named_parameters()) +
        list(model.layer_gates.named_parameters()) +
        list(model.layer_norm.named_parameters())
    )
    classifier_params = list(model.classifier.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_params], 'lr': config.bert_learning_rate, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in projection_params], 'lr': config.projection_learning_rate, 'weight_decay': config.weight_decay},
        {'params': [p for n, p in classifier_params], 'lr': config.projection_learning_rate, 'weight_decay': config.weight_decay},
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)
    print(f">>> 分层学习率: BERT={config.bert_learning_rate}, Projection/Classifier={config.projection_learning_rate}")

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
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

    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        # ========== 训练阶段 ==========
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            concept_vector = batch['concept_vector'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, concept_vector, token_type_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            scheduler.step()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False):
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['concept_vector'].to(device),
                    batch['token_type_ids'].to(device)
                )
                v_loss = criterion(outputs, batch['labels'].to(device))
                total_val_loss += v_loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels_list.extend(batch['labels'].numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # ========== 测试集评估（仅观察，不参与模型筛选）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} [Test]", leave=False):
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['concept_vector'].to(device),
                    batch['token_type_ids'].to(device)
                )
                t_loss = criterion(outputs, batch['labels'].to(device))
                total_test_loss += t_loss.item()
                test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_labels_list.extend(batch['labels'].numpy())

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

    :param config: DualChannelFusionConfig 配置对象（用于获取base_path）
    :param timestamp: 实验时间戳
    """
    experiment_dir = config.base_path / "experiments_dual_channel_fusion" / timestamp
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
    tokenizer = BertTokenizer.from_pretrained(str(saved_config.bert_path))

    # 加载测试数据
    test_texts, test_labels, test_concepts = load_data(saved_config, "test")
    test_dataset = tokenize_and_build_dataset(
        test_texts, test_labels, test_concepts, tokenizer, int(saved_config.max_seq_length)
    )
    test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

    # 加载最佳模型
    concept_dim = test_dataset.concept_vectors.shape[1]
    model = DualChannelFusion(
        bert_path=str(saved_config.bert_path),
        concept_dim=concept_dim,
        dropout_rate=saved_config.dropout_rate
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['concept_vector'].to(device),
                batch['token_type_ids'].to(device)
            )
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 输出到控制台
    print("\n" + "=" * 30)
    print("      Fusion 测试集评估结果")
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
        f.write("Fusion 测试集评估结果\n")
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
            "content": test_texts[i],
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
    参数加载逻辑：训练模式基于dual_channel_fusion_config.py，使用命令行参数更新配置，并保存到config.json中
    测试模式从实验目录的config.json中加载参数配置
    """
    args = parse_args()

    if args.mode in ['all', 'train']:
        # 获取完整参数配置
        config = update_DualChannelFusionConfig(args)

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
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "raw_data_path": str(config.raw_data_path),
            "processed_path": str(config.processed_path),
            "bert_path": str(config.bert_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "bert_learning_rate": config.bert_learning_rate,
            "projection_learning_rate": config.projection_learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "max_seq_length": config.max_seq_length,
            "dropout_rate": config.dropout_rate,
            "patience": config.patience,
            "label_smoothing": config.label_smoothing,
            "use_focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma
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

        # 加载tokenizer和数据
        tokenizer = BertTokenizer.from_pretrained(str(config.bert_path))
        train_texts, train_labels, train_concepts = load_data(config, "train")
        test_texts, test_labels, test_concepts = load_data(config, "test")

        # 从训练集中按9:1比例划分验证集（分层抽样）
        train_texts_split, val_texts_split, train_labels_split, val_labels_split, train_concepts_split, val_concepts_split = train_test_split(
            train_texts, train_labels, train_concepts,
            test_size=0.1, stratify=train_labels, random_state=config.seed
        )

        train_dataset = tokenize_and_build_dataset(
            train_texts_split, train_labels_split, train_concepts_split, tokenizer, config.max_seq_length
        )
        val_dataset = tokenize_and_build_dataset(
            val_texts_split, val_labels_split, val_concepts_split, tokenizer, config.max_seq_length
        )
        test_dataset = tokenize_and_build_dataset(
            test_texts, test_labels, test_concepts, tokenizer, config.max_seq_length
        )

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
        config = DualChannelFusionConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
