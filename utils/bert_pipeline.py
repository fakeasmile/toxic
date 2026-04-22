"""BERT Baseline 训练与测试流水线。

整合训练和测试功能，实现训练完成后自动测试的流水线。
支持命令行参数配置，确保训练-测试配置一致性。

训练流程:
    1. 加载BERT_config.py默认配置，用命令行参数覆盖（优先级: 命令行 > BERT_config.py）
    2. 生成时间戳实验目录，保存完整配置到config.json
    3. 加载原始JSON数据，用BertTokenizer编码，按9:1分层抽样划分训练/验证集
    4. 训练: AdamW + linear warmup/decay + 梯度裁剪，tqdm显示进度
    5. 每个epoch记录验证集(Loss, F1, Precision, Recall)和测试集(Loss, F1，仅观察不参与模型筛选)
    6. 基于验证集F1进行最佳模型选择和早停
    7. 训练结束后绘制双子图曲线(上图: Loss, 下图: Score)
    8. all模式下自动评估最佳模型在测试集上的表现

测试流程:
    从实验目录的config.json恢复训练配置，恢复随机种子设置，
    加载最佳模型权重，在测试集上计算Precision/Recall/F1并保存结果。

使用示例:
    # 1. 训练+测试
    python utils/bert_pipeline.py --mode all

    # 2. 仅训练模式
    python utils/bert_pipeline.py --mode train

    # 3. 仅测试模式 (必须指定实验时间戳)
    python utils/bert_pipeline.py --mode test --timestamp 20260421-120000

    # 4. 自定义完整超参数
    python utils/bert_pipeline.py --mode all --dataset_name COLD --seed 1 --use_deterministic --batch_size 16 --epochs 5 --max_seq_length 128 --patience 2

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳 (如: 20260421-120000)

    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)

    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (确保实验可复现，默认: False)

    训练超参数:
        --batch_size        批次大小 (默认: 16)
        --epochs            训练轮数 (默认: 5)
        --max_seq_length    最大序列长度 (默认: 128)
        --patience          早停耐心值 (验证集F1连续patience个epoch未提升则停止训练, 默认: 2)

参数优先级:
    - 训练模式: 命令行参数 > BERT_config.py（命令行参数覆盖BERT_config参数）
    - 测试模式: 强制使用实验目录的 config.json (忽略命令行超参数)

输出文件:
    实验目录结构 (experiments_bert/<timestamp>/):
        ├── config.json              # 实验配置快照
        ├── best_model.pth           # 最佳模型权重
        ├── metrics.png              # 训练曲线图 (上图: Loss, 下图: Score)
        └── test_results/            # 测试结果目录 (仅 all/test 模式)
            ├── metrics.json         # 测试集评估指标
            └── classification_report.txt  # 详细分类报告
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.BERT_config import BERTConfig
from models.bert import BERTBaseline

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


class TextDataset(Dataset):
    """文本分类数据集，封装tokenizer编码后的输入和标签。"""

    def __init__(self, encodings, labels):
        """
        :param encodings: tokenizer编码结果（包含input_ids, attention_mask, token_type_ids）
        :param labels: 标签张量
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BERT Baseline 训练与测试流水线")

    # 运行模式
    parser.add_argument('--mode', choices=['all', 'train', 'test'], default='all',
                        help='运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)')
    parser.add_argument('--timestamp', default=None,
                        help='测试模式时的实验时间戳 (如: 20260421-120000)')

    # 数据集配置
    parser.add_argument('--dataset_name', default=None,
                        help='数据集名称 (TOXICN/COLD)')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False,
                        help='启用确定性模式')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_seq_length', type=int, default=None, help='最大序列长度')
    parser.add_argument('--patience', type=int, default=None,
                        help='早停耐心值 (验证集F1连续patience个epoch未提升则停止训练)')

    return parser.parse_args()


def update_BERTConfig(args):
    """基于BERT_config.py默认值，根据命令行参数更新配置对象。

    优先级: 命令行参数 > BERTConfig默认值

    :param args: 命令行参数
    :return: 更新后的 BERTConfig 对象
    """
    config = BERTConfig()

    if args.dataset_name is not None:
        config.dataset_name = args.dataset_name
    config.train_path = config.base_path / "data" / "raw" / config.dataset_name / "train.json"
    config.test_path = config.base_path / "data" / "raw" / config.dataset_name / "test.json"

    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length
    if args.patience is not None:
        config.patience = args.patience

    return config


def load_raw_data(config, mode):
    """加载原始JSON数据，提取文本内容和标签。

    :param config: BERTConfig 配置对象
    :param mode: "train" 或 "test"
    :return: (texts, labels) 文本列表和标签列表
    """
    path = config.train_path if mode == "train" else config.test_path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["content"] for item in data if isinstance(item["content"], str)]
    labels = [item["toxic"] for item in data if isinstance(item["content"], str)]
    return texts, labels


def tokenize_data(texts, labels, tokenizer, max_seq_length):
    """对文本进行tokenize编码，返回TextDataset。

    :param texts: 文本列表
    :param labels: 标签列表
    :param tokenizer: BertTokenizer 实例
    :param max_seq_length: 最大序列长度
    :return: TextDataset 实例
    """
    encodings = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_seq_length, return_tensors='pt'
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return TextDataset(encodings, labels_tensor)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    """绘制训练曲线图（上下双子图）。

    上图: Loss曲线（Val Loss + Test Loss）
    下图: Score曲线（Val F1, Val Precision, Val Recall, Test F1）

    :param config: BERTConfig 配置对象
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
    ax1.set_title('BERT Baseline Training Metrics')
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
    """训练BERT Baseline模型。

    使用AdamW优化器 + linear warmup/decay学习率调度 + 梯度裁剪。
    基于验证集F1进行早停和最佳模型选择，同时观察测试集F1但不参与模型筛选。

    :param config: BERTConfig 配置对象
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

    # 初始化模型
    model = BERTBaseline(
        bert_path=str(config.bert_path),
        dropout_rate=config.dropout_rate
    ).to(device)

    # 损失函数、优化器、学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

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
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
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

    :param config: BERTConfig 配置对象（用于获取base_path）
    :param timestamp: 实验时间戳
    """
    experiment_dir = config.base_path / "experiments_bert" / timestamp
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
    test_texts, test_labels = load_raw_data(saved_config, "test")
    test_dataset = tokenize_data(test_texts, test_labels, tokenizer, int(saved_config.max_seq_length))
    test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

    # 加载最佳模型
    model = BERTBaseline(bert_path=str(saved_config.bert_path), dropout_rate=saved_config.dropout_rate)
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
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
    print("  BERT Baseline 测试集评估结果")
    print("=" * 30)
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 (Macro):        {f1:.4f}")
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
        f.write(f"BERT Baseline 测试集评估结果\n{'=' * 30}\n")
        f.write(f"Precision (Macro): {precision:.4f}\nRecall (Macro):    {recall:.4f}\nF1 (Macro):        {f1:.4f}\n")
        f.write(f"{'-' * 30}\n详细分类报告:\n{report}\n{'=' * 30}\n")


def main():
    """
    参数加载逻辑：训练模式基于BERT_config.py，使用命令行参数更新配置，并保存到config.json中
    测试模式从实验目录的config.json中加载参数配置
    """
    args = parse_args()

    if args.mode in ['all', 'train']:
        # 获取完整参数配置
        config = update_BERTConfig(args)

        # 生成时间戳并创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.base_path / "experiments_bert" / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存完整配置到config.json（显式赋值）
        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "bert_path": str(config.bert_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "max_seq_length": config.max_seq_length,
            "dropout_rate": config.dropout_rate,
            "patience": config.patience,
        }
        with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
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
        train_texts, train_labels = load_raw_data(config, "train")
        test_texts, test_labels = load_raw_data(config, "test")

        # 从训练集中按9:1比例划分验证集（分层抽样）
        train_texts_split, val_texts_split, train_labels_split, val_labels_split = train_test_split(
            train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=config.seed
        )

        train_dataset = tokenize_data(train_texts_split, train_labels_split, tokenizer, config.max_seq_length)
        val_dataset = tokenize_data(val_texts_split, val_labels_split, tokenizer, config.max_seq_length)
        test_dataset = tokenize_data(test_texts, test_labels, tokenizer, config.max_seq_length)

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
        config = BERTConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
