"""DCBM-CN 训练与测试流水线

整合训练和测试功能，实现训练完成后自动测试。
支持命令行参数配置，确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试 (TOXICN, 197概念)
    python utils/dcbm_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-GPTQ-Int8

    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/dcbm_pipeline.py --mode test --timestamp 20260607-120000

    # 3. 使用自定义词典
    python utils/dcbm_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-GPTQ-Int8 --adjective_path data/raw/adjective/implicit_toxic_concepts.csv
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig
from models.dcbm_cn import DCBM_CN

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']

# 话题标签映射
TOPIC_MAP = {
    "race": 0,
    "gender": 1,
    "region": 2,
    "lgbt": 3,
    "other": 4,
}
NUM_TOPICS = 5


class ToxicDataset(Dataset):
    """有害言论数据集

    整合原始文本（用于RoBERTa编码）和显式概念向量（用于分类器输入）。
    """

    def __init__(self, texts, input_ids, attention_masks, explicit_concepts, labels, topics):
        self.texts = texts
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.explicit_concepts = explicit_concepts
        self.labels = labels
        self.topics = topics

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'explicit_concept': self.explicit_concepts[idx],
            'label': self.labels[idx],
            'topic': self.topics[idx],
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DCBM-CN 训练与测试流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'],
                        default='all', help='运行模式: all/train/test')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='测试模式时的实验时间戳')

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='TOXICN', help='数据集名称')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct-GPTQ-Int8',
                        help='概念向量目录名（对应data/processed/{dataset_name}/{model_name}/）')
    parser.add_argument('--adjective_path', type=str, default=None,
                        help='自定义形容词词典路径（CSV），不指定则使用默认路径')

    # RoBERTa配置
    parser.add_argument('--roberta_path', type=str, default=None,
                        help='RoBERTa模型路径，不指定则自动在models/下查找')

    # 模型结构参数
    parser.add_argument('--latent_dim', type=int, default=32, help='隐式概念维度')
    parser.add_argument('--hidden_features', type=int, default=128, help='分类器隐藏层维度')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout比率')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='峰值学习率')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')

    # 损失权重
    parser.add_argument('--alpha_ib', type=float, default=1.0, help='IB损失权重')
    parser.add_argument('--beta_adv', type=float, default=1.0, help='对抗损失权重')
    parser.add_argument('--gamma_sparse', type=float, default=0.01, help='L1稀疏约束系数')

    # 退火参数
    parser.add_argument('--beta_ib_target', type=float, default=4.0, help='IB退火目标值')
    parser.add_argument('--warmup_ratio', type=float, default=0.2, help='退火warmup比例')

    return parser.parse_args()


def find_roberta_path(config):
    """在models/目录下查找RoBERTa模型路径"""
    models_dir = config.models_path
    # 优先级: chinese-roberta-wwm-ext > bert-base-chinese
    candidates = ["chinese-roberta-wwm-ext", "bert-base-chinese"]
    for name in candidates:
        path = models_dir / name
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"在 {models_dir} 下未找到RoBERTa模型，候选: {candidates}")


def extract_topic_label(sample):
    """从数据样本中提取话题标签"""
    if "topic" not in sample:
        return NUM_TOPICS - 1  # 默认归为"other"
    topic = sample["topic"]
    if isinstance(topic, str):
        return TOPIC_MAP.get(topic, NUM_TOPICS - 1)
    elif isinstance(topic, list):
        # multi-hot: 取第一个非零位置
        for i, v in enumerate(topic):
            if v == 1:
                return i
        return NUM_TOPICS - 1
    return NUM_TOPICS - 1


def load_data(config, mode, tokenizer, max_length=128):
    """加载原始文本+显式概念向量+标签+话题标签

    Returns:
        ToxicDataset
    """
    # 加载概念向量文件
    concept_path = config.processed_path / config.dataset_name / config.model_name / f"concept_{mode}.json"
    if not concept_path.exists():
        raise FileNotFoundError(f"概念向量文件不存在: {concept_path}")

    with open(concept_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts = []
    explicit_concepts = []
    labels = []
    topics = []

    for item in raw_data:
        texts.append(item["content"])
        explicit_concepts.append(item["concept"])
        labels.append(item["toxic"])
        topics.append(extract_topic_label(item))

    # Tokenize
    encodings = tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )

    return ToxicDataset(
        texts=texts,
        input_ids=encodings["input_ids"],
        attention_masks=encodings["attention_mask"],
        explicit_concepts=torch.tensor(explicit_concepts, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        topics=torch.tensor(topics, dtype=torch.long),
    )


def get_annealing_value(epoch, total_epochs, target, warmup_ratio):
    """计算退火值（线性从0增加到target）"""
    warmup_epochs = int(total_epochs * warmup_ratio)
    if epoch < warmup_epochs:
        return target * (epoch + 1) / warmup_epochs
    return target


def train(config_dict, train_dataset, val_dataset, test_dataset, device):
    """训练DCBM-CN模型

    Args:
        config_dict: 配置字典
        train_dataset, val_dataset, test_dataset: 数据集
        device: 计算设备
    Returns:
        训练指标历史
    """
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config_dict["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config_dict["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config_dict["batch_size"], shuffle=False)

    # 获取显式概念维度
    explicit_dim = train_dataset.explicit_concepts.shape[1]

    # 初始化模型
    model = DCBM_CN(
        roberta_path=config_dict["roberta_path"],
        explicit_dim=explicit_dim,
        latent_dim=config_dict["latent_dim"],
        num_topics=NUM_TOPICS,
        hidden_features=config_dict["hidden_features"],
        dropout_rate=config_dict["dropout_rate"],
    ).to(device)

    # 优化器（仅训练非RoBERTa参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config_dict["max_lr"] / 25.0)

    total_steps = len(train_loader) * config_dict["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config_dict["max_lr"],
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # 损失函数
    cls_criterion = nn.CrossEntropyLoss()
    topic_criterion = nn.CrossEntropyLoss()

    # 训练状态
    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    # 指标记录
    history = {
        'epochs': [], 'val_loss': [], 'val_f1': [], 'val_p': [], 'val_r': [],
        'test_f1': [], 'test_loss': [],
        'kl_loss': [], 'adv_loss': [], 'sparse_loss': [],
        'beta_ib': [], 'lambda_adv': [],
    }

    for epoch in range(config_dict["epochs"]):
        # 退火值
        beta_ib = get_annealing_value(epoch, config_dict["epochs"],
                                       config_dict["beta_ib_target"], config_dict["warmup_ratio"])
        lambda_adv = get_annealing_value(epoch, config_dict["epochs"],
                                          1.0, config_dict["warmup_ratio"])

        # ========== 训练阶段 ==========
        model.train()
        epoch_cls_loss = 0.0
        epoch_ib_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_sparse_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            explicit = batch['explicit_concept'].to(device)
            labels = batch['label'].to(device)
            topics = batch['topic'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, explicit,
                           lambda_adv=lambda_adv, use_mean=False)

            # 分类损失
            l_cls = cls_criterion(outputs['logits'], labels)
            # IB损失
            l_ib = outputs['kl_loss']
            # 对抗损失
            l_adv = topic_criterion(outputs['topic_pred'], topics)
            # 稀疏损失
            l_sparse = outputs['l1_penalty']

            # 总损失
            loss = (l_cls
                    + config_dict["alpha_ib"] * beta_ib * l_ib
                    + config_dict["beta_adv"] * l_adv
                    + config_dict["gamma_sparse"] * l_sparse)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_cls_loss += l_cls.item()
            epoch_ib_loss += l_ib.item()
            epoch_adv_loss += l_adv.item()
            epoch_sparse_loss += l_sparse.item()

        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                explicit = batch['explicit_concept'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, explicit,
                               lambda_adv=lambda_adv, use_mean=True)

                v_loss = cls_criterion(outputs['logits'], labels)
                total_val_loss += v_loss.item()
                val_preds.extend(torch.argmax(outputs['logits'], dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # ========== 测试集评估（仅观察）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                explicit = batch['explicit_concept'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, explicit,
                               lambda_adv=lambda_adv, use_mean=True)

                t_loss = cls_criterion(outputs['logits'], labels)
                total_test_loss += t_loss.item()
                test_preds.extend(torch.argmax(outputs['logits'], dim=1).cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        # 记录指标
        history['epochs'].append(epoch + 1)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_p'].append(val_p)
        history['val_r'].append(val_r)
        history['test_f1'].append(test_f1)
        history['test_loss'].append(avg_test_loss)
        history['kl_loss'].append(epoch_ib_loss / len(train_loader))
        history['adv_loss'].append(epoch_adv_loss / len(train_loader))
        history['sparse_loss'].append(epoch_sparse_loss / len(train_loader))
        history['beta_ib'].append(beta_ib)
        history['lambda_adv'].append(lambda_adv)

        print(f"Epoch {epoch + 1}: "
              f"Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}, "
              f"Test F1={test_f1:.4f}, "
              f"β_IB={beta_ib:.2f}, λ_adv={lambda_adv:.2f}")

        # ========== 最佳模型选择与早停 ==========
        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> 发现更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config_dict["patience"]:
            print(f">>> 早停触发: 验证集F1已连续 {config_dict['patience']} 个epoch未提升")
            break

    # 保存最佳模型
    if best_state_dict is not None:
        torch.save(best_state_dict, Path(config_dict["experiment_path"]) / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return history, model, explicit_dim


def plot_metrics(experiment_path, history):
    """绘制训练曲线图（三子图）"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    epochs = history['epochs']

    # 上图: Loss
    ax1.plot(epochs, history['val_loss'], color='tab:red', label='Val Loss')
    ax1.plot(epochs, history['test_loss'], color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('DCBM-CN Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 中图: Score
    ax2.plot(epochs, history['val_f1'], color='tab:blue', label='Val F1')
    ax2.plot(epochs, history['test_f1'], color='tab:red', linestyle='-.', label='Test F1')
    ax2.plot(epochs, history['val_p'], color='tab:green', linestyle='--', label='Val Precision')
    ax2.plot(epochs, history['val_r'], color='tab:orange', linestyle=':', label='Val Recall')
    ax2.set_ylabel('Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 下图: 辅助损失和退火参数
    ax3.plot(epochs, history['kl_loss'], color='tab:purple', label='KL Loss')
    ax3.plot(epochs, history['adv_loss'], color='tab:brown', label='Adv Loss')
    ax3.plot(epochs, history['sparse_loss'], color='tab:olive', label='Sparse Loss')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, history['beta_ib'], color='tab:cyan', linestyle='--', label='β_IB')
    ax3_twin.plot(epochs, history['lambda_adv'], color='tab:pink', linestyle='--', label='λ_adv')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3_twin.set_ylabel('Annealing')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = Path(experiment_path) / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def evaluate(config_dict, timestamp, device):
    """评估指定实验的最佳模型在测试集上的表现"""
    experiment_dir = Path(config_dict["base_path"]) / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    # 从实验目录加载训练时保存的配置
    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    # 加载tokenizer和数据
    tokenizer = AutoTokenizer.from_pretrained(saved_config["roberta_path"])

    # 创建临时配置对象用于load_data
    config = MLPConfig()
    config.dataset_name = saved_config["dataset_name"]
    config.model_name = saved_config["model_name"]

    test_dataset = load_data(config, "test", tokenizer)

    test_loader = DataLoader(test_dataset, batch_size=saved_config["batch_size"], shuffle=False)

    # 加载最佳模型
    explicit_dim = test_dataset.explicit_concepts.shape[1]
    model = DCBM_CN(
        roberta_path=saved_config["roberta_path"],
        explicit_dim=explicit_dim,
        latent_dim=saved_config["latent_dim"],
        num_topics=NUM_TOPICS,
        hidden_features=saved_config["hidden_features"],
        dropout_rate=saved_config["dropout_rate"],
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth",
                                      map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            explicit = batch['explicit_concept'].to(device)

            outputs = model(input_ids, attention_mask, explicit,
                           lambda_adv=1.0, use_mean=True)
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 输出到控制台
    print("\n" + "=" * 30)
    print("      DCBM-CN 测试集评估结果")
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
        f.write("DCBM-CN 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
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
            "content": test_dataset.texts[i],
            "true_label": int(all_labels[i]),
            "true_label_name": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_label_name": label_names[int(all_preds[i])],
            "correct": bool(all_preds[i] == all_labels[i]),
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    config = MLPConfig()

    # RoBERTa路径
    roberta_path = args.roberta_path or find_roberta_path(config)

    if args.mode in ['all', 'train']:
        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(roberta_path)

        # 加载数据
        config.dataset_name = args.dataset_name
        config.model_name = args.model_name

        train_dataset = load_data(config, "train", tokenizer)
        test_dataset = load_data(config, "test", tokenizer)

        # 从训练集中按9:1比例划分验证集（分层抽样）
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=0.1,
            stratify=train_dataset.labels.numpy(),
            random_state=args.seed,
        )

        val_dataset = ToxicDataset(
            texts=[train_dataset.texts[i] for i in val_indices],
            input_ids=train_dataset.input_ids[val_indices],
            attention_masks=train_dataset.attention_masks[val_indices],
            explicit_concepts=train_dataset.explicit_concepts[val_indices],
            labels=train_dataset.labels[val_indices],
            topics=train_dataset.topics[val_indices],
        )
        train_dataset = ToxicDataset(
            texts=[train_dataset.texts[i] for i in train_indices],
            input_ids=train_dataset.input_ids[train_indices],
            attention_masks=train_dataset.attention_masks[train_indices],
            explicit_concepts=train_dataset.explicit_concepts[train_indices],
            labels=train_dataset.labels[train_indices],
            topics=train_dataset.topics[train_indices],
        )

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
        print(f">>> 显式概念维度: {train_dataset.explicit_concepts.shape[1]}")

        # 生成时间戳并创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # 构建配置字典
        config_dict = {
            "base_path": str(config.base_path),
            "timestamp": timestamp,
            "experiment_path": str(experiment_dir),
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "roberta_path": roberta_path,
            "latent_dim": args.latent_dim,
            "hidden_features": args.hidden_features,
            "dropout_rate": args.dropout_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_lr": args.max_lr,
            "patience": args.patience,
            "seed": args.seed,
            "alpha_ib": args.alpha_ib,
            "beta_adv": args.beta_adv,
            "gamma_sparse": args.gamma_sparse,
            "beta_ib_target": args.beta_ib_target,
            "warmup_ratio": args.warmup_ratio,
        }

        # 保存配置
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        # 训练
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> 正在使用设备: {device}")

        history, model, explicit_dim = train(config_dict, train_dataset, val_dataset, test_dataset, device)

        # 绘制训练曲线
        plot_metrics(str(experiment_dir), history)

        # all模式下执行测试
        if args.mode == 'all':
            evaluate(config_dict, timestamp, device)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_dict = {"base_path": str(config.base_path)}
        evaluate(config_dict, args.timestamp, device)


if __name__ == '__main__':
    main()
