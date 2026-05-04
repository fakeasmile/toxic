"""BERT多任务学习模型训练与测试流水线。

核心思想：概念向量不作为输入特征，而是作为辅助训练目标。
BERT同时学习毒性分类（主任务）和概念向量预测（辅助任务），
概念信息作为正则化信号引导BERT学到更有结构的语义表示。

使用示例:
    # 训练+测试
    python utils/multitask_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --template likert --use_deterministic

    # 仅测试
    python utils/multitask_pipeline.py --mode test --timestamp 20260504-120000 --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --template likert
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.multitask_config import MultiTaskConfig
from models.bert import BERTMultiTask

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


class MultiTaskDataset(Dataset):
    """多任务数据集，同时包含文本编码、标签和概念向量。"""

    def __init__(self, encodings, labels, concept_vectors):
        self.encodings = encodings
        self.labels = labels
        self.concept_vectors = concept_vectors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['concept_vector'] = self.concept_vectors[idx]
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="BERT多任务学习训练与测试流水线")

    parser.add_argument('--mode', choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', default=None)
    parser.add_argument('--dataset_name', default='TOXICN')
    parser.add_argument('--model_name', default='Qwen2.5-7B-Instruct-AWQ')
    parser.add_argument('--template', default='likert')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--concept_loss_weight', type=float, default=None)

    return parser.parse_args()


def update_config(args):
    config = MultiTaskConfig()

    config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    config.template = args.template

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
    if args.concept_loss_weight is not None:
        config.concept_loss_weight = args.concept_loss_weight

    return config


def load_data_with_concepts(config, mode):
    """加载原始文本数据和概念向量，按content字段对齐。"""
    text_path = config.train_path if mode == "train" else config.test_path
    concept_path = config.train_concept_path if mode == "train" else config.test_concept_path

    with open(text_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    with open(concept_path, "r", encoding="utf-8") as f:
        concept_data = json.load(f)

    # 按content对齐
    concept_map = {}
    for item in concept_data:
        concept_map[item['content']] = {
            'concept': item['concept'],
            'toxic': item['toxic']
        }

    texts, labels, concepts = [], [], []
    for item in text_data:
        content = item['content']
        if content in concept_map:
            texts.append(content)
            labels.append(item['toxic'])
            concepts.append(concept_map[content]['concept'])

    return texts, labels, np.array(concepts, dtype=np.float32)


def create_dataset(texts, labels, concepts, tokenizer, max_seq_length):
    encodings = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_seq_length, return_tensors='pt'
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    concepts_tensor = torch.tensor(concepts, dtype=torch.float32)
    return MultiTaskDataset(encodings, labels_tensor, concepts_tensor)


def plot_metrics(config, epochs, val_losses, val_f1_scores, test_f1_scores, test_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('BERTMultiTask Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='-.', label='Test F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train(config, train_dataset, val_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    concept_dim = train_dataset.concept_vectors.shape[1]
    model = BERTMultiTask(
        bert_path=str(config.bert_path),
        concept_dim=concept_dim,
        dropout_rate=config.dropout_rate,
        concept_loss_weight=config.concept_loss_weight,
    ).to(device)

    print(f">>> 模型: BERTMultiTask, concept_dim={concept_dim}, λ={config.concept_loss_weight}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list = []
    val_loss_history = []
    val_f1_history = []
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
            concept_vector = batch['concept_vector'].to(device)

            optimizer.zero_grad()
            loss, cls_logits, concept_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                concept_vector=concept_vector,
                token_type_ids=token_type_ids,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False):
                loss, cls_logits, _ = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device),
                    concept_vector=batch['concept_vector'].to(device),
                    token_type_ids=batch['token_type_ids'].to(device),
                )
                total_val_loss += loss.item()
                val_preds.extend(torch.argmax(cls_logits, dim=1).cpu().numpy())
                val_labels_list.extend(batch['labels'].numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        # ========== 测试集评估（仅观察）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} [Test]", leave=False):
                loss, cls_logits, _ = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device),
                    concept_vector=batch['concept_vector'].to(device),
                    token_type_ids=batch['token_type_ids'].to(device),
                )
                total_test_loss += loss.item()
                test_preds.extend(torch.argmax(cls_logits, dim=1).cpu().numpy())
                test_labels_list.extend(batch['labels'].numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        test_f1_history.append(test_f1)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}, Val F1={val_f1:.4f}, "
              f"Test Loss={avg_test_loss:.4f}, Test F1={test_f1:.4f}")

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

    return epoch_list, val_loss_history, val_f1_history, test_f1_history, test_loss_history


def evaluate(config, timestamp):
    experiment_dir = config.base_path / "experiments_multitask" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(saved_config.bert_path))

    test_texts, test_labels, test_concepts = load_data_with_concepts(saved_config, "test")
    test_dataset = create_dataset(test_texts, test_labels, test_concepts, tokenizer, int(saved_config.max_seq_length))
    test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

    concept_dim = test_dataset.concept_vectors.shape[1]
    model = BERTMultiTask(
        bert_path=str(saved_config.bert_path),
        concept_dim=concept_dim,
        dropout_rate=saved_config.dropout_rate,
        concept_loss_weight=saved_config.concept_loss_weight,
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            _, cls_logits, _ = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
            )
            all_preds.extend(torch.argmax(cls_logits, dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("  BERTMultiTask 测试集评估结果")
    print("=" * 30)
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 (Macro):        {f1:.4f}")
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
        f.write(f"BERTMultiTask 测试集评估结果\n{'=' * 30}\n")
        f.write(f"Precision (Macro): {precision:.4f}\nRecall (Macro):    {recall:.4f}\nF1 (Macro):        {f1:.4f}\n")
        f.write(f"{'-' * 30}\n详细分类报告:\n{report}\n{'=' * 30}\n")


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = update_config(args)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.base_path / "experiments_multitask" / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

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
            "concept_loss_weight": config.concept_loss_weight,
        }
        with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")

        tokenizer = BertTokenizer.from_pretrained(str(config.bert_path))
        train_texts, train_labels, train_concepts = load_data_with_concepts(config, "train")
        test_texts, test_labels, test_concepts = load_data_with_concepts(config, "test")

        train_texts_split, val_texts_split, train_labels_split, val_labels_split, train_concepts_split, val_concepts_split = train_test_split(
            train_texts, train_labels, train_concepts, test_size=0.1, stratify=train_labels, random_state=config.seed
        )

        train_dataset = create_dataset(train_texts_split, train_labels_split, train_concepts_split, tokenizer, config.max_seq_length)
        val_dataset = create_dataset(val_texts_split, val_labels_split, val_concepts_split, tokenizer, config.max_seq_length)
        test_dataset = create_dataset(test_texts, test_labels, test_concepts, tokenizer, config.max_seq_length)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        metrics = train(config, train_dataset, val_dataset, test_dataset)
        plot_metrics(config, *metrics)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = MultiTaskConfig()
        config.dataset_name = args.dataset_name
        config.model_name = args.model_name
        config.template = args.template
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
