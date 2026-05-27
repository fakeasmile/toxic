import argparse
import json
import sys
import copy
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.cotd_cbm_config import CoTDCBMConfig
from models.cotd_cbm import CoTDCBMModel

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def load_distill_data(config, dataset_name, split):
    distill_path = config.processed_path / dataset_name / config.llm_model_name / "cotd_cbm" / f"distill_{split}.json"
    if not distill_path.exists():
        print(f">>> 警告: 蒸馏数据文件不存在: {distill_path}")
        return None
    with open(distill_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_raw_data(dataset_name, split, raw_data_path):
    raw_path = raw_data_path / dataset_name / f"{split}.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class CoTDCBMDataset(Dataset):
    def __init__(self, texts, labels, soft_labels, concept_scores, tokenizer, max_length):
        self.labels = labels
        self.soft_labels = soft_labels
        self.concept_scores = concept_scores
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.soft_labels is not None:
            item["soft_labels"] = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        if self.concept_scores is not None:
            item["concept_labels"] = torch.tensor(self.concept_scores[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    soft_labels = []
    concept_labels = []
    has_soft = "soft_labels" in batch[0]
    has_concept = "concept_labels" in batch[0]

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(item["labels"])
        if has_soft:
            soft_labels.append(item["soft_labels"])
        if has_concept:
            concept_labels.append(item["concept_labels"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }
    if has_soft:
        result["soft_labels"] = torch.stack(soft_labels)
    if has_concept:
        result["concept_labels"] = torch.stack(concept_labels)
    return result


def evaluate_epoch(model, loader, device, has_soft_labels, has_concept_labels):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_concept_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            soft_labels = batch["soft_labels"].to(device) if has_soft_labels else None
            concept_labels = batch["concept_labels"].to(device) if has_concept_labels else None

            logits, concept_probs, loss = model(input_ids, attention_mask, labels, soft_labels, concept_labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_probs.extend(concept_probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels, all_concept_probs


def train(config, model, train_loader, val_loader, test_loader, device):
    plm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "plm" in name:
                plm_params.append(param)
            else:
                other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": plm_params, "lr": config.plm_lr},
        {"params": other_params, "lr": config.lr},
    ], weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    has_soft_labels = any("soft_labels" in b for b in [train_loader.dataset[0]]) if len(train_loader.dataset) > 0 else False
    has_concept_labels = any("concept_labels" in b for b in [train_loader.dataset[0]]) if len(train_loader.dataset) > 0 else False

    sample = train_loader.dataset[0]
    has_soft_labels = "soft_labels" in sample
    has_concept_labels = "concept_labels" in sample

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list = []
    train_loss_history = []
    val_loss_history = []
    val_f1_history = []
    test_f1_history = []

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            soft_labels = batch["soft_labels"].to(device) if has_soft_labels else None
            concept_labels = batch["concept_labels"].to(device) if has_concept_labels else None

            optimizer.zero_grad()
            logits, concept_probs, loss = model(input_ids, attention_mask, labels, soft_labels, concept_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_preds, val_labels, _ = evaluate_epoch(model, val_loader, device, has_soft_labels, has_concept_labels)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        test_loss, test_preds, test_labels, _ = evaluate_epoch(model, test_loader, device, has_soft_labels, has_concept_labels)
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)
        test_f1_history.append(test_f1)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}, "
              f"Test Loss = {test_loss:.4f}, Test F1 = {test_f1:.4f}")

        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> 发现更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停触发: 验证集F1已连续 {config.patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pt")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    plot_metrics(config, epoch_list, train_loss_history, val_loss_history, val_f1_history, test_f1_history)

    return {
        "epochs": epoch_list,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_f1": val_f1_history,
        "test_f1": test_f1_history,
    }


def plot_metrics(config, epochs, train_losses, val_losses, val_f1_scores, test_f1_scores):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, train_losses, color='tab:blue', label='Train Loss')
    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('CoTD-CBM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='--', label='Test F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def evaluate(config, timestamp):
    experiment_dir = config.experiment_path / timestamp if hasattr(config, 'experiment_path') and not isinstance(config.experiment_path, Path) else Path(str(config.base_path)) / "experiments" / timestamp

    base_path = Path(str(config.base_path)) if hasattr(config, 'base_path') else project_root
    experiment_dir = base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config_dict = json.load(f)

    saved_config = SimpleNamespace()
    for k, v in saved_config_dict.items():
        setattr(saved_config, k, v)

    saved_config.models_path = Path(saved_config.models_path)
    saved_config.raw_data_path = Path(saved_config.raw_data_path)
    saved_config.processed_path = Path(saved_config.processed_path)
    saved_config.experiment_path = Path(saved_config.experiment_path)
    saved_config.base_path = Path(saved_config.base_path)

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(saved_config.models_path / saved_config.plm_name))

    raw_test_data = load_raw_data(saved_config.dataset_name, "test", saved_config.raw_data_path)
    distill_test_data = load_distill_data(saved_config, saved_config.dataset_name, "test")

    test_texts = [item["content"] for item in raw_test_data]
    test_labels = [item["toxic"] for item in raw_test_data]

    test_soft_labels = None
    test_concept_scores = None
    if distill_test_data is not None and not getattr(saved_config, 'no_soft_label', False):
        distill_map = {item["content"]: item for item in distill_test_data}
        test_soft_labels = []
        for text in test_texts:
            if text in distill_map:
                test_soft_labels.append(distill_map[text]["soft_label"])
            else:
                test_soft_labels.append([0.5, 0.5])

    if distill_test_data is not None and not getattr(saved_config, 'no_concept_loss', False):
        distill_map = {item["content"]: item for item in distill_test_data}
        test_concept_scores = []
        for text in test_texts:
            if text in distill_map:
                test_concept_scores.append(distill_map[text]["concept_scores"])
            else:
                test_concept_scores.append([0.5] * saved_config.num_concepts)

    test_dataset = CoTDCBMDataset(
        test_texts, test_labels, test_soft_labels, test_concept_scores,
        tokenizer, saved_config.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=saved_config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CoTDCBMModel(
        plm_name=str(saved_config.models_path / saved_config.plm_name),
        num_concepts=saved_config.num_concepts,
        num_classes=saved_config.num_classes,
        dropout=saved_config.dropout,
        concept_loss_weight=saved_config.concept_loss_weight,
        soft_label_weight=saved_config.soft_label_weight,
        use_residual=not getattr(saved_config, 'no_residual', False),
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pt", map_location=device, weights_only=False))
    model.to(device).eval()

    has_soft = test_soft_labels is not None
    has_concept = test_concept_scores is not None

    all_preds = []
    all_labels = []
    all_concept_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            soft_labels = batch["soft_labels"].to(device) if has_soft else None
            concept_labels = batch["concept_labels"].to(device) if has_concept else None

            logits, concept_probs = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_probs.extend(concept_probs.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      CoTD-CBM 测试集评估结果")
    print("=" * 30)
    print(f"准确率 (Accuracy):         {accuracy:.4f}")
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(accuracy, 4),
            "macro_f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("CoTD-CBM 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"准确率 (Accuracy):         {accuracy:.4f}\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    distill_map = {}
    if distill_test_data is not None:
        distill_map = {item["content"]: item for item in distill_test_data}

    predictions = []
    for i in range(len(all_preds)):
        pred_item = {
            "content": test_texts[i],
            "true_label": int(all_labels[i]),
            "pred_label": int(all_preds[i]),
            "concept_scores": all_concept_probs[i].tolist() if isinstance(all_concept_probs[i], np.ndarray) else all_concept_probs[i],
        }
        if test_texts[i] in distill_map:
            distill_item = distill_map[test_texts[i]]
            pred_item["soft_label"] = distill_item.get("soft_label", None)
            pred_item["rationale"] = distill_item.get("rationale", "")
        predictions.append(pred_item)

    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="CoTD-CBM 训练与测试流水线")

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--plm_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--freeze_plm', action='store_true', default=False)
    parser.add_argument('--no_soft_label', action='store_true', default=False)
    parser.add_argument('--no_concept_loss', action='store_true', default=False)
    parser.add_argument('--no_residual', action='store_true', default=False)
    parser.add_argument('--soft_label_weight', type=float, default=None)
    parser.add_argument('--concept_loss_weight', type=float, default=None)

    args = parser.parse_args()

    if args.mode in ['all', 'train']:
        config = CoTDCBMConfig()

        config.dataset_name = args.dataset_name
        if args.plm_name is not None:
            config.plm_name = args.plm_name
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.lr is not None:
            config.lr = args.lr
        if args.patience is not None:
            config.patience = args.patience
        if args.freeze_plm:
            config.freeze_plm = True
        if args.no_soft_label:
            config.no_soft_label = True
        else:
            config.no_soft_label = False
        if args.no_concept_loss:
            config.no_concept_loss = True
        else:
            config.no_concept_loss = False
        if args.no_residual:
            config.no_residual = True
            config.use_residual = False
        else:
            config.no_residual = False
        if args.soft_label_weight is not None:
            config.soft_label_weight = args.soft_label_weight
        if args.concept_loss_weight is not None:
            config.concept_loss_weight = args.concept_loss_weight

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir
        config.timestamp = timestamp

        config_dict = {}
        for k, v in vars(config.__class__).items():
            if not k.startswith("_") and not callable(v):
                config_dict[k] = str(v) if isinstance(v, Path) else v
        for k, v in vars(config).items():
            if not k.startswith("_"):
                config_dict[k] = str(v) if isinstance(v, Path) else v
        config_dict["timestamp"] = timestamp
        config_dict["experiment_path"] = str(config.experiment_path)
        config_dict["base_path"] = str(config.base_path)
        config_dict["raw_data_path"] = str(config.raw_data_path)
        config_dict["processed_path"] = str(config.processed_path)
        config_dict["models_path"] = str(config.models_path)

        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式 (Reproducibility Enabled)")
        else:
            print(">>> 已禁用确定性模式 (Randomness Enabled), 结果将不可复现")

        raw_train_data = load_raw_data(config.dataset_name, "train", config.raw_data_path)
        raw_test_data = load_raw_data(config.dataset_name, "test", config.raw_data_path)

        distill_train_data = load_distill_data(config, config.dataset_name, "train")
        distill_test_data = load_distill_data(config, config.dataset_name, "test")

        train_texts = [item["content"] for item in raw_train_data]
        train_labels = [item["toxic"] for item in raw_train_data]

        test_texts = [item["content"] for item in raw_test_data]
        test_labels = [item["toxic"] for item in raw_test_data]

        train_soft_labels = None
        train_concept_scores = None
        if distill_train_data is not None and not config.no_soft_label:
            distill_map = {item["content"]: item for item in distill_train_data}
            train_soft_labels = []
            for text in train_texts:
                if text in distill_map:
                    train_soft_labels.append(distill_map[text]["soft_label"])
                else:
                    train_soft_labels.append([0.5, 0.5])

        if distill_train_data is not None and not config.no_concept_loss:
            distill_map = {item["content"]: item for item in distill_train_data}
            train_concept_scores = []
            for text in train_texts:
                if text in distill_map:
                    train_concept_scores.append(distill_map[text]["concept_scores"])
                else:
                    train_concept_scores.append([0.5] * config.num_concepts)

        test_soft_labels = None
        test_concept_scores = None
        if distill_test_data is not None and not config.no_soft_label:
            distill_map = {item["content"]: item for item in distill_test_data}
            test_soft_labels = []
            for text in test_texts:
                if text in distill_map:
                    test_soft_labels.append(distill_map[text]["soft_label"])
                else:
                    test_soft_labels.append([0.5, 0.5])

        if distill_test_data is not None and not config.no_concept_loss:
            distill_map = {item["content"]: item for item in distill_test_data}
            test_concept_scores = []
            for text in test_texts:
                if text in distill_map:
                    test_concept_scores.append(distill_map[text]["concept_scores"])
                else:
                    test_concept_scores.append([0.5] * config.num_concepts)

        train_indices, val_indices = train_test_split(
            range(len(train_texts)), test_size=0.1,
            stratify=train_labels, random_state=config.seed
        )

        val_texts = [train_texts[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]
        val_soft_labels = None
        val_concept_scores = None
        if train_soft_labels is not None:
            val_soft_labels = [train_soft_labels[i] for i in val_indices]
        if train_concept_scores is not None:
            val_concept_scores = [train_concept_scores[i] for i in val_indices]

        train_texts_split = [train_texts[i] for i in train_indices]
        train_labels_split = [train_labels[i] for i in train_indices]
        train_soft_labels_split = None
        train_concept_scores_split = None
        if train_soft_labels is not None:
            train_soft_labels_split = [train_soft_labels[i] for i in train_indices]
        if train_concept_scores is not None:
            train_concept_scores_split = [train_concept_scores[i] for i in train_indices]

        tokenizer = AutoTokenizer.from_pretrained(str(config.models_path / config.plm_name))

        train_dataset = CoTDCBMDataset(
            train_texts_split, train_labels_split, train_soft_labels_split, train_concept_scores_split,
            tokenizer, config.max_length
        )
        val_dataset = CoTDCBMDataset(
            val_texts, val_labels, val_soft_labels, val_concept_scores,
            tokenizer, config.max_length
        )
        test_dataset = CoTDCBMDataset(
            test_texts, test_labels, test_soft_labels, test_concept_scores,
            tokenizer, config.max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> 正在使用设备: {device}")

        model = CoTDCBMModel(
            plm_name=str(config.models_path / config.plm_name),
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            dropout=config.dropout,
            concept_loss_weight=config.concept_loss_weight,
            soft_label_weight=config.soft_label_weight,
            use_residual=config.use_residual,
        ).to(device)

        if config.freeze_plm:
            for param in model.plm.parameters():
                param.requires_grad = False
            print(">>> 已冻结PLM参数")

        train(config, model, train_loader, val_loader, test_loader, device)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = CoTDCBMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
