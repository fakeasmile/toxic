import argparse
import json
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.lrd_cbm_config import LRDCBMConfig
from models.lrd_cbm import LRDCBMModel

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def load_rationale_data(config, dataset_name, split):
    path = config.processed_path / dataset_name / config.llm_model_name / "lrd_cbm" / f"rationale_{split}.json"
    if not path.exists():
        print(f">>> 警告: rationale文件不存在: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_concept_scores(config, dataset_name, split):
    path = config.processed_path / dataset_name / config.llm_model_name / "lrd_cbm" / f"concept_scores_{split}.json"
    if not path.exists():
        print(f">>> 警告: concept_scores文件不存在: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_raw_data(dataset_name, split):
    path = Path(__file__).parent.parent / "data" / "raw" / dataset_name / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class LRDCBMDataset(Dataset):
    def __init__(self, texts, labels, rationales, concept_scores, tokenizer, max_length, rationale_max_length):
        self.labels = labels
        self.has_rationale = rationales is not None
        self.has_concept = concept_scores is not None

        self.text_encodings = tokenizer(
            texts, max_length=max_length, padding=False, truncation=True, return_tensors=None
        )

        if self.has_rationale:
            self.rationale_encodings = tokenizer(
                rationales, max_length=rationale_max_length, padding=False, truncation=True, return_tensors=None
            )
        else:
            dummy_len = len(texts)
            self.rationale_encodings = {
                "input_ids": [[] for _ in range(dummy_len)],
                "attention_mask": [[] for _ in range(dummy_len)],
            }

        if self.has_concept:
            self.concept_labels = [torch.tensor(cs, dtype=torch.float32) for cs in concept_scores]
        else:
            self.concept_labels = None

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.text_encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.text_encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.has_rationale:
            item["rationale_ids"] = torch.tensor(self.rationale_encodings["input_ids"][idx], dtype=torch.long)
            item["rationale_mask"] = torch.tensor(self.rationale_encodings["attention_mask"][idx], dtype=torch.long)
        else:
            max_rlen = max(len(x) for x in self.rationale_encodings["input_ids"]) if self.rationale_encodings["input_ids"] else 1
            item["rationale_ids"] = torch.zeros(max_rlen, dtype=torch.long)
            item["rationale_mask"] = torch.zeros(max_rlen, dtype=torch.long)

        if self.has_concept:
            item["concept_labels"] = self.concept_labels[idx]

        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    max_input_len = max(item["input_ids"].size(0) for item in batch)
    max_rationale_len = max(item["rationale_ids"].size(0) for item in batch)
    max_rationale_len = max(max_rationale_len, 1)

    input_ids_list = []
    attention_mask_list = []
    rationale_ids_list = []
    rationale_mask_list = []
    labels_list = []
    concept_labels_list = []
    has_concept = "concept_labels" in batch[0]

    for item in batch:
        pad_len = max_input_len - item["input_ids"].size(0)
        input_ids_list.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask_list.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))

        r_pad_len = max_rationale_len - item["rationale_ids"].size(0)
        rationale_ids_list.append(torch.cat([item["rationale_ids"], torch.zeros(r_pad_len, dtype=torch.long)]))
        rationale_mask_list.append(torch.cat([item["rationale_mask"], torch.zeros(r_pad_len, dtype=torch.long)]))

        labels_list.append(item["labels"])

        if has_concept:
            concept_labels_list.append(item["concept_labels"])

    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "rationale_ids": torch.stack(rationale_ids_list),
        "rationale_mask": torch.stack(rationale_mask_list),
        "labels": torch.stack(labels_list),
    }

    if has_concept:
        result["concept_labels"] = torch.stack(concept_labels_list)

    return result


def evaluate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_concept_probs = []
    has_concept = "concept_labels" in loader.dataset[0] if len(loader.dataset) > 0 else False

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rationale_ids = batch["rationale_ids"].to(device)
            rationale_mask = batch["rationale_mask"].to(device)
            labels = batch["labels"].to(device)
            concept_labels = batch.get("concept_labels", None)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            outputs = model(input_ids, attention_mask, rationale_ids, rationale_mask, labels, concept_labels)
            if len(outputs) == 3:
                logits, concept_probs, loss = outputs
            else:
                logits, concept_probs = outputs
                loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_probs.extend(concept_probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels, all_concept_probs


def train(config, model, train_loader, val_loader, test_loader, device):
    plm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "text_plm" in name or "rationale_plm" in name:
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

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list = []
    train_loss_history = []
    val_f1_history = []
    test_f1_history = []

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        has_concept = "concept_labels" in train_loader.dataset[0] if len(train_loader.dataset) > 0 else False

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rationale_ids = batch["rationale_ids"].to(device)
            rationale_mask = batch["rationale_mask"].to(device)
            labels = batch["labels"].to(device)
            concept_labels = batch["concept_labels"].to(device) if has_concept else None

            optimizer.zero_grad()
            logits, concept_probs, loss = model(
                input_ids, attention_mask, rationale_ids, rationale_mask, labels, concept_labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_preds, val_labels, _ = evaluate_epoch(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        test_loss, test_preds, test_labels, _ = evaluate_epoch(model, test_loader, device)
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        train_loss_history.append(avg_train_loss)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epoch_list, train_loss_history, color='tab:blue', label='Train Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('LRD-CBM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epoch_list, val_f1_history, color='tab:blue', label='Val F1')
    ax2.plot(epoch_list, test_f1_history, color='tab:red', linestyle='--', label='Test F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "training_curves.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()

    return {
        "epochs": epoch_list,
        "train_loss": train_loss_history,
        "val_f1": val_f1_history,
        "test_f1": test_f1_history,
    }


def evaluate(config, timestamp):
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config_dict = json.load(f)

    saved_config = LRDCBMConfig()
    for key, value in saved_config_dict.items():
        setattr(saved_config, key, value)
    saved_config.base_path = Path(saved_config.base_path) if isinstance(saved_config.base_path, str) else saved_config.base_path
    saved_config.experiment_path = Path(saved_config.experiment_path) if isinstance(saved_config.experiment_path, str) else saved_config.experiment_path
    saved_config.models_path = Path(saved_config.models_path) if isinstance(saved_config.models_path, str) else saved_config.models_path
    saved_config.raw_data_path = Path(saved_config.raw_data_path) if isinstance(saved_config.raw_data_path, str) else saved_config.raw_data_path
    saved_config.processed_path = Path(saved_config.processed_path) if isinstance(saved_config.processed_path, str) else saved_config.processed_path

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(Path(saved_config.models_path) / saved_config.plm_name))

    raw_test_data = load_raw_data(saved_config.dataset_name, "test")
    rationale_data = load_rationale_data(saved_config, saved_config.dataset_name, "test")
    concept_scores_data = load_concept_scores(saved_config, saved_config.dataset_name, "test")

    rationales_dict = {}
    if rationale_data is not None:
        for item in rationale_data:
            rationales_dict[item["content"]] = item.get("rationale", "")

    concept_scores_dict = {}
    if concept_scores_data is not None:
        for item in concept_scores_data:
            concept_scores_dict[item["content"]] = item.get("concept_scores", None)

    test_texts = []
    test_labels = []
    test_rationales = []
    test_concept_scores = []

    for item in raw_test_data:
        test_texts.append(item["content"])
        test_labels.append(item["toxic"])
        test_rationales.append(rationales_dict.get(item["content"], ""))
        cs = concept_scores_dict.get(item["content"], None)
        test_concept_scores.append(cs)

    has_concept = any(cs is not None for cs in test_concept_scores)
    if not has_concept:
        test_concept_scores = None
    else:
        test_concept_scores = [cs if cs is not None else [0.0] * saved_config.num_concepts for cs in test_concept_scores]

    if not saved_config.use_rationale:
        test_rationales = None

    test_dataset = LRDCBMDataset(
        test_texts, test_labels, test_rationales, test_concept_scores,
        tokenizer, saved_config.max_length, saved_config.rationale_max_length
    )
    test_loader = DataLoader(
        test_dataset, batch_size=saved_config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = LRDCBMModel(
        plm_name=str(Path(saved_config.models_path) / saved_config.plm_name),
        num_concepts=saved_config.num_concepts,
        num_classes=saved_config.num_classes,
        dropout=saved_config.dropout,
        share_plm=saved_config.share_plm,
        concept_loss_weight=saved_config.concept_loss_weight,
        use_rationale=saved_config.use_rationale,
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pt", map_location=device, weights_only=False))
    model.to(device).eval()

    all_preds = []
    all_labels = []
    all_concept_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rationale_ids = batch["rationale_ids"].to(device)
            rationale_mask = batch["rationale_mask"].to(device)

            logits, concept_probs = model(input_ids, attention_mask, rationale_ids, rationale_mask)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
            all_concept_probs.extend(concept_probs.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      LRD-CBM 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(np.mean(np.array(all_preds) == np.array(all_labels)), 4),
            "macro_f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(experiment_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("LRD-CBM 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    predictions = []
    for i in range(len(all_preds)):
        pred_item = {
            "content": test_texts[i],
            "true_label": int(all_labels[i]),
            "pred_label": int(all_preds[i]),
            "concept_scores": all_concept_probs[i].tolist(),
        }
        if test_rationales is not None:
            pred_item["rationale"] = test_rationales[i]
        predictions.append(pred_item)

    with open(experiment_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="LRD-CBM 训练与测试流水线")

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--plm_name', type=str, default='chinese-roberta-wwm-ext')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--freeze_plm', action='store_true', default=False)
    parser.add_argument('--no_rationale', action='store_true', default=False)
    parser.add_argument('--no_concept_loss', action='store_true', default=False)
    parser.add_argument('--share_plm', type=bool, default=True)
    parser.add_argument('--concept_loss_weight', type=float, default=None)

    args = parser.parse_args()

    if args.mode in ['all', 'train']:
        config = LRDCBMConfig()
        config.dataset_name = args.dataset_name
        config.plm_name = args.plm_name

        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.lr is not None:
            config.lr = args.lr
        if args.patience is not None:
            config.patience = args.patience
        if args.concept_loss_weight is not None:
            config.concept_loss_weight = args.concept_loss_weight

        config.freeze_plm = args.freeze_plm
        config.use_rationale = not args.no_rationale
        config.share_plm = args.share_plm

        if args.no_concept_loss:
            config.concept_loss_weight = 0.0

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        config_dict = {}
        for key, value in sorted(config.__dict__.items()):
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        config_dict["timestamp"] = timestamp

        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式 (Reproducibility Enabled)")
        else:
            print(">>> 已禁用确定性模式 (Randomness Enabled), 结果将不可复现")

        raw_train_data = load_raw_data(config.dataset_name, "train")
        raw_test_data = load_raw_data(config.dataset_name, "test")

        rationale_data = load_rationale_data(config, config.dataset_name, "train")
        concept_scores_data = load_concept_scores(config, config.dataset_name, "train")

        rationales_dict = {}
        if rationale_data is not None:
            for item in rationale_data:
                rationales_dict[item["content"]] = item.get("rationale", "")

        concept_scores_dict = {}
        if concept_scores_data is not None:
            for item in concept_scores_data:
                concept_scores_dict[item["content"]] = item.get("concept_scores", None)

        train_texts = []
        train_labels = []
        train_rationales = []
        train_concept_scores = []

        for item in raw_train_data:
            train_texts.append(item["content"])
            train_labels.append(item["toxic"])
            train_rationales.append(rationales_dict.get(item["content"], ""))
            cs = concept_scores_dict.get(item["content"], None)
            train_concept_scores.append(cs)

        has_concept = any(cs is not None for cs in train_concept_scores)
        if not has_concept:
            train_concept_scores = None
        else:
            train_concept_scores = [cs if cs is not None else [0.0] * config.num_concepts for cs in train_concept_scores]

        if not config.use_rationale:
            train_rationales = None

        train_texts_np = np.array(train_texts)
        train_labels_np = np.array(train_labels)

        train_idx, val_idx = train_test_split(
            np.arange(len(train_texts)), test_size=0.1,
            stratify=train_labels_np, random_state=config.seed
        )

        split_texts = train_texts_np[train_idx].tolist()
        split_labels = train_labels_np[train_idx].tolist()
        val_texts = train_texts_np[val_idx].tolist()
        val_labels = train_labels_np[val_idx].tolist()

        split_rationales = None
        val_rationales = None
        if train_rationales is not None:
            train_rationales_np = np.array(train_rationales)
            split_rationales = train_rationales_np[train_idx].tolist()
            val_rationales = train_rationales_np[val_idx].tolist()

        split_concept_scores = None
        val_concept_scores = None
        if train_concept_scores is not None:
            train_concept_scores_np = np.array(train_concept_scores, dtype=object)
            split_concept_scores = train_concept_scores_np[train_idx].tolist()
            val_concept_scores = train_concept_scores_np[val_idx].tolist()

        test_texts = [item["content"] for item in raw_test_data]
        test_labels = [item["toxic"] for item in raw_test_data]

        test_rationale_data = load_rationale_data(config, config.dataset_name, "test")
        test_concept_scores_data = load_concept_scores(config, config.dataset_name, "test")

        test_rationales_dict = {}
        if test_rationale_data is not None:
            for item in test_rationale_data:
                test_rationales_dict[item["content"]] = item.get("rationale", "")

        test_concept_scores_dict = {}
        if test_concept_scores_data is not None:
            for item in test_concept_scores_data:
                test_concept_scores_dict[item["content"]] = item.get("concept_scores", None)

        test_rationales = []
        test_concept_scores = []
        for item in raw_test_data:
            test_rationales.append(test_rationales_dict.get(item["content"], ""))
            cs = test_concept_scores_dict.get(item["content"], None)
            test_concept_scores.append(cs)

        test_has_concept = any(cs is not None for cs in test_concept_scores)
        if not test_has_concept:
            test_concept_scores = None
        else:
            test_concept_scores = [cs if cs is not None else [0.0] * config.num_concepts for cs in test_concept_scores]

        if not config.use_rationale:
            test_rationales = None

        tokenizer = AutoTokenizer.from_pretrained(str(config.models_path / config.plm_name))

        train_dataset = LRDCBMDataset(
            split_texts, split_labels, split_rationales, split_concept_scores,
            tokenizer, config.max_length, config.rationale_max_length
        )
        val_dataset = LRDCBMDataset(
            val_texts, val_labels, val_rationales, val_concept_scores,
            tokenizer, config.max_length, config.rationale_max_length
        )
        test_dataset = LRDCBMDataset(
            test_texts, test_labels, test_rationales, test_concept_scores,
            tokenizer, config.max_length, config.rationale_max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> 正在使用设备: {device}")

        model = LRDCBMModel(
            plm_name=str(config.models_path / config.plm_name),
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            dropout=config.dropout,
            share_plm=config.share_plm,
            concept_loss_weight=config.concept_loss_weight,
            use_rationale=config.use_rationale,
        )

        if config.freeze_plm:
            for param in model.text_plm.parameters():
                param.requires_grad = False
            if not config.share_plm and hasattr(model, 'rationale_plm'):
                for param in model.rationale_plm.parameters():
                    param.requires_grad = False
            print(">>> 已冻结PLM参数")

        model.to(device)

        train(config, model, train_loader, val_loader, test_loader, device)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = LRDCBMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
