import argparse
import json
import sys
import time
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
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

from configs.ttt_cbm_config import TTTCBMConfig
from models.ttt_cbm import TTTCBMModel

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def load_concept_scores(config, dataset_name, split):
    path1 = config.processed_path / dataset_name / config.llm_model_name / "cotd_cbm" / f"distill_{split}.json"
    if path1.exists():
        with open(path1, "r", encoding="utf-8") as f:
            return json.load(f)

    path2 = config.processed_path / dataset_name / config.llm_model_name / "rcwn_concepts" / f"concept_{split}.json"
    if path2.exists():
        with open(path2, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def load_raw_data(dataset_name, split):
    path = Path(__file__).parent.parent / "data" / "raw" / dataset_name / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TTTCBMDataset(Dataset):
    def __init__(self, texts, labels, concept_scores, tokenizer, max_length):
        self.labels = labels
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
        if self.concept_scores is not None:
            item["concept_labels"] = torch.tensor(self.concept_scores[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    max_len = max(ids.size(0) for ids in input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    for ids, mask in zip(input_ids, attention_mask):
        pad_len = max_len - ids.size(0)
        padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": labels,
    }

    if "concept_labels" in batch[0]:
        result["concept_labels"] = torch.stack([item["concept_labels"] for item in batch])

    return result


def evaluate_epoch(model, loader, device, use_ttt=False):
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
            concept_labels = batch.get("concept_labels")
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            if use_ttt:
                model.ttt_adapt(input_ids, attention_mask)

            outputs = model(input_ids, attention_mask, labels, concept_labels)
            logits, concept_probs, loss = outputs

            if use_ttt:
                model.ttt_restore()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_probs.extend(concept_probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels, all_concept_probs


def train(config, model, train_loader, val_loader, test_loader, device):
    plm_params = list(model.plm.parameters())
    other_params = [p for p in model.parameters() if p not in set(plm_params)]

    optimizer = torch.optim.AdamW([
        {"params": plm_params, "lr": config.plm_lr},
        {"params": other_params, "lr": config.lr},
    ], weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list = []
    val_loss_history = []
    val_f1_history = []
    test_loss_history = []
    test_f1_history = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            concept_labels = batch.get("concept_labels")
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            optimizer.zero_grad()
            logits, concept_probs, loss = model(input_ids, attention_mask, labels, concept_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps

        val_loss, val_preds, val_labels, _ = evaluate_epoch(model, val_loader, device, use_ttt=False)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        test_loss, test_preds, test_labels, _ = evaluate_epoch(model, test_loader, device, use_ttt=config.ttt_enabled)
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)
        test_loss_history.append(test_loss)
        test_f1_history.append(test_f1)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}, "
              f"Test Loss={test_loss:.4f}, Test F1={test_f1:.4f}")

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
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    plot_metrics(config, epoch_list, val_loss_history, val_f1_history, test_loss_history, test_f1_history)

    return {
        "epochs": epoch_list,
        "val_losses": val_loss_history,
        "val_f1_scores": val_f1_history,
        "test_losses": test_loss_history,
        "test_f1_scores": test_f1_history,
    }


def plot_metrics(config, epochs, val_losses, val_f1_scores, test_losses, test_f1_scores):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('TTT-CBM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='-.', label='Test F1')
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
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config_dict = json.load(f)

    saved_config = TTTCBMConfig()
    for k, v in saved_config_dict.items():
        if k in ("base_path", "raw_data_path", "processed_path", "experiment_path", "models_path", "concept_path"):
            setattr(saved_config, k, Path(v))
        else:
            setattr(saved_config, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_test_data = load_raw_data(saved_config.dataset_name, "test")
    concept_test_data = load_concept_scores(saved_config, saved_config.dataset_name, "test")

    test_texts = [item["content"] for item in raw_test_data]
    test_labels = [item["toxic"] for item in raw_test_data]

    test_concept_scores = None
    if concept_test_data is not None:
        score_map = {item["content"]: item["concept_scores"] for item in concept_test_data}
        test_concept_scores = [score_map.get(text) for text in test_texts]

    plm_path = str(saved_config.models_path / saved_config.plm_name)
    tokenizer = AutoTokenizer.from_pretrained(plm_path)
    test_dataset = TTTCBMDataset(test_texts, test_labels, test_concept_scores, tokenizer, saved_config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=saved_config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TTTCBMModel(
        plm_name=plm_path,
        num_concepts=saved_config.num_concepts,
        num_classes=saved_config.num_classes,
        dropout=saved_config.dropout,
        concept_loss_weight=saved_config.concept_loss_weight,
        use_residual=saved_config.use_residual,
        ttt_enabled=saved_config.ttt_enabled,
        ttt_lr=saved_config.ttt_lr,
        ttt_steps=saved_config.ttt_steps,
        ttt_mlm_mask_ratio=saved_config.ttt_mlm_mask_ratio,
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    results = {}

    _, preds_no_ttt, labels_no_ttt, _ = evaluate_epoch(model, test_loader, device, use_ttt=False)
    f1_no_ttt = f1_score(labels_no_ttt, preds_no_ttt, average='macro')
    p_no_ttt = precision_score(labels_no_ttt, preds_no_ttt, average='macro', zero_division=0)
    r_no_ttt = recall_score(labels_no_ttt, preds_no_ttt, average='macro', zero_division=0)
    report_no_ttt = classification_report(labels_no_ttt, preds_no_ttt, target_names=["Non-Toxic", "Toxic"])
    results["without_ttt"] = {
        "precision_macro": round(p_no_ttt, 4),
        "recall_macro": round(r_no_ttt, 4),
        "f1_macro": round(f1_no_ttt, 4),
    }

    _, preds_with_ttt, labels_with_ttt, _ = evaluate_epoch(model, test_loader, device, use_ttt=True)
    f1_with_ttt = f1_score(labels_with_ttt, preds_with_ttt, average='macro')
    p_with_ttt = precision_score(labels_with_ttt, preds_with_ttt, average='macro', zero_division=0)
    r_with_ttt = recall_score(labels_with_ttt, preds_with_ttt, average='macro', zero_division=0)
    report_with_ttt = classification_report(labels_with_ttt, preds_with_ttt, target_names=["Non-Toxic", "Toxic"])
    results["with_ttt"] = {
        "precision_macro": round(p_with_ttt, 4),
        "recall_macro": round(r_with_ttt, 4),
        "f1_macro": round(f1_with_ttt, 4),
    }

    print("\n" + "=" * 30)
    print("  TTT-CBM 测试集评估结果 (Without TTT)")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {p_no_ttt:.4f}")
    print(f"召回率 (Recall - Macro):    {r_no_ttt:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1_no_ttt:.4f}")
    print("-" * 30)
    print(report_no_ttt)

    print("=" * 30)
    print("  TTT-CBM 测试集评估结果 (With TTT)")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {p_with_ttt:.4f}")
    print(f"召回率 (Recall - Macro):    {r_with_ttt:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1_with_ttt:.4f}")
    print("-" * 30)
    print(report_with_ttt)
    print("=" * 30)

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("TTT-CBM 测试集评估结果 (Without TTT)\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {p_no_ttt:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {r_no_ttt:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1_no_ttt:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report_no_ttt)
        f.write("\n\n")
        f.write("TTT-CBM 测试集评估结果 (With TTT)\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {p_with_ttt:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {r_with_ttt:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1_with_ttt:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report_with_ttt)
        f.write("\n" + "=" * 30 + "\n")

    label_names = ["Non-Toxic", "Toxic"]
    predictions = []
    for i in range(len(preds_no_ttt)):
        predictions.append({
            "index": i,
            "content": test_texts[i],
            "true_label": int(labels_no_ttt[i]),
            "true_label_name": label_names[int(labels_no_ttt[i])],
            "pred_without_ttt": int(preds_no_ttt[i]),
            "pred_without_ttt_name": label_names[int(preds_no_ttt[i])],
            "pred_with_ttt": int(preds_with_ttt[i]),
            "pred_with_ttt_name": label_names[int(preds_with_ttt[i])],
            "correct_without_ttt": bool(preds_no_ttt[i] == labels_no_ttt[i]),
            "correct_with_ttt": bool(preds_with_ttt[i] == labels_with_ttt[i]),
        })

    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="TTT-CBM 训练与测试流水线")

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)

    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--plm_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)

    parser.add_argument('--freeze_plm', action='store_true', default=False)
    parser.add_argument('--no_ttt', action='store_true', default=False)
    parser.add_argument('--no_concept_loss', action='store_true', default=False)
    parser.add_argument('--no_residual', action='store_true', default=False)

    parser.add_argument('--ttt_lr', type=float, default=None)
    parser.add_argument('--ttt_steps', type=int, default=None)
    parser.add_argument('--concept_loss_weight', type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = TTTCBMConfig()
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
        if args.no_ttt:
            config.ttt_enabled = False
        if args.no_concept_loss:
            config.concept_loss_weight = 0.0
        if args.no_residual:
            config.use_residual = False

        if args.ttt_lr is not None:
            config.ttt_lr = args.ttt_lr
        if args.ttt_steps is not None:
            config.ttt_steps = args.ttt_steps
        if args.concept_loss_weight is not None:
            config.concept_loss_weight = args.concept_loss_weight

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        config_dict = {
            "base_path": str(config.base_path),
            "raw_data_path": str(config.raw_data_path),
            "processed_path": str(config.processed_path),
            "experiment_path": str(config.experiment_path),
            "models_path": str(config.models_path),
            "concept_path": str(config.concept_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "plm_name": config.plm_name,
            "plm_hidden_size": config.plm_hidden_size,
            "max_length": config.max_length,
            "freeze_plm": config.freeze_plm,
            "plm_lr": config.plm_lr,
            "num_concepts": config.num_concepts,
            "num_classes": config.num_classes,
            "dropout": config.dropout,
            "concept_loss_weight": config.concept_loss_weight,
            "use_residual": config.use_residual,
            "ttt_enabled": config.ttt_enabled,
            "ttt_lr": config.ttt_lr,
            "ttt_steps": config.ttt_steps,
            "ttt_mlm_mask_ratio": config.ttt_mlm_mask_ratio,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "patience": config.patience,
            "llm_model_name": config.llm_model_name,
            "dataset_name": config.dataset_name,
            "timestamp": timestamp,
        }
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

        concept_train_data = load_concept_scores(config, config.dataset_name, "train")
        concept_test_data = load_concept_scores(config, config.dataset_name, "test")

        train_texts = [item["content"] for item in raw_train_data]
        train_labels = [item["toxic"] for item in raw_train_data]

        test_texts = [item["content"] for item in raw_test_data]
        test_labels = [item["toxic"] for item in raw_test_data]

        train_concept_scores = None
        if concept_train_data is not None:
            score_map = {item["content"]: item["concept_scores"] for item in concept_train_data}
            train_concept_scores = [score_map.get(text) for text in train_texts]

        test_concept_scores = None
        if concept_test_data is not None:
            score_map = {item["content"]: item["concept_scores"] for item in concept_test_data}
            test_concept_scores = [score_map.get(text) for text in test_texts]

        train_texts_np = np.array(train_texts)
        train_labels_np = np.array(train_labels)
        train_concept_np = np.array(train_concept_scores) if train_concept_scores is not None else None

        train_idx, val_idx = train_test_split(
            np.arange(len(train_texts)),
            test_size=0.1,
            stratify=train_labels_np,
            random_state=config.seed
        )

        val_texts = train_texts_np[val_idx].tolist()
        val_labels = train_labels_np[val_idx].tolist()
        val_concept_scores = train_concept_np[val_idx].tolist() if train_concept_np is not None else None

        train_texts_split = train_texts_np[train_idx].tolist()
        train_labels_split = train_labels_np[train_idx].tolist()
        train_concept_split = train_concept_np[train_idx].tolist() if train_concept_np is not None else None

        plm_path = str(config.models_path / config.plm_name)
        tokenizer = AutoTokenizer.from_pretrained(plm_path)

        train_dataset = TTTCBMDataset(train_texts_split, train_labels_split, train_concept_split, tokenizer, config.max_length)
        val_dataset = TTTCBMDataset(val_texts, val_labels, val_concept_scores, tokenizer, config.max_length)
        test_dataset = TTTCBMDataset(test_texts, test_labels, test_concept_scores, tokenizer, config.max_length)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> 正在使用设备: {device}")

        model = TTTCBMModel(
            plm_name=plm_path,
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            dropout=config.dropout,
            concept_loss_weight=config.concept_loss_weight,
            use_residual=config.use_residual,
            ttt_enabled=config.ttt_enabled,
            ttt_lr=config.ttt_lr,
            ttt_steps=config.ttt_steps,
            ttt_mlm_mask_ratio=config.ttt_mlm_mask_ratio,
        )

        if config.freeze_plm:
            for param in model.plm.parameters():
                param.requires_grad = False
            print(">>> PLM参数已冻结")

        model.to(device)

        train(config, model, train_loader, val_loader, test_loader, device)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = TTTCBMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
