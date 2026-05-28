import json
import argparse
import sys
import copy
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.scbm_llm_config import SCBMLLMConfig
from models.scbm_llm import SCBMLLMModel

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_distill_data(config, dataset_name, split):
    distill_path = config.processed_path / dataset_name / config.llm_model_name / "cotd_cbm" / f"distill_{split}.json"
    if not distill_path.exists():
        return None
    with open(distill_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_raw_data(dataset_name, split, raw_data_path):
    raw_path = raw_data_path / dataset_name / f"{split}.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class SCBMLLMDataset(Dataset):
    def __init__(self, texts, labels, soft_labels, concept_scores, tokenizer, max_length):
        self.labels = labels
        self.soft_labels = soft_labels
        self.concept_scores = concept_scores

        system_prompt = "你是一个中文有害言论检测系统。请判断以下文本是否包含有害言论。"

        self.encodings = []
        for text in texts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(
                formatted,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.encodings.append(encoded)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings[idx]["input_ids"].squeeze(0),
            "attention_mask": self.encodings[idx]["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.soft_labels is not None:
            item["soft_labels"] = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        if self.concept_scores is not None:
            item["concept_labels"] = torch.tensor(self.concept_scores[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_epoch(model, loader, device):
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
            soft_labels = batch.get("soft_labels")
            concept_labels = batch.get("concept_labels")

            if soft_labels is not None:
                soft_labels = soft_labels.to(device)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            logits, concept_probs, loss = model(
                input_ids, attention_mask, labels, soft_labels, concept_labels
            )

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_probs.extend(concept_probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels, all_concept_probs


def train(config, model, train_loader, val_loader, test_loader, device):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list = []
    val_loss_history = []
    val_f1_history = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            soft_labels = batch.get("soft_labels")
            concept_labels = batch.get("concept_labels")

            if soft_labels is not None:
                soft_labels = soft_labels.to(device)
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)

            logits, concept_probs, loss = model(
                input_ids, attention_mask, labels, soft_labels, concept_labels
            )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        val_loss, val_preds, val_labels, _ = evaluate_epoch(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        epoch_list.append(epoch + 1)
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0

            model.llm.save_pretrained(config.experiment_path / "lora_weights")
            torch.save(
                {
                    "concept_bottleneck": model.concept_bottleneck.state_dict(),
                    "classifier": model.classifier.state_dict(),
                },
                config.experiment_path / "cbm_weights.pth",
            )
            print(f">>> Best model saved (Val F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> Early stopping: Val F1 has not improved for {config.patience} epochs")
            break

    plot_metrics(config, epoch_list, val_loss_history, val_f1_history)

    print(f">>> Best model: Epoch {best_epoch}, Val F1: {best_f1:.4f}")
    return epoch_list, val_loss_history, val_f1_history


def plot_metrics(config, epochs, val_losses, val_f1_scores):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('S-CBM-LLM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> Training plot saved to: {save_path}")
    plt.close()


def evaluate(config, timestamp):
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config_dict = json.load(f)

    saved_config = copy.copy(config)
    for k, v in saved_config_dict.items():
        if k.endswith("_path") and isinstance(v, str):
            setattr(saved_config, k, Path(v))
        else:
            setattr(saved_config, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_test_data = load_raw_data(saved_config.dataset_name, "test", saved_config.raw_data_path)
    distill_test_data = load_distill_data(saved_config, saved_config.dataset_name, "test")

    test_texts = [item["content"] for item in raw_test_data]
    test_labels = [item["toxic"] for item in raw_test_data]

    test_soft_labels = None
    test_concept_scores = None
    if distill_test_data is not None:
        distill_map = {item["content"]: item for item in distill_test_data}
        test_soft_labels = []
        test_concept_scores = []
        for item in raw_test_data:
            d = distill_map.get(item["content"])
            if d is not None:
                test_soft_labels.append(d.get("soft_label"))
                test_concept_scores.append(d.get("concept_scores"))
            else:
                test_soft_labels.append(None)
                test_concept_scores.append(None)
        has_soft = all(s is not None for s in test_soft_labels)
        has_concept = all(c is not None for c in test_concept_scores)
        test_soft_labels = test_soft_labels if has_soft else None
        test_concept_scores = test_concept_scores if has_concept else None

    tokenizer = AutoTokenizer.from_pretrained(
        saved_config.models_path / saved_config.llm_model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_dataset = SCBMLLMDataset(
        test_texts, test_labels, test_soft_labels, test_concept_scores,
        tokenizer, saved_config.max_length,
    )
    test_loader = DataLoader(test_dataset, batch_size=saved_config.batch_size, shuffle=False)

    model = SCBMLLMModel(
        model_name=str(saved_config.models_path / saved_config.llm_model_name),
        num_concepts=saved_config.num_concepts,
        num_classes=saved_config.num_classes,
        concept_layer_idx=saved_config.concept_layer_idx,
        lora_r=saved_config.lora_r,
        lora_alpha=saved_config.lora_alpha,
        lora_dropout=saved_config.lora_dropout,
        concept_loss_weight=saved_config.concept_loss_weight,
        soft_label_weight=saved_config.soft_label_weight,
        soft_label_temperature=saved_config.soft_label_temperature,
        use_residual=saved_config.use_residual,
    )

    from peft import PeftModel
    model.llm = PeftModel.from_pretrained(model.llm, experiment_dir / "lora_weights")
    cbm_state = torch.load(experiment_dir / "cbm_weights.pth", map_location=device, weights_only=False)
    model.concept_bottleneck.load_state_dict(cbm_state["concept_bottleneck"])
    model.classifier.load_state_dict(cbm_state["classifier"])
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("   S-CBM-LLM Test Results")
    print("=" * 30)
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}")
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
        f.write("S-CBM-LLM Test Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Precision (Macro): {precision:.4f}\n")
        f.write(f"Recall (Macro):    {recall:.4f}\n")
        f.write(f"F1 Score (Macro):  {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

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
            "correct": bool(all_preds[i] == all_labels[i]),
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="S-CBM-LLM Training and Testing Pipeline")

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--lora_r', type=int, default=None)
    parser.add_argument('--concept_layer_idx', type=int, default=None)
    parser.add_argument('--no_soft_label', action='store_true', default=False)
    parser.add_argument('--no_concept_loss', action='store_true', default=False)
    parser.add_argument('--no_residual', action='store_true', default=False)
    parser.add_argument('--soft_label_weight', type=float, default=None)
    parser.add_argument('--concept_loss_weight', type=float, default=None)

    args = parser.parse_args()

    if args.mode in ['all', 'train']:
        config = SCBMLLMConfig()

        config.dataset_name = args.dataset_name

        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.lr is not None:
            config.lr = args.lr
        if args.patience is not None:
            config.patience = args.patience
        if args.lora_r is not None:
            config.lora_r = args.lora_r
        if args.concept_layer_idx is not None:
            config.concept_layer_idx = args.concept_layer_idx
        if args.no_soft_label:
            config.soft_label_weight = 0.0
        if args.no_concept_loss:
            config.concept_loss_weight = 0.0
        if args.no_residual:
            config.use_residual = False
        if args.soft_label_weight is not None:
            config.soft_label_weight = args.soft_label_weight
        if args.concept_loss_weight is not None:
            config.concept_loss_weight = args.concept_loss_weight

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        config_dict = {}
        for k, v in vars(type(config)).items():
            if not k.startswith("_"):
                config_dict[k] = str(v) if isinstance(v, Path) else v
        config_dict["dataset_name"] = config.dataset_name
        config_dict["experiment_path"] = str(config.experiment_path)

        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> Config saved to: {experiment_dir / 'config.json'}\n")

        print(f">>> Loading raw data...")
        raw_train_data = load_raw_data(config.dataset_name, "train", config.raw_data_path)
        raw_test_data = load_raw_data(config.dataset_name, "test", config.raw_data_path)

        print(f">>> Loading distill data...")
        distill_train_data = load_distill_data(config, config.dataset_name, "train")
        distill_test_data = load_distill_data(config, config.dataset_name, "test")

        train_texts = [item["content"] for item in raw_train_data]
        train_labels = [item["toxic"] for item in raw_train_data]

        train_soft_labels = None
        train_concept_scores = None
        if distill_train_data is not None and config.soft_label_weight > 0:
            distill_map = {item["content"]: item for item in distill_train_data}
            train_soft_labels = []
            train_concept_scores = []
            for item in raw_train_data:
                d = distill_map.get(item["content"])
                if d is not None:
                    train_soft_labels.append(d.get("soft_label"))
                    train_concept_scores.append(d.get("concept_scores"))
                else:
                    train_soft_labels.append(None)
                    train_concept_scores.append(None)
            has_soft = all(s is not None for s in train_soft_labels)
            has_concept = all(c is not None for c in train_concept_scores)
            train_soft_labels = train_soft_labels if has_soft else None
            train_concept_scores = train_concept_scores if has_concept else None

        test_texts = [item["content"] for item in raw_test_data]
        test_labels = [item["toxic"] for item in raw_test_data]

        test_soft_labels = None
        test_concept_scores = None
        if distill_test_data is not None:
            distill_map = {item["content"]: item for item in distill_test_data}
            test_soft_labels = []
            test_concept_scores = []
            for item in raw_test_data:
                d = distill_map.get(item["content"])
                if d is not None:
                    test_soft_labels.append(d.get("soft_label"))
                    test_concept_scores.append(d.get("concept_scores"))
                else:
                    test_soft_labels.append(None)
                    test_concept_scores.append(None)
            has_soft = all(s is not None for s in test_soft_labels)
            has_concept = all(c is not None for c in test_concept_scores)
            test_soft_labels = test_soft_labels if has_soft else None
            test_concept_scores = test_concept_scores if has_concept else None

        train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=0.1,
            stratify=train_labels,
            random_state=config.seed,
        )

        val_soft_labels = None
        val_concept_scores = None
        if train_soft_labels is not None:
            train_soft_labels_split, val_soft_labels = train_test_split(
                train_soft_labels,
                test_size=0.1,
                stratify=train_labels,
                random_state=config.seed,
            )
            train_soft_labels = train_soft_labels_split
        if train_concept_scores is not None:
            train_concept_scores_split, val_concept_scores = train_test_split(
                train_concept_scores,
                test_size=0.1,
                stratify=train_labels,
                random_state=config.seed,
            )
            train_concept_scores = train_concept_scores_split

        print(f">>> Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.models_path / config.llm_model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f">>> Creating datasets...")
        train_dataset = SCBMLLMDataset(
            train_texts_split, train_labels_split, train_soft_labels, train_concept_scores,
            tokenizer, config.max_length,
        )
        val_dataset = SCBMLLMDataset(
            val_texts, val_labels, val_soft_labels, val_concept_scores,
            tokenizer, config.max_length,
        )
        test_dataset = SCBMLLMDataset(
            test_texts, test_labels, test_soft_labels, test_concept_scores,
            tokenizer, config.max_length,
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        print(f">>> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        print(f">>> Initializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SCBMLLMModel(
            model_name=str(config.models_path / config.llm_model_name),
            num_concepts=config.num_concepts,
            num_classes=config.num_classes,
            concept_layer_idx=config.concept_layer_idx,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            concept_loss_weight=config.concept_loss_weight,
            soft_label_weight=config.soft_label_weight,
            soft_label_temperature=config.soft_label_temperature,
            use_residual=config.use_residual,
        )

        print(f">>> Training...")
        train(config, model, train_loader, val_loader, test_loader, device)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("Error: --timestamp is required for test mode")
            sys.exit(1)
        config = SCBMLLMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
