"""TSCD训练与测试流水线

Two-Stage Concept Distillation: 两阶段概念蒸馏框架。
阶段一：PLM + SCL + FGM → 性能最大化
阶段二：概念蒸馏 → 事后可解释性

使用示例：
    # 完整两阶段训练
    python utils/tscd_pipeline.py --mode all --dataset_name TOXICN --plm_name chinese-roberta-wwm-ext

    # 仅阶段一
    python utils/tscd_pipeline.py --mode stage1 --dataset_name TOXICN --plm_name chinese-roberta-wwm-ext

    # 仅阶段二（需要先完成阶段一）
    python utils/tscd_pipeline.py --mode stage2 --stage1_timestamp 20260524-120000

    # 测试
    python utils/tscd_pipeline.py --mode test --timestamp 20260524-120000
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.tscd_config import TSCDConfig
from models.tscd_stage1 import Stage1Model, FGM, supervised_contrastive_loss
from models.tscd_stage2 import ConceptDistiller


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, concept_vectors=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.concept_vectors = concept_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {}
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.concept_vectors is not None:
            item["concept_target"] = torch.tensor(self.concept_vectors[idx], dtype=torch.float32)
        return item


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, concept_vectors):
        self.embeddings = embeddings
        self.concept_vectors = concept_vectors

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": torch.tensor(self.embeddings[idx], dtype=torch.float32),
            "concept_target": torch.tensor(self.concept_vectors[idx], dtype=torch.float32),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="TSCD训练与测试流水线")
    parser.add_argument('--mode', type=str, choices=['all', 'stage1', 'stage2', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--stage1_timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--plm_name', type=str, default=None)
    parser.add_argument('--num_concepts', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)
    parser.add_argument('--no_scl', action='store_true', default=False)
    parser.add_argument('--no_fgm', action='store_true', default=False)
    parser.add_argument('--s1_epochs', type=int, default=None)
    parser.add_argument('--s1_batch_size', type=int, default=None)
    parser.add_argument('--s1_lr', type=float, default=None)
    parser.add_argument('--s1_patience', type=int, default=None)
    parser.add_argument('--lambda_scl', type=float, default=None)
    parser.add_argument('--lambda_adv', type=float, default=None)
    parser.add_argument('--fgm_epsilon', type=float, default=None)
    parser.add_argument('--s2_epochs', type=int, default=None)
    parser.add_argument('--s2_lr', type=float, default=None)
    parser.add_argument('--s2_patience', type=int, default=None)
    parser.add_argument('--lambda_concept', type=float, default=None)
    parser.add_argument('--llm_model_name', type=str, default=None)
    return parser.parse_args()


def update_config(args):
    config = TSCDConfig()
    simple_map = {
        'dataset_name': 'dataset_name', 'plm_name': 'plm_name',
        'num_concepts': 'num_concepts', 'seed': 'seed',
        's1_epochs': 's1_epochs', 's1_batch_size': 's1_batch_size',
        's1_lr': 's1_lr', 's1_patience': 's1_patience',
        'lambda_scl': 'lambda_scl', 'lambda_adv': 'lambda_adv',
        'fgm_epsilon': 'fgm_epsilon',
        's2_epochs': 's2_epochs', 's2_lr': 's2_lr',
        's2_patience': 's2_patience', 'lambda_concept': 'lambda_concept',
        'llm_model_name': 'llm_model_name',
    }
    for arg_name, config_name in simple_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, config_name, val)
    if args.no_scl:
        config.use_scl = False
    if args.no_fgm:
        config.use_fgm = False
    if args.use_deterministic:
        config.use_deterministic = True

    config.train_concept_path = (config.processed_path / config.dataset_name
                                 / config.llm_model_name / "rcwn_concepts" / "concept_train.json")
    config.test_concept_path = (config.processed_path / config.dataset_name
                                / config.llm_model_name / "rcwn_concepts" / "concept_test.json")
    return config


def load_raw_data(raw_data_path, dataset_name, mode):
    data_path = raw_data_path / dataset_name / f"{mode}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["content"] for d in data]
    labels = [d["toxic"] for d in data]
    return texts, labels


def load_concept_vectors(concept_path):
    with open(concept_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d["concept"] for d in data]


def train_stage1(config, train_dataset, val_dataset, test_dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [阶段一] 使用设备: {device}")

    plm_path = config.models_path / config.plm_name
    if not plm_path.exists():
        raise ValueError(f"PLM path {plm_path} does not exist")

    model = Stage1Model(
        plm_name=str(plm_path),
        hidden_dim=config.s1_hidden_dim,
        dropout=config.s1_dropout,
    ).to(device)

    print(f">>> [阶段一] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f">>> [阶段一] 可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f">>> [阶段一] SCL: {config.use_scl}, FGM: {config.use_fgm}")

    train_loader = DataLoader(train_dataset, batch_size=config.s1_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.s1_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.s1_batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.s1_lr, weight_decay=config.s1_weight_decay)

    total_steps = len(train_loader) * config.s1_epochs
    warmup_steps = int(total_steps * config.s1_warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    fgm = FGM(model, epsilon=config.fgm_epsilon) if config.use_fgm else None

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(config.s1_epochs):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_scl = 0.0
        total_adv = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, proj, _ = model(input_ids, attention_mask)
            cls_loss = criterion(logits, labels)

            loss = cls_loss

            if config.use_scl and labels.sum() > 0 and (1 - labels).sum() > 0:
                scl_loss = supervised_contrastive_loss(proj, labels, config.scl_temperature)
                loss = loss + config.lambda_scl * scl_loss
                total_scl += scl_loss.item()
            else:
                scl_loss = torch.tensor(0.0)

            loss.backward()

            if fgm is not None:
                fgm.attack()
                adv_logits, _, _ = model(input_ids, attention_mask)
                adv_cls_loss = criterion(adv_logits, labels)
                adv_loss = config.lambda_adv * adv_cls_loss
                adv_loss.backward()
                fgm.restore()
                total_adv += adv_cls_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_cls += cls_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls / len(train_loader)
        avg_scl = total_scl / max(len(train_loader), 1)
        avg_adv = total_adv / max(len(train_loader), 1)

        model.eval()
        val_f1, val_p, val_r = _evaluate_model(model, val_loader, device)
        test_f1, _, _ = _evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (cls={avg_cls:.4f} scl={avg_scl:.4f} adv={avg_adv:.4f}) "
              f"| Val F1={val_f1:.4f} P={val_p:.4f} R={val_r:.4f} | Test F1={test_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> [阶段一] 发现更优模型 (Val F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.s1_patience:
            print(f">>> [阶段一] 早停: 连续 {config.s1_patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "stage1_best_model.pth")
        print(f">>> [阶段一] 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return model


def _evaluate_model(model, data_loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            logits, _, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return f1, p, r


def extract_embeddings(model, dataset, tokenizer, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, _, h = model(input_ids, attention_mask)
            all_embeddings.append(h.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def train_stage2(config, train_embeddings, train_concepts, val_embeddings, val_concepts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [阶段二] 使用设备: {device}")

    actual_num_concepts = len(train_concepts[0])
    if config.num_concepts != actual_num_concepts:
        print(f">>> 警告: num_concepts={config.num_concepts} 与概念向量维度={actual_num_concepts} 不匹配，自动修正")
        config.num_concepts = actual_num_concepts

    input_dim = train_embeddings.shape[1]
    distiller = ConceptDistiller(
        input_dim, config.num_concepts,
        hidden_dim=config.s2_concept_hidden_dim,
    ).to(device)

    print(f">>> [阶段二] 概念蒸馏器参数量: {sum(p.numel() for p in distiller.parameters()):,}")

    train_emb_dataset = EmbeddingDataset(train_embeddings, train_concepts)
    val_emb_dataset = EmbeddingDataset(val_embeddings, val_concepts)
    train_loader = DataLoader(train_emb_dataset, batch_size=config.s2_batch_size, shuffle=True)
    val_loader = DataLoader(val_emb_dataset, batch_size=config.s2_batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = AdamW(distiller.parameters(), lr=config.s2_lr, weight_decay=config.s2_weight_decay)

    best_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(config.s2_epochs):
        distiller.train()
        total_loss = 0.0

        for batch in train_loader:
            embeddings = batch["embedding"].to(device)
            targets = batch["concept_target"].to(device)

            preds = distiller(embeddings)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        distiller.eval()
        val_loss = 0.0
        concept_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embedding"].to(device)
                targets = batch["concept_target"].to(device)
                preds = distiller(embeddings)
                val_loss += criterion(preds, targets).item()
                concept_mae += (preds - targets).abs().mean().item()

        val_loss /= len(val_loader)
        concept_mae /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f} | Val MAE={concept_mae:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in distiller.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> [阶段二] 发现更优模型 (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.s2_patience:
            print(f">>> [阶段二] 早停: 连续 {config.s2_patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "stage2_best_model.pth")
        print(f">>> [阶段二] 最佳模型: Epoch {best_epoch}, Val Loss: {best_loss:.4f}")

    return distiller


def evaluate(config, timestamp=None):
    experiment_dir = config.experiment_path / timestamp if timestamp else config.experiment_path
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plm_path = Path(saved["plm_path"])
    s1_model = Stage1Model(
        plm_name=str(plm_path),
        hidden_dim=saved["s1_hidden_dim"],
        dropout=saved["s1_dropout"],
    )
    s1_model.load_state_dict(
        torch.load(experiment_dir / "stage1_best_model.pth", map_location=device, weights_only=False)
    )
    s1_model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(plm_path))

    test_texts, test_labels = load_raw_data(saved["raw_data_path"], saved["dataset_name"], "test")
    test_concepts = load_concept_vectors(saved["test_concept_path"])

    test_dataset = HateSpeechDataset(
        test_texts, test_labels, tokenizer=tokenizer,
        max_length=saved.get("s1_max_seq_length", 128)
    )
    test_loader = DataLoader(test_dataset, batch_size=saved["s1_batch_size"], shuffle=False)

    all_preds, all_labels = [], []
    all_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            logits, _, h = s1_model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_embeddings.append(h.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 40)
    print("   TSCD 阶段一 测试集评估结果")
    print("=" * 40)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 40)
    print(report)

    s2_model_path = experiment_dir / "stage2_best_model.pth"
    all_concept_scores = None
    if s2_model_path.exists():
        s2_model = ConceptDistiller(
            all_embeddings.shape[1], saved["num_concepts"],
            hidden_dim=saved.get("s2_concept_hidden_dim", 128),
        )
        s2_model.load_state_dict(
            torch.load(s2_model_path, map_location=device, weights_only=False)
        )
        s2_model.to(device).eval()

        with torch.no_grad():
            emb_tensor = torch.tensor(all_embeddings, dtype=torch.float32).to(device)
            all_concept_scores = s2_model(emb_tensor).cpu().numpy()

        concept_mae = np.abs(all_concept_scores - np.array(test_concepts)).mean()
        print(f"\n概念蒸馏 MAE: {concept_mae:.4f}")

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "f1_macro": round(float(f1), 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("TSCD 测试集评估结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 40 + "\n")

    label_names = ["Non-Toxic", "Toxic"]
    predictions = []
    for i in range(len(all_preds)):
        item = {
            "index": i,
            "content": test_texts[i],
            "true_label": int(all_labels[i]),
            "pred_label": int(all_preds[i]),
            "correct": bool(all_preds[i] == all_labels[i]),
        }
        if all_concept_scores is not None:
            item["concept_scores"] = [round(float(s), 4) for s in all_concept_scores[i]]
            item["concept_targets"] = [round(float(s), 4) for s in test_concepts[i]]
        predictions.append(item)

    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n>>> 测试结果已保存到: {test_results_dir}")


def main():
    args = parse_args()
    config = update_config(args)

    if args.mode in ['all', 'stage1', 'stage2']:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        plm_path = config.models_path / config.plm_name

        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "plm_name": config.plm_name,
            "plm_path": str(plm_path),
            "num_concepts": config.num_concepts,
            "llm_model_name": config.llm_model_name,
            "train_concept_path": str(config.train_concept_path),
            "test_concept_path": str(config.test_concept_path),
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "s1_batch_size": config.s1_batch_size,
            "s1_epochs": config.s1_epochs,
            "s1_lr": config.s1_lr,
            "s1_warmup_ratio": config.s1_warmup_ratio,
            "s1_weight_decay": config.s1_weight_decay,
            "s1_patience": config.s1_patience,
            "s1_dropout": config.s1_dropout,
            "s1_hidden_dim": config.s1_hidden_dim,
            "s1_max_seq_length": config.s1_max_seq_length,
            "use_scl": config.use_scl,
            "scl_temperature": config.scl_temperature,
            "lambda_scl": config.lambda_scl,
            "use_fgm": config.use_fgm,
            "fgm_epsilon": config.fgm_epsilon,
            "lambda_adv": config.lambda_adv,
            "s2_batch_size": config.s2_batch_size,
            "s2_epochs": config.s2_epochs,
            "s2_lr": config.s2_lr,
            "s2_weight_decay": config.s2_weight_decay,
            "s2_patience": config.s2_patience,
            "s2_concept_hidden_dim": config.s2_concept_hidden_dim,
            "s2_max_seq_length": config.s2_max_seq_length,
            "lambda_concept": config.lambda_concept,
            "raw_data_path": str(config.raw_data_path),
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("TSCD训练 - 配置信息")
        print("=" * 60)
        print(config)
        print("=" * 60 + "\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)

    # Stage 1
    if args.mode in ['all', 'stage1']:
        tokenizer = AutoTokenizer.from_pretrained(str(plm_path))

        train_texts, train_labels = load_raw_data(config.raw_data_path, config.dataset_name, "train")
        test_texts, test_labels = load_raw_data(config.raw_data_path, config.dataset_name, "test")

        train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1,
            stratify=train_labels, random_state=config.seed
        )

        train_dataset = HateSpeechDataset(
            train_texts_split, train_labels_split, tokenizer=tokenizer,
            max_length=config.s1_max_seq_length
        )
        val_dataset = HateSpeechDataset(
            val_texts, val_labels, tokenizer=tokenizer,
            max_length=config.s1_max_seq_length
        )
        test_dataset = HateSpeechDataset(
            test_texts, test_labels, tokenizer=tokenizer,
            max_length=config.s1_max_seq_length
        )

        print(f">>> [阶段一] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        s1_model = train_stage1(config, train_dataset, val_dataset, test_dataset, tokenizer)

    # Stage 2
    if args.mode in ['all', 'stage2']:
        if args.mode == 'stage2' and args.stage1_timestamp:
            s1_experiment_dir = config.experiment_path.parent / args.stage1_timestamp
        else:
            s1_experiment_dir = config.experiment_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s1_model_path = s1_experiment_dir / "stage1_best_model.pth"
        if not s1_model_path.exists():
            raise FileNotFoundError(f"阶段一模型不存在: {s1_model_path}")

        with open(s1_experiment_dir / "config.json", "r", encoding="utf-8") as f:
            s1_config = json.load(f)

        plm_path = Path(s1_config["plm_path"])
        tokenizer = AutoTokenizer.from_pretrained(str(plm_path))

        s1_model = Stage1Model(
            plm_name=str(plm_path),
            hidden_dim=s1_config["s1_hidden_dim"],
            dropout=s1_config["s1_dropout"],
        )
        s1_model.load_state_dict(
            torch.load(s1_model_path, map_location=device, weights_only=False)
        )
        s1_model.to(device).eval()

        train_texts, train_labels = load_raw_data(config.raw_data_path, config.dataset_name, "train")
        test_texts, test_labels = load_raw_data(config.raw_data_path, config.dataset_name, "test")
        train_concepts = load_concept_vectors(config.train_concept_path)
        test_concepts = load_concept_vectors(config.test_concept_path)

        train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1,
            stratify=train_labels, random_state=config.seed
        )
        train_concepts_split, val_concepts = train_test_split(
            train_concepts, test_size=0.1,
            stratify=train_labels, random_state=config.seed
        )

        train_s1_dataset = HateSpeechDataset(
            train_texts_split, train_labels_split, tokenizer=tokenizer,
            max_length=config.s1_max_seq_length
        )
        val_s1_dataset = HateSpeechDataset(
            val_texts, val_labels, tokenizer=tokenizer,
            max_length=config.s1_max_seq_length
        )

        print(">>> [阶段二] 提取训练集嵌入...")
        train_embeddings = extract_embeddings(s1_model, train_s1_dataset, tokenizer, device)
        print(">>> [阶段二] 提取验证集嵌入...")
        val_embeddings = extract_embeddings(s1_model, val_s1_dataset, tokenizer, device)

        print(f">>> [阶段二] 训练嵌入: {train_embeddings.shape}, 验证嵌入: {val_embeddings.shape}")

        distiller = train_stage2(
            config, train_embeddings, train_concepts_split,
            val_embeddings, val_concepts
        )

    # Test
    if args.mode == 'all':
        evaluate(config)
    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = TSCDConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
