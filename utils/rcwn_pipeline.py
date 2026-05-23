"""RCWN训练与测试流水线

Residual Concept Whitening Network的训练和评估流程。
支持两阶段训练：阶段一PLM fine-tuning，阶段二RCWN联合训练。

使用示例：
    # 完整训练（阶段一+阶段二）
    python utils/rcwn_pipeline.py --mode all --dataset_name TOXICN --plm_name chinese-roberta-wwm-ext

    # 仅测试
    python utils/rcwn_pipeline.py --mode test --timestamp 20260523-120000

命令行参数说明:
    --mode              运行模式: all (训练+测试), train (仅训练), test (仅测试)
    --timestamp         测试模式时的实验时间戳
    --dataset_name      数据集名称 (默认: TOXICN)
    --plm_name          PLM模型名称 (默认: chinese-roberta-wwm-ext)
    --num_concepts      概念数量 (默认: 40)
    --batch_size        批次大小 (默认: 32)
    --epochs            训练轮数 (默认: 30)
    --patience          早停耐心值 (默认: 5)
    --lambda_align      概念对齐损失权重 (默认: 1.0)
    --lambda_ortho      正交性损失权重 (默认: 0.1)
    --plm_frozen        是否冻结PLM参数 (默认: True)
    --seed              随机种子 (默认: 1)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.rcwn_config import RCWNConfig
from models.rcwn import RCWN


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, concept_vectors, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.concept_vectors = concept_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "concept_target": torch.tensor(self.concept_vectors[idx], dtype=torch.float32),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="RCWN训练与测试流水线")

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--plm_name', type=str, default=None)
    parser.add_argument('--num_concepts', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--lambda_align', type=float, default=None)
    parser.add_argument('--lambda_ortho', type=float, default=None)
    parser.add_argument('--freeze_plm', action='store_true', default=None)
    parser.add_argument('--unfreeze_plm', action='store_true', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--residual_hidden_dim', type=int, default=None)
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--warmup_ratio', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--llm_model_name', type=str, default=None)

    return parser.parse_args()


def update_config(args):
    config = RCWNConfig()

    arg_map = {
        'dataset_name': 'dataset_name',
        'plm_name': 'plm_name',
        'num_concepts': 'num_concepts',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'patience': 'patience',
        'lambda_align': 'lambda_align',
        'lambda_ortho': 'lambda_ortho',
        'seed': 'seed',
        'dropout': 'dropout',
        'residual_hidden_dim': 'residual_hidden_dim',
        'max_seq_length': 'max_seq_length',
        'weight_decay': 'weight_decay',
        'warmup_ratio': 'warmup_ratio',
        'llm_model_name': 'llm_model_name',
    }

    for arg_name, config_name in arg_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, config_name, val)

    if args.learning_rate is not None:
        config.max_lr = args.learning_rate

    if args.freeze_plm:
        config.plm_frozen = True
    elif args.unfreeze_plm:
        config.plm_frozen = False
    if args.use_deterministic:
        config.use_deterministic = True

    config.train_concept_path = (config.processed_path / config.dataset_name
                                 / config.llm_model_name / "rcwn_concepts" / "concept_train.json")
    config.test_concept_path = (config.processed_path / config.dataset_name
                                / config.llm_model_name / "rcwn_concepts" / "concept_test.json")

    return config


def load_raw_data(config, mode):
    data_path = config.raw_data_path / config.dataset_name / f"{mode}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["content"] for d in data]
    labels = [d["toxic"] for d in data]
    return texts, labels


def load_concept_vectors(concept_path):
    with open(concept_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vectors = [d["concept"] for d in data]
    return vectors


def train(config, train_dataset, val_dataset, test_dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    plm_path = config.models_path / config.plm_name
    if not plm_path.exists():
        raise ValueError(f"PLM path {plm_path} does not exist")

    model = RCWN(
        plm_name=str(plm_path),
        num_concepts=config.num_concepts,
        residual_hidden_dim=config.residual_hidden_dim,
        dropout=config.dropout,
        plm_frozen=config.plm_frozen,
    ).to(device)

    print(f">>> 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f">>> 可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    param_groups = [
        {"params": [p for p in model.cw_layer.parameters()], "lr": config.cw_lr},
        {"params": [model.concept_scale, model.concept_shift], "lr": config.cw_lr},
        {"params": [p for p in model.concept_head.parameters()], "lr": config.head_lr},
        {"params": [p for p in model.residual_head.parameters()], "lr": config.head_lr},
        {"params": [model.alpha], "lr": config.head_lr},
    ]
    if not config.plm_frozen:
        param_groups.append({
            "params": [p for p in model.plm.parameters() if p.requires_grad],
            "lr": config.plm_lr,
        })

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_align_loss = 0.0
        total_ortho_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            concept_targets = batch["concept_target"].to(device)

            optimizer.zero_grad()

            y, z_c, alpha, losses = model(
                input_ids, attention_mask, concept_targets
            )

            cls_loss = criterion(y, labels)
            align_loss = losses.get("align", torch.tensor(0.0, device=device))
            ortho_loss = losses["ortho"]

            loss = cls_loss + config.lambda_align * align_loss + config.lambda_ortho * ortho_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            model.cw_layer.orthogonalize()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_align_loss += align_loss.item()
            total_ortho_loss += ortho_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls_loss / len(train_loader)
        avg_align = total_align_loss / len(train_loader)
        avg_ortho = total_ortho_loss / len(train_loader)

        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                y, _, alpha, _ = model(input_ids, attention_mask)
                preds = torch.argmax(y, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        test_preds, test_labels_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                y, _, _, _ = model(input_ids, attention_mask)
                preds = torch.argmax(y, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(labels.numpy())

        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (cls={avg_cls:.4f} align={avg_align:.4f} "
              f"ortho={avg_ortho:.4f}) | Val F1={val_f1:.4f} P={val_p:.4f} R={val_r:.4f} | "
              f"Test F1={test_f1:.4f} | alpha={alpha.item():.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> 发现更优模型 (Val F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停: 验证集F1已连续 {config.patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return model


def evaluate(config, timestamp=None):
    experiment_dir = config.experiment_path / timestamp if timestamp else config.experiment_path
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config_dict = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plm_path = Path(saved_config_dict["plm_path"])
    model = RCWN(
        plm_name=str(plm_path),
        num_concepts=saved_config_dict["num_concepts"],
        residual_hidden_dim=saved_config_dict["residual_hidden_dim"],
        dropout=saved_config_dict["dropout"],
        plm_frozen=saved_config_dict["plm_frozen"],
    )

    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(plm_path)

    test_data_path = Path(saved_config_dict["raw_data_path"]) / saved_config_dict["dataset_name"] / "test.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_texts = [d["content"] for d in test_data]
    test_labels = [d["toxic"] for d in test_data]
    test_concepts = load_concept_vectors(saved_config_dict["test_concept_path"])

    test_dataset = HateSpeechDataset(
        test_texts, test_labels, test_concepts, tokenizer,
        max_length=saved_config_dict.get("max_seq_length", 128)
    )
    test_loader = DataLoader(test_dataset, batch_size=saved_config_dict["batch_size"], shuffle=False)

    all_preds, all_labels, all_concept_scores = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            y, z_c, alpha, _ = model(input_ids, attention_mask)
            preds = torch.argmax(y, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_concept_scores.extend(z_c.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      RCWN 测试集评估结果")
    print("=" * 30)
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
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("RCWN 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
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
            "concept_scores": [round(float(s), 4) for s in all_concept_scores[i]],
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f">>> 测试结果已保存到: {test_results_dir}")


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = update_config(args)

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
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "max_lr": config.max_lr,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "residual_hidden_dim": config.residual_hidden_dim,
            "patience": config.patience,
            "lambda_align": config.lambda_align,
            "lambda_ortho": config.lambda_ortho,
            "plm_frozen": config.plm_frozen,
            "max_seq_length": config.max_seq_length,
            "raw_data_path": str(config.raw_data_path),
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("RCWN训练 - 配置信息")
        print("=" * 60)
        print(config)
        print("=" * 60 + "\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")

        tokenizer = AutoTokenizer.from_pretrained(str(plm_path))

        train_texts, train_labels = load_raw_data(config, "train")
        test_texts, test_labels = load_raw_data(config, "test")

        train_concepts = load_concept_vectors(config.train_concept_path)
        test_concepts = load_concept_vectors(config.test_concept_path)

        actual_num_concepts = len(train_concepts[0])
        if config.num_concepts != actual_num_concepts:
            print(f">>> 警告: num_concepts={config.num_concepts} 与概念向量维度={actual_num_concepts} 不匹配，自动修正")
            config.num_concepts = actual_num_concepts

        train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1,
            stratify=train_labels, random_state=config.seed
        )
        train_concepts_split, val_concepts = train_test_split(
            train_concepts, test_size=0.1,
            stratify=train_labels, random_state=config.seed
        )

        train_dataset = HateSpeechDataset(
            train_texts_split, train_labels_split, train_concepts_split,
            tokenizer, config.max_seq_length
        )
        val_dataset = HateSpeechDataset(
            val_texts, val_labels, val_concepts,
            tokenizer, config.max_seq_length
        )
        test_dataset = HateSpeechDataset(
            test_texts, test_labels, test_concepts,
            tokenizer, config.max_seq_length
        )

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        model = train(config, train_dataset, val_dataset, test_dataset, tokenizer)

        if args.mode == 'all':
            evaluate(config)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = RCWNConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
