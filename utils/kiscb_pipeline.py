import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.KISCB_config import KISCBConfig
from models.kiscb import KISCB
from models.knowledge_preprocessor import CodedTermRecognizer
from utils.seed import set_reproducibility

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']

PLATFORM_MAP = {"zhihu": 0, "tieba": 1}
TOPIC_MAP = {"race": 0, "gender": 1, "region": 2, "lgbt": 3}


def parse_args():
    parser = argparse.ArgumentParser(
        description="KI-SCB 训练与测试统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--plm_lr', type=float, default=None)
    parser.add_argument('--max_lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)

    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--concept_emb_dim', type=int, default=None)

    parser.add_argument('--lambda_toxic', type=float, default=None)
    parser.add_argument('--lambda_target', type=float, default=None)
    parser.add_argument('--lambda_strategy', type=float, default=None)
    parser.add_argument('--lambda_intent', type=float, default=None)
    parser.add_argument('--lambda_tone', type=float, default=None)
    parser.add_argument('--lambda_consistency', type=float, default=None)

    return parser.parse_args()


def update_config(args):
    config = KISCBConfig()
    config.dataset_name = args.dataset_name

    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.plm_lr is not None:
        config.plm_lr = args.plm_lr
    if args.max_lr is not None:
        config.max_lr = args.max_lr
    if args.patience is not None:
        config.patience = args.patience

    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    if args.concept_emb_dim is not None:
        config.concept_emb_dim = args.concept_emb_dim

    if args.lambda_toxic is not None:
        config.lambda_toxic = args.lambda_toxic
    if args.lambda_target is not None:
        config.lambda_target = args.lambda_target
    if args.lambda_strategy is not None:
        config.lambda_strategy = args.lambda_strategy
    if args.lambda_intent is not None:
        config.lambda_intent = args.lambda_intent
    if args.lambda_tone is not None:
        config.lambda_tone = args.lambda_tone
    if args.lambda_consistency is not None:
        config.lambda_consistency = args.lambda_consistency

    return config


class TOXICNDataset(Dataset):
    def __init__(self, data, tokenizer, recognizer, max_length, platform_map, topic_map):
        self.samples = []

        contents = [item["content"] for item in data]
        coded_term_features = recognizer.forward(contents)

        for i, item in enumerate(data):
            encoding_orig = tokenizer(
                contents[i], max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )

            matched_terms = recognizer.match_terms(contents[i])
            if matched_terms:
                enhanced_text = contents[i] + " " + " ".join(matched_terms)
            else:
                enhanced_text = contents[i]
            encoding_enhanced = tokenizer(
                enhanced_text, max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )

            platform_id = platform_map.get(item.get("platform", "zhihu"), 0)
            topic_id = topic_map.get(item.get("topic", "race"), 0)

            toxic_label = item["toxic"]
            expression_label = item.get("expression", 0)
            target_label = item.get("target", [0] * 5)
            intent_label = item.get("intent", [0.0] * 5)
            tone_label = item.get("tone", 0)

            self.samples.append({
                "input_ids_orig": encoding_orig["input_ids"].squeeze(0),
                "attention_mask_orig": encoding_orig["attention_mask"].squeeze(0),
                "input_ids_enhanced": encoding_enhanced["input_ids"].squeeze(0),
                "attention_mask_enhanced": encoding_enhanced["attention_mask"].squeeze(0),
                "coded_term_features": coded_term_features[i],
                "platform_id": torch.tensor(platform_id, dtype=torch.long),
                "topic_id": torch.tensor(topic_id, dtype=torch.long),
                "toxic_label": torch.tensor(toxic_label, dtype=torch.long),
                "expression_label": torch.tensor(expression_label, dtype=torch.long),
                "target_label": torch.tensor(target_label, dtype=torch.float),
                "intent_label": torch.tensor(intent_label, dtype=torch.float),
                "tone_label": torch.tensor(tone_label, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["input_ids_orig"],
            s["attention_mask_orig"],
            s["input_ids_enhanced"],
            s["attention_mask_enhanced"],
            s["coded_term_features"],
            s["platform_id"],
            s["topic_id"],
            s["toxic_label"],
            s["expression_label"],
            s["target_label"],
            s["intent_label"],
            s["tone_label"],
        )


def load_data(config, mode, tokenizer, recognizer):
    data_path = Path(config.raw_data_path) / config.dataset_name / f"{mode}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TOXICNDataset(data, tokenizer, recognizer, config.max_length,
                         PLATFORM_MAP, TOPIC_MAP)


def compute_loss(config, model, batch, ce_criterion, bce_criterion, device):
    (input_ids_orig, attention_mask_orig, input_ids_enhanced, attention_mask_enhanced,
     coded_term_features, platform_ids, topic_ids, toxic_labels,
     expression_labels, target_labels, intent_labels, tone_labels) = batch

    input_ids_orig = input_ids_orig.to(device)
    attention_mask_orig = attention_mask_orig.to(device)
    input_ids_enhanced = input_ids_enhanced.to(device)
    attention_mask_enhanced = attention_mask_enhanced.to(device)
    coded_term_features = coded_term_features.to(device)
    platform_ids = platform_ids.to(device)
    topic_ids = topic_ids.to(device)
    toxic_labels = toxic_labels.to(device)
    expression_labels = expression_labels.to(device)
    target_labels = target_labels.to(device).float()
    intent_labels = intent_labels.to(device).float()
    tone_labels = tone_labels.to(device)

    logits, target_probs, strategy_logits, intent_probs, tone_logits = model(
        input_ids_orig, attention_mask_orig, input_ids_enhanced, attention_mask_enhanced,
        coded_term_features, platform_ids, topic_ids
    )

    loss_toxic = ce_criterion(logits, toxic_labels)
    loss_target = bce_criterion(target_probs, target_labels)
    loss_strategy = ce_criterion(strategy_logits, expression_labels)

    if config.lambda_intent > 0:
        loss_intent = bce_criterion(intent_probs, intent_labels)
    else:
        loss_intent = torch.tensor(0.0, device=device)

    if config.lambda_tone > 0:
        loss_tone = ce_criterion(tone_logits, tone_labels)
    else:
        loss_tone = torch.tensor(0.0, device=device)

    toxic_prob = torch.softmax(logits, dim=-1)[:, 1]
    consistency_loss = torch.relu(toxic_prob - intent_probs.max(dim=-1).values).mean()

    total_loss = (config.lambda_toxic * loss_toxic
                  + config.lambda_target * loss_target
                  + config.lambda_strategy * loss_strategy
                  + config.lambda_intent * loss_intent
                  + config.lambda_tone * loss_tone
                  + config.lambda_consistency * consistency_loss)

    return total_loss, logits, toxic_labels


def train(config, train_dataset, val_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    num_coded_terms = train_dataset[0][4].shape[0]
    model = KISCB(
        plm_name=config.plm_name,
        num_coded_terms=num_coded_terms,
        num_platforms=config.num_platforms,
        num_topics=config.num_topics,
        num_targets=config.num_targets,
        num_strategies=config.num_strategies,
        num_intents=config.num_intents,
        num_tones=config.num_tones,
        concept_emb_dim=config.concept_emb_dim,
        dropout_rate=config.dropout_rate,
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()

    plm_params = list(model.plm.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('plm.')]
    optimizer = torch.optim.AdamW([
        {'params': plm_params, 'lr': config.plm_lr},
        {'params': other_params, 'lr': config.max_lr},
    ], weight_decay=config.weight_decay)

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
    val_precision_history = []
    val_recall_history = []
    test_f1_history = []
    test_loss_history = []

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            total_loss, _, _ = compute_loss(config, model, batch, ce_criterion, bce_criterion, device)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss, logits, toxic_labels = compute_loss(config, model, batch, ce_criterion, bce_criterion, device)
                total_val_loss += val_loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(toxic_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                t_loss, logits, toxic_labels = compute_loss(config, model, batch, ce_criterion, bce_criterion, device)
                total_test_loss += t_loss.item()
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels_list.extend(toxic_labels.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

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

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history)


def evaluate(config, timestamp):
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    if saved_config.use_deterministic:
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(saved_config.plm_name)
    recognizer = CodedTermRecognizer(str(saved_config.toxic_lexicon_path))

    test_dataset = load_data(saved_config, "test", tokenizer, recognizer)
    test_loader = DataLoader(test_dataset, batch_size=int(saved_config.batch_size), shuffle=False)

    data_path = Path(saved_config.raw_data_path) / saved_config.dataset_name / "test.json"
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    contents = [item["content"] for item in raw_data]

    num_coded_terms = len(recognizer.terms)
    model = KISCB(
        plm_name=saved_config.plm_name,
        num_coded_terms=num_coded_terms,
        num_platforms=saved_config.num_platforms,
        num_topics=saved_config.num_topics,
        num_targets=saved_config.num_targets,
        num_strategies=saved_config.num_strategies,
        num_intents=saved_config.num_intents,
        num_tones=saved_config.num_tones,
        concept_emb_dim=saved_config.concept_emb_dim,
        dropout_rate=saved_config.dropout_rate,
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            (input_ids_orig, attention_mask_orig, input_ids_enhanced, attention_mask_enhanced,
             coded_term_features, platform_ids, topic_ids, toxic_labels, *_) = batch

            input_ids_orig = input_ids_orig.to(device)
            attention_mask_orig = attention_mask_orig.to(device)
            input_ids_enhanced = input_ids_enhanced.to(device)
            attention_mask_enhanced = attention_mask_enhanced.to(device)
            coded_term_features = coded_term_features.to(device)
            platform_ids = platform_ids.to(device)
            topic_ids = topic_ids.to(device)

            logits, _, _, _, _ = model(
                input_ids_orig, attention_mask_orig, input_ids_enhanced, attention_mask_enhanced,
                coded_term_features, platform_ids, topic_ids
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(toxic_labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      KI-SCB 测试集评估结果")
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
        f.write("KI-SCB 测试集评估结果\n")
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
            "content": contents[i],
            "true_label": int(all_labels[i]),
            "true_label_name": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_label_name": label_names[int(all_preds[i])],
            "correct": bool(all_preds[i] == all_labels[i])
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('KI-SCB Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

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


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = update_config(args)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        tokenizer = AutoTokenizer.from_pretrained(config.plm_name)
        recognizer = CodedTermRecognizer(str(config.toxic_lexicon_path))
        num_coded_terms = len(recognizer.terms)

        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "raw_data_path": str(config.raw_data_path),
            "processed_path": str(config.processed_path),
            "models_path": str(config.models_path),
            "homo_graph_path": str(config.homo_graph_path),
            "toxic_lexicon_path": str(config.toxic_lexicon_path),
            "coded_terms_path": str(config.coded_terms_path),
            "dataset_name": config.dataset_name,
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "plm_name": config.plm_name,
            "max_length": config.max_length,
            "num_platforms": config.num_platforms,
            "num_topics": config.num_topics,
            "num_targets": config.num_targets,
            "num_strategies": config.num_strategies,
            "num_intents": config.num_intents,
            "num_tones": config.num_tones,
            "num_coded_terms": num_coded_terms,
            "concept_emb_dim": config.concept_emb_dim,
            "dropout_rate": config.dropout_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "plm_lr": config.plm_lr,
            "max_lr": config.max_lr,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "patience": config.patience,
            "lambda_toxic": config.lambda_toxic,
            "lambda_target": config.lambda_target,
            "lambda_strategy": config.lambda_strategy,
            "lambda_intent": config.lambda_intent,
            "lambda_tone": config.lambda_tone,
            "lambda_consistency": config.lambda_consistency,
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        if config.use_deterministic:
            set_reproducibility(config)
            print(">>> 已启用确定性模式 (Reproducibility Enabled)")
        else:
            print(">>> 已禁用确定性模式 (Randomness Enabled), 结果将不可复现")

        train_dataset = load_data(config, "train", tokenizer, recognizer)
        test_dataset = load_data(config, "test", tokenizer, recognizer)

        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=0.1,
            stratify=[train_dataset.samples[i]["toxic_label"].item() for i in range(len(train_dataset))],
            random_state=config.seed
        )
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset_split = Subset(train_dataset, train_indices)

        print(f">>> 训练集: {len(train_dataset_split)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        metrics = train(config, train_dataset_split, val_dataset, test_dataset)
        plot_metrics(config, *metrics)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = KISCBConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
