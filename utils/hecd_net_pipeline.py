"""HECD-Net 训练与测试流水线

整合训练和测试功能，支持：
- 从 TOXICN 原始 JSON 加载文本和多维标注
- 对比概念学习损失
- 多任务辅助头 (topic, expression, target)
- 平台-主题适配器
- 早停与最佳模型选择

使用示例:
    # 训练+测试
    python utils/hecd_net_pipeline.py --mode all

    # 仅测试
    python utils/hecd_net_pipeline.py --mode test --timestamp 20260528-120000
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
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.hecd_net_config import HECDNetConfig
from models.hecd_net import HECDNet, contrastive_concept_loss
from utils.seed import set_seed

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


# Platform / Topic 映射
PLATFORM_MAP = {"zhihu": 0, "tieba": 1}
TOPIC_MAP = {"race": 0, "gender": 1, "region": 2, "lgbt": 3}


class TOXICNDataset(Dataset):
    """TOXICN 数据集，返回原始文本和全部标注"""
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item["content"]
        toxic = item["toxic"]

        # Tokenize
        encoding = self.tokenizer(
            content,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Platform / Topic
        platform_id = PLATFORM_MAP.get(item.get("platform", "zhihu"), 0)
        topic_id = TOPIC_MAP.get(item.get("topic", "race"), 0)

        # 辅助标注
        topic_label = topic_id  # 主topic作为topic分类标签
        expression_label = item.get("expression", 0)
        target_vec = item.get("target", [0, 0, 0, 0, 0])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(toxic, dtype=torch.long),
            "platform_ids": torch.tensor(platform_id, dtype=torch.long),
            "topic_ids": torch.tensor(topic_id, dtype=torch.long),
            "topic_labels": torch.tensor(topic_label, dtype=torch.long),
            "expression_labels": torch.tensor(expression_label, dtype=torch.long),
            "target_labels": torch.tensor(target_vec, dtype=torch.float32),
        }


def load_raw_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def parse_args():
    parser = argparse.ArgumentParser(description="HECD-Net 训练与测试流水线")
    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None, help='测试模式时间戳')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--plm_lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--no_graph', action='store_true', help='禁用概念图')
    parser.add_argument('--no_adapter', action='store_true', help='禁用平台-主题适配器')
    parser.add_argument('--no_contrastive', action='store_true', help='禁用对比学习')
    parser.add_argument('--no_auxiliary', action='store_true', help='禁用辅助任务')
    parser.add_argument('--no_residual', action='store_true', help='禁用残差连接')
    return parser.parse_args()


def update_config(args):
    cfg = HECDNetConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.plm_lr is not None:
        cfg.plm_lr = args.plm_lr
    if args.patience is not None:
        cfg.patience = args.patience
    if args.no_graph:
        cfg.use_graph = False
    if args.no_adapter:
        cfg.use_adapter = False
    if args.no_contrastive:
        cfg.use_contrastive = False
    if args.no_auxiliary:
        cfg.use_auxiliary = False
    if args.no_residual:
        cfg.use_residual = False
    return cfg


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_contrastive = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            platform_ids=batch.get("platform_ids"),
            topic_ids=batch.get("topic_ids"),
            labels=batch["labels"],
            topic_labels=batch.get("topic_labels"),
            expression_labels=batch.get("expression_labels"),
            target_labels=batch.get("target_labels"),
        )

        loss = outputs["loss"]

        # 对比学习损失
        if config.use_contrastive:
            cl = contrastive_concept_loss(
                outputs["attn_weights"],
                batch["labels"],
                temperature=config.contrastive_temperature
            )
            loss = loss + config.contrastive_weight * cl
            total_contrastive += cl.item()

        # 反向传播
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        total_ce += outputs["loss"].item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            platform_ids=batch.get("platform_ids"),
            topic_ids=batch.get("topic_ids"),
            labels=batch["labels"],
        )

        total_loss += outputs["loss"].item()
        preds = torch.argmax(outputs["logits_toxic"], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1, p, r


def plot_metrics(config, epochs, train_losses, val_losses, val_f1s):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curve')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1s, label='Val Macro F1', color='tab:green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.set_title('Validation F1')
    ax2.grid(True, linestyle='--', alpha=0.6)

    save_path = config.experiment_path / "training_curves.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f">>> 训练曲线已保存至: {save_path}")


def train_model(config, train_dataset, val_dataset):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")

    # 本地模型路径
    plm_path = getattr(config, "plm_local_path", config.plm_name)
    print(f">>> 加载模型路径: {plm_path}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(plm_path)

    train_loader = DataLoader(
        TOXICNDataset(train_dataset, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        TOXICNDataset(val_dataset, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 模型
    model = HECDNet(
        plm_name=plm_path,
        num_concepts=config.num_concepts,
        num_classes=config.num_classes,
        num_topics=config.num_topics,
        num_expressions=config.num_expressions,
        num_targets=config.num_targets,
        num_platforms=config.num_platforms,
        plm_hidden_size=config.plm_hidden_size,
        concept_dim=config.concept_dim,
        num_gat_layers=config.num_gat_layers,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_num_heads=config.gat_num_heads,
        adapter_dim=config.adapter_dim,
        dropout=config.dropout,
        use_residual=config.use_residual,
        use_graph=config.use_graph,
        use_adapter=config.use_adapter,
        use_auxiliary=config.use_auxiliary,
        concept_graph_path=str(config.concept_graph_path) if config.use_graph else None,
    ).to(device)

    # 优化器：PLM 用较小学习率
    plm_params = list(model.plm.named_parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("plm.")
    ]

    optimizer = AdamW([
        {"params": [p for n, p in plm_params], "lr": config.plm_lr},
        {"params": other_params, "lr": config.lr}
    ], weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses, val_f1s = [], [], []
    epoch_list = []

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        val_loss, val_f1, val_p, val_r = evaluate_epoch(model, val_loader, device)

        epoch_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | P: {val_p:.4f} | R: {val_r:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
            print(f">>> 新最佳模型: Val F1 = {val_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f">>> 早停触发，已 {config.patience} 个 epoch 未提升")
            break

    if best_state is not None:
        torch.save(best_state, config.experiment_path / "best_model.pt")
        print(f">>> 最佳模型已保存 (Epoch {best_epoch}, F1={best_f1:.4f})")

    plot_metrics(config, epoch_list, train_losses, val_losses, val_f1s)
    return best_f1


@torch.no_grad()
def test_model(config, timestamp):
    experiment_dir = config.experiment_path.parent / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_cfg_dict = json.load(f)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    plm_path = saved_cfg_dict.get("plm_local_path", saved_cfg_dict.get("plm_name", config.plm_name))
    print(f">>> 加载模型路径: {plm_path}")
    tokenizer = AutoTokenizer.from_pretrained(plm_path)

    # 加载测试数据
    test_data = load_raw_data(config.raw_data_path / config.dataset_name / "test.json")
    test_loader = DataLoader(
        TOXICNDataset(test_data, tokenizer, saved_cfg_dict.get("max_length", 128)),
        batch_size=saved_cfg_dict.get("batch_size", 16),
        shuffle=False,
        collate_fn=collate_fn
    )

    # 恢复模型
    model = HECDNet(
        plm_name=plm_path,
        num_concepts=saved_cfg_dict.get("num_concepts", 56),
        num_classes=2,
        num_topics=saved_cfg_dict.get("num_topics", 4),
        num_expressions=saved_cfg_dict.get("num_expressions", 4),
        num_targets=saved_cfg_dict.get("num_targets", 5),
        num_platforms=saved_cfg_dict.get("num_platforms", 2),
        plm_hidden_size=saved_cfg_dict.get("plm_hidden_size", 768),
        concept_dim=saved_cfg_dict.get("concept_dim", 64),
        num_gat_layers=saved_cfg_dict.get("num_gat_layers", 2),
        gat_hidden_dim=saved_cfg_dict.get("gat_hidden_dim", 128),
        gat_num_heads=saved_cfg_dict.get("gat_num_heads", 4),
        adapter_dim=saved_cfg_dict.get("adapter_dim", 64),
        dropout=saved_cfg_dict.get("dropout", 0.3),
        use_residual=saved_cfg_dict.get("use_residual", True),
        use_graph=saved_cfg_dict.get("use_graph", True),
        use_adapter=saved_cfg_dict.get("use_adapter", True),
        use_auxiliary=saved_cfg_dict.get("use_auxiliary", True),
        concept_graph_path=str(config.concept_graph_path),
    ).to(device)

    model.load_state_dict(torch.load(experiment_dir / "best_model.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            platform_ids=batch.get("platform_ids"),
            topic_ids=batch.get("topic_ids"),
        )
        preds = torch.argmax(outputs["logits_toxic"], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 40)
    print("HECD-Net 测试集评估结果")
    print("=" * 40)
    print(f"Precision (Macro): {p:.4f}")
    print(f"Recall    (Macro): {r:.4f}")
    print(f"F1 Score  (Macro): {f1:.4f}")
    print("-" * 40)
    print(report)
    print("=" * 40)

    # 保存结果
    test_dir = experiment_dir / "test_results"
    test_dir.mkdir(parents=True, exist_ok=True)

    with open(test_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"precision_macro": round(p, 4), "recall_macro": round(r, 4), "f1_macro": round(f1, 4)},
                  f, indent=2, ensure_ascii=False)

    with open(test_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Precision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\n\n{report}")

    # 保存预测 + 解释
    predictions = []
    for i in range(len(all_preds)):
        predictions.append({
            "index": i,
            "true_label": int(all_labels[i]),
            "pred_label": int(all_preds[i]),
            "correct": bool(all_preds[i] == all_labels[i])
        })
    with open(test_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    config = update_config(args)

    if args.mode in ['all', 'train']:
        set_seed(config.seed)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存配置
        config_dict = config.to_dict()
        config_dict["timestamp"] = timestamp
        with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置已保存至: {experiment_dir / 'config.json'}")

        # 加载数据
        train_data = load_raw_data(config.raw_data_path / config.dataset_name / "train.json")
        train_split, val_split = train_test_split(
            train_data, test_size=0.1, random_state=config.seed,
            stratify=[d["toxic"] for d in train_data]
        )
        print(f">>> 训练集: {len(train_split)}, 验证集: {len(val_split)}")

        best_f1 = train_model(config, train_split, val_split)

        if args.mode == 'all':
            test_model(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        test_model(config, args.timestamp)


if __name__ == '__main__':
    main()
