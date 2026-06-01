"""知识增强黑盒分类器训练与测试流水线（Phase 1 of KIPC-CBM）

整合训练和测试功能，实现训练完成后自动测试的流水线。
支持命令行参数配置，确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试
    python utils/blackbox_pipeline.py --mode all --dataset_name TOXICN --epochs 30 --patience 10 --use_dual_encoder --use_coded_terms --use_homophone --use_multitask

    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/blackbox_pipeline.py --mode test --timestamp 20260601-143000

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳 (如: 20260601-143000)

    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)

    知识注入开关:
        --use_dual_encoder  启用双路编码器 (默认: True)
        --no_dual_encoder   禁用双路编码器
        --use_coded_terms   启用编码术语注入 (默认: True)
        --no_coded_terms    禁用编码术语注入
        --use_homophone     启用谐音还原 (默认: True)
        --no_homophone      禁用谐音还原
        --use_multitask     启用多任务学习 (默认: True)
        --no_multitask      禁用多任务学习
        --use_platform      启用平台嵌入 (默认: True)
        --no_platform       禁用平台嵌入

    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (默认: False)

    训练超参数:
        --batch_size        批次大小 (默认: 16)
        --epochs            训练轮数 (默认: 30)
        --lr_backbone       骨干学习率 (默认: 2e-5)
        --lr_head           分类头学习率 (默认: 1e-3)
        --patience          早停耐心值 (默认: 10)
        --max_len           最大序列长度 (默认: 128)

    多任务权重:
        --lambda_topic      topic损失权重 (默认: 0.3)
        --lambda_expression expression损失权重 (默认: 0.3)

输出文件:
    实验目录结构 (experiments/<timestamp>/):
        ├── config.json              # 实验配置快照
        ├── best_model.pth           # 最佳模型权重
        ├── metrics.png              # 训练曲线图
        └── test_results/            # 测试结果目录 (仅 all/test 模式)
            ├── metrics.json         # 测试集评估指标
            ├── classification_report.txt  # 详细分类报告
            └── predictions.json     # 逐条预测结果
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
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.kipc_config import KIPCConfig
from models.blackbox_classifier import KnowledgeEnhancedClassifier
from utils.knowledge_utils import (
    HomophoneRestorer, CodedTermMatcher,
    get_platform_id, get_topic_id,
    build_default_homophone_map, build_default_coded_terms,
)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    parser = argparse.ArgumentParser(
        description="知识增强黑盒分类器训练流水线 (KIPC-CBM Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')

    parser.add_argument('--use_dual_encoder', action='store_true', default=True)
    parser.add_argument('--no_dual_encoder', action='store_false', dest='use_dual_encoder')
    parser.add_argument('--use_coded_terms', action='store_true', default=True)
    parser.add_argument('--no_coded_terms', action='store_false', dest='use_coded_terms')
    parser.add_argument('--use_homophone', action='store_true', default=True)
    parser.add_argument('--no_homophone', action='store_false', dest='use_homophone')
    parser.add_argument('--use_multitask', action='store_true', default=True)
    parser.add_argument('--no_multitask', action='store_false', dest='use_multitask')
    parser.add_argument('--use_platform', action='store_true', default=True)
    parser.add_argument('--no_platform', action='store_false', dest='use_platform')

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr_backbone', type=float, default=None)
    parser.add_argument('--lr_head', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--max_len', type=int, default=None)

    parser.add_argument('--lambda_topic', type=float, default=None)
    parser.add_argument('--lambda_expression', type=float, default=None)

    return parser.parse_args()


def update_config(args):
    config = KIPCConfig()
    config.dataset_name = args.dataset_name

    config.use_dual_encoder = args.use_dual_encoder
    config.use_coded_terms = args.use_coded_terms
    config.use_homophone = args.use_homophone
    config.use_multitask = args.use_multitask
    config.use_platform = args.use_platform

    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr_backbone is not None:
        config.lr_backbone = args.lr_backbone
    if args.lr_head is not None:
        config.lr_head = args.lr_head
    if args.patience is not None:
        config.patience = args.patience
    if args.max_len is not None:
        config.max_len = args.max_len
    if args.lambda_topic is not None:
        config.lambda_topic = args.lambda_topic
    if args.lambda_expression is not None:
        config.lambda_expression = args.lambda_expression

    return config


class ToxicDataset(Dataset):
    def __init__(self, data, tokenizer, config, homo_restorer, coded_matcher, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.homo_restorer = homo_restorer
        self.coded_matcher = coded_matcher
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        return self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item["content"]
        toxic = item["toxic"]
        platform = item.get("platform", "zhihu")
        topic_str = item.get("topic", "none")
        expression = item.get("expression", 0)
        target = item.get("target", [0, 0, 0, 0, 0])

        orig_enc = self._tokenize(content)

        if self.config.use_homophone and self.homo_restorer.homo_map:
            restored = self.homo_restorer.restore(content)
        else:
            restored = content

        if self.config.use_dual_encoder and restored != content:
            rest_enc = self._tokenize(restored)
        elif self.config.use_dual_encoder:
            rest_enc = orig_enc
        else:
            rest_enc = None

        result = {
            "input_ids_orig": orig_enc["input_ids"].squeeze(0),
            "attention_mask_orig": orig_enc["attention_mask"].squeeze(0),
            "toxic_label": torch.tensor(toxic, dtype=torch.long),
            "topic_label": torch.tensor(get_topic_id(topic_str), dtype=torch.long),
            "expression_label": torch.tensor(expression, dtype=torch.long),
            "target_label": torch.tensor(target, dtype=torch.float),
        }

        if rest_enc is not None:
            result["input_ids_rest"] = rest_enc["input_ids"].squeeze(0)
            result["attention_mask_rest"] = rest_enc["attention_mask"].squeeze(0)

        if self.config.use_coded_terms and self.coded_matcher.term_list:
            matched = self.coded_matcher.match(content)
            coded_multi_hot = torch.zeros(self.coded_matcher.num_terms, dtype=torch.float)
            for tid in matched:
                if tid < self.coded_matcher.num_terms:
                    coded_multi_hot[tid] = 1.0
            result["coded_multi_hot"] = coded_multi_hot
        else:
            result["coded_multi_hot"] = torch.zeros(max(self.coded_matcher.num_terms, 1), dtype=torch.float)

        if self.config.use_platform:
            result["platform_ids"] = torch.tensor(get_platform_id(platform), dtype=torch.long)
        else:
            result["platform_ids"] = torch.tensor(0, dtype=torch.long)

        return result


def load_data(config):
    data_path = config.raw_data_path / config.dataset_name
    with open(data_path / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_path / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return train_data, test_data


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    criterion_toxic = nn.CrossEntropyLoss()
    criterion_topic = nn.CrossEntropyLoss()
    criterion_expr = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids_orig = batch["input_ids_orig"].to(device)
        attention_mask_orig = batch["attention_mask_orig"].to(device)
        toxic_label = batch["toxic_label"].to(device)

        kwargs = {
            "input_ids_orig": input_ids_orig,
            "attention_mask_orig": attention_mask_orig,
        }

        if config.use_dual_encoder:
            kwargs["input_ids_rest"] = batch["input_ids_rest"].to(device)
            kwargs["attention_mask_rest"] = batch["attention_mask_rest"].to(device)

        if config.use_coded_terms:
            kwargs["coded_multi_hot"] = batch["coded_multi_hot"].to(device)

        if config.use_platform:
            kwargs["platform_ids"] = batch["platform_ids"].to(device)

        outputs = model(**kwargs)

        loss = criterion_toxic(outputs["toxic_logits"], toxic_label)

        if config.use_multitask:
            topic_label = batch["topic_label"].to(device)
            expr_label = batch["expression_label"].to(device)
            loss = loss + config.lambda_topic * criterion_topic(outputs["topic_logits"], topic_label)
            loss = loss + config.lambda_expression * criterion_expr(outputs["expression_logits"], expr_label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs["toxic_logits"], dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(toxic_label.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    criterion_toxic = nn.CrossEntropyLoss()
    criterion_topic = nn.CrossEntropyLoss()
    criterion_expr = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids_orig = batch["input_ids_orig"].to(device)
        attention_mask_orig = batch["attention_mask_orig"].to(device)
        toxic_label = batch["toxic_label"].to(device)

        kwargs = {
            "input_ids_orig": input_ids_orig,
            "attention_mask_orig": attention_mask_orig,
        }

        if config.use_dual_encoder:
            kwargs["input_ids_rest"] = batch["input_ids_rest"].to(device)
            kwargs["attention_mask_rest"] = batch["attention_mask_rest"].to(device)

        if config.use_coded_terms:
            kwargs["coded_multi_hot"] = batch["coded_multi_hot"].to(device)

        if config.use_platform:
            kwargs["platform_ids"] = batch["platform_ids"].to(device)

        outputs = model(**kwargs)

        loss = criterion_toxic(outputs["toxic_logits"], toxic_label)

        if config.use_multitask:
            topic_label = batch["topic_label"].to(device)
            expr_label = batch["expression_label"].to(device)
            loss = loss + config.lambda_topic * criterion_topic(outputs["topic_logits"], topic_label)
            loss = loss + config.lambda_expression * criterion_expr(outputs["expression_logits"], expr_label)

        total_loss += loss.item()
        preds = torch.argmax(outputs["toxic_logits"], dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(toxic_label.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1, precision, recall


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('Blackbox Training Metrics')
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


def train(config, train_dataset, val_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = KnowledgeEnhancedClassifier(
        roberta_path=config.roberta_path,
        num_coded_terms=config.num_coded_terms if config.use_coded_terms else 1,
        num_platforms=3 if config.use_platform else 1,
        coded_emb_dim=config.coded_term_emb_dim,
        platform_emb_dim=config.platform_emb_dim,
        use_dual_encoder=config.use_dual_encoder,
        use_coded_terms=config.use_coded_terms,
        use_homophone=config.use_homophone,
        use_multitask=config.use_multitask,
        use_platform=config.use_platform,
        num_topics=config.num_topics,
        num_expressions=config.num_expressions,
    ).to(device)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "roberta" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": config.lr_backbone},
        {"params": head_params, "lr": config.lr_head},
    ])

    total_steps = len(train_loader) * config.epochs
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[config.lr_backbone, config.lr_head],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
        three_phase=False,
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
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, config)

        val_loss, val_f1, val_p, val_r = evaluate_epoch(model, val_loader, device, config)
        test_loss, test_f1, _, _ = evaluate_epoch(model, test_loader, device, config)

        epoch_list.append(epoch + 1)
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)
        val_precision_history.append(val_p)
        val_recall_history.append(val_r)
        test_f1_history.append(test_f1)
        test_loss_history.append(test_loss)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss={train_loss:.4f}, Train F1={train_f1:.4f} | "
              f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}, Val P={val_p:.4f}, Val R={val_r:.4f} | "
              f"Test Loss={test_loss:.4f}, Test F1={test_f1:.4f}")

        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    from types import SimpleNamespace

    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(saved_config.roberta_path))

    homo_restorer = HomophoneRestorer(saved_config.homo_dict_path if hasattr(saved_config, 'homo_dict_path') else None)
    coded_matcher = CodedTermMatcher(
        saved_config.coded_terms_path if hasattr(saved_config, 'coded_terms_path') else None,
        max_terms=saved_config.num_coded_terms if saved_config.use_coded_terms else 200,
    )

    with open(Path(saved_config.raw_data_path) / saved_config.dataset_name / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_dataset = ToxicDataset(test_data, tokenizer, saved_config, homo_restorer, coded_matcher, saved_config.max_len)
    test_loader = DataLoader(test_dataset, batch_size=saved_config.batch_size, shuffle=False)

    model = KnowledgeEnhancedClassifier(
        roberta_path=Path(saved_config.roberta_path),
        num_coded_terms=saved_config.num_coded_terms if saved_config.use_coded_terms else 1,
        num_platforms=3 if saved_config.use_platform else 1,
        coded_emb_dim=saved_config.coded_term_emb_dim,
        platform_emb_dim=saved_config.platform_emb_dim,
        use_dual_encoder=saved_config.use_dual_encoder,
        use_coded_terms=saved_config.use_coded_terms,
        use_homophone=saved_config.use_homophone,
        use_multitask=saved_config.use_multitask,
        use_platform=saved_config.use_platform,
        num_topics=saved_config.num_topics,
        num_expressions=saved_config.num_expressions,
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids_orig = batch["input_ids_orig"].to(device)
            attention_mask_orig = batch["attention_mask_orig"].to(device)

            kwargs = {"input_ids_orig": input_ids_orig, "attention_mask_orig": attention_mask_orig}
            if saved_config.use_dual_encoder:
                kwargs["input_ids_rest"] = batch["input_ids_rest"].to(device)
                kwargs["attention_mask_rest"] = batch["attention_mask_rest"].to(device)
            if saved_config.use_coded_terms:
                kwargs["coded_multi_hot"] = batch["coded_multi_hot"].to(device)
            if saved_config.use_platform:
                kwargs["platform_ids"] = batch["platform_ids"].to(device)

            outputs = model(**kwargs)
            preds = torch.argmax(outputs["toxic_logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["toxic_label"].numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      黑盒分类器 测试集评估结果")
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
        f.write("黑盒分类器 测试集评估结果\n")
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
            "content": test_data[i]["content"],
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

    if args.mode in ['all', 'train']:
        config = update_config(args)

        if not config.roberta_path.exists():
            print(f"错误: RoBERTa 模型路径不存在: {config.roberta_path}")
            print("请先下载 chinese-roberta-wwm-ext 到 models/ 目录")
            sys.exit(1)

        if not config.homo_dict_path.exists():
            print(f">>> 谐音映射文件不存在，生成默认映射到: {config.homo_dict_path}")
            config.homo_dict_path.parent.mkdir(parents=True, exist_ok=True)
            build_default_homophone_map(config.homo_dict_path)

        if not config.coded_terms_path.exists():
            print(f">>> 编码术语词表不存在，生成默认词表到: {config.coded_terms_path}")
            config.coded_terms_path.parent.mkdir(parents=True, exist_ok=True)
            build_default_coded_terms(config.coded_terms_path)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        tokenizer = AutoTokenizer.from_pretrained(str(config.roberta_path))

        homo_restorer = HomophoneRestorer(str(config.homo_dict_path))
        coded_matcher = CodedTermMatcher(str(config.coded_terms_path), max_terms=config.num_coded_terms)
        config.num_coded_terms = coded_matcher.num_terms

        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")
        else:
            print(">>> 已禁用确定性模式, 结果将不可复现")

        print(f"\n{'=' * 60}")
        print("知识增强黑盒分类器 - 配置信息")
        print("=" * 60)
        print(f"数据集: {config.dataset_name}")
        print(f"双路编码器: {config.use_dual_encoder}")
        print(f"编码术语注入: {config.use_coded_terms} (词表大小: {coded_matcher.num_terms})")
        print(f"谐音还原: {config.use_homophone} (映射数: {len(homo_restorer.homo_map)})")
        print(f"多任务学习: {config.use_multitask}")
        print(f"平台嵌入: {config.use_platform}")
        print(f"学习率: backbone={config.lr_backbone}, head={config.lr_head}")
        print(f"多任务权重: lambda_topic={config.lambda_topic}, lambda_expression={config.lambda_expression}")
        print("=" * 60 + "\n")

        train_data, test_data = load_data(config)

        train_data_split, val_data_split = train_test_split(
            train_data, test_size=0.1,
            stratify=[d["toxic"] for d in train_data],
            random_state=config.seed,
        )

        train_dataset = ToxicDataset(train_data_split, tokenizer, config, homo_restorer, coded_matcher, config.max_len)
        val_dataset = ToxicDataset(val_data_split, tokenizer, config, homo_restorer, coded_matcher, config.max_len)
        test_dataset = ToxicDataset(test_data, tokenizer, config, homo_restorer, coded_matcher, config.max_len)

        print(f">>> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        metrics = train(config, train_dataset, val_dataset, test_dataset)
        plot_metrics(config, *metrics)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = KIPCConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
