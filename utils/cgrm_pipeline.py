"""CGRM (Concept-Guided Rationale Model) 训练与测试。

整合训练和测试功能,实现训练完成后自动测试的流水线。
支持命令行参数配置,确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试
    python utils/cgrm_pipeline.py --mode all --dataset_name TOXICN
    
    # 2. 仅测试模式 (必须指定实验时间戳)
    python utils/cgrm_pipeline.py --mode test --timestamp 20260530-120000

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳 (如: 20260530-120000)
    
    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)
    
    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (默认: False)
    
    训练超参数:
        --batch_size        批次大小 (默认: 16)
        --epochs            训练轮数 (默认: 30)
        --plm_lr            PLM学习率 (默认: 2e-5)
        --max_lr            峰值学习率 (默认: 1e-4)
        --patience          早停耐心值 (默认: 5)
    
    CGRM模型结构参数:
        --concept_dim       概念嵌入维度 (默认: 64)
        --num_heads         注意力头数 (默认: 4)
        --dropout_rate      Dropout比率 (默认: 0.3)

参数优先级:
    - 训练模式: 命令行参数 > CGRM_config.py
    - 测试模式: 强制使用实验目录的 config.json

输出文件:
    实验目录结构 (experiments/<timestamp>/):
        ├── config.json              # 实验配置快照
        ├── best_model.pth           # 最佳模型权重
        ├── metrics.png              # 训练曲线图
        └── test_results/            # 测试结果目录
            ├── metrics.json         # 测试集评估指标
            ├── classification_report.txt
            └── predictions.json     # 逐条预测结果
"""

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.CGRM_config import CGRMConfig
from models.cgrm import CGRM

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CGRM 训练与测试统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 运行模式
    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'], default='all',
                        help='运行模式: all (训练+测试), train (仅训练), test (仅测试)')

    # 测试模式必需参数
    parser.add_argument('--timestamp', type=str, default=None,
                        help='测试模式时的实验时间戳')

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default='TOXICN', help='数据集名称 (TOXICN/COLD)')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False, help='启用确定性模式')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--plm_lr', type=float, default=None, help='PLM学习率')
    parser.add_argument('--max_lr', type=float, default=None, help='峰值学习率')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值')

    # 模型结构参数
    parser.add_argument('--concept_dim', type=int, default=None, help='概念嵌入维度')
    parser.add_argument('--num_heads', type=int, default=None, help='注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=None, help='Dropout比率')

    return parser.parse_args()


def update_CGRMConfig(args):
    """基于CGRM_config参数，根据命令行参数更新配置对象"""
    config = CGRMConfig()

    # 数据集配置
    config.dataset_name = args.dataset_name

    # 随机种子
    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    # 训练超参数
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

    # 模型结构参数
    if args.concept_dim is not None:
        config.concept_dim = args.concept_dim
    if args.num_heads is not None:
        config.num_heads = args.num_heads
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate

    return config


class TextDataset(Dataset):
    """文本数据集，返回 tokenized 输入和标签"""
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        content = item["content"]
        if not isinstance(content, str) or len(content.strip()) == 0:
            content = "无内容"
        toxic = item["toxic"]

        encoding = self.tokenizer(
            content,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(toxic, dtype=torch.long),
        }


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def load_raw_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    """绘制训练曲线图（上下双子图）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('CGRM Training Metrics')
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


def train(config, train_data, val_data, test_data):
    """训练CGRM模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    plm_path = config.plm_local_path
    print(f">>> 加载模型路径: {plm_path}")

    tokenizer = AutoTokenizer.from_pretrained(plm_path)

    train_loader = DataLoader(
        TextDataset(train_data, tokenizer, config.max_length),
        batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        TextDataset(val_data, tokenizer, config.max_length),
        batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        TextDataset(test_data, tokenizer, config.max_length),
        batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 初始化模型
    model = CGRM(
        plm_name=plm_path,
        num_concepts=config.num_concepts,
        concept_dim=config.concept_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout_rate,
    ).to(device)

    # 优化器: PLM 用较小学习率
    plm_params = [p for n, p in model.named_parameters() if n.startswith("plm.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("plm.")]

    optimizer = optim.AdamW([
        {"params": plm_params, "lr": config.plm_lr},
        {"params": other_params, "lr": config.max_lr / config.div_factor},
    ])

    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.plm_lr, config.max_lr],
        total_steps=total_steps,
        pct_start=config.pct_start,
        anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        three_phase=False
    )

    criterion = nn.CrossEntropyLoss()

    # 训练状态
    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    # 指标记录
    epoch_list = []
    val_loss_history = []
    val_f1_history = []
    val_precision_history = []
    val_recall_history = []
    test_f1_history = []
    test_loss_history = []

    for epoch in range(config.epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in train_pbar:
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            outputs = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                outputs = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                total_val_loss += outputs["loss"].item()
                val_preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy())
                val_labels_list.extend(batch["labels"].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        # ========== 测试集评估（仅观察）==========
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                outputs = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                total_test_loss += outputs["loss"].item()
                test_preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy())
                test_labels_list.extend(batch["labels"].cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        # 记录指标
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

        # ========== 最佳模型选择与早停 ==========
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

    # 保存最佳模型
    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history)


def evaluate(config, timestamp):
    """评估指定实验的最佳模型在测试集上的表现"""
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config = SimpleNamespace(**json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plm_path = saved_config.plm_local_path
    print(f">>> 加载模型路径: {plm_path}")
    tokenizer = AutoTokenizer.from_pretrained(plm_path)

    # 加载测试数据
    test_data = load_raw_data(config.raw_data_path / saved_config.dataset_name / "test.json")
    test_loader = DataLoader(
        TextDataset(test_data, tokenizer, saved_config.max_length),
        batch_size=int(saved_config.batch_size), shuffle=False, collate_fn=collate_fn
    )

    # 加载最佳模型
    model = CGRM(
        plm_name=plm_path,
        num_concepts=int(saved_config.num_concepts),
        concept_dim=int(saved_config.concept_dim),
        hidden_dim=int(saved_config.hidden_dim),
        num_heads=int(saved_config.num_heads),
        dropout=float(saved_config.dropout_rate),
    )
    model.load_state_dict(torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False))
    model.to(device).eval()

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 输出到控制台
    print("\n" + "=" * 30)
    print("      CGRM 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    # 持久化保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("CGRM 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    # 保存逐条预测结果
    label_names = ["Non-Toxic", "Toxic"]
    predictions = []
    for i in range(len(all_preds)):
        predictions.append({
            "index": i,
            "true_label": int(all_labels[i]),
            "true_label_name": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_label_name": label_names[int(all_preds[i])],
            "correct": bool(all_preds[i] == all_labels[i])
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = update_CGRMConfig(args)

        # 生成时间戳并创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        # 保存完整配置到config.json
        config_dict = {
            "timestamp": timestamp,
            "experiment_path": str(config.experiment_path),
            "dataset_name": config.dataset_name,
            "plm_name": config.plm_name,
            "plm_local_path": config.plm_local_path,
            "max_length": config.max_length,
            "seed": config.seed,
            "use_deterministic": config.use_deterministic,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "plm_lr": config.plm_lr,
            "max_lr": config.max_lr,
            "pct_start": config.pct_start,
            "div_factor": config.div_factor,
            "final_div_factor": config.final_div_factor,
            "anneal_strategy": config.anneal_strategy,
            "patience": config.patience,
            "num_concepts": config.num_concepts,
            "concept_dim": config.concept_dim,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "dropout_rate": config.dropout_rate,
        }
        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        # 随机种子
        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式")
        else:
            print(">>> 已禁用确定性模式, 结果将不可复现")

        # 加载数据
        train_data = load_raw_data(config.raw_data_path / config.dataset_name / "train.json")
        test_data = load_raw_data(config.raw_data_path / config.dataset_name / "test.json")

        # 从训练集中按9:1比例划分验证集（分层抽样）
        train_split, val_split = train_test_split(
            train_data, test_size=0.1,
            stratify=[d["toxic"] for d in train_data],
            random_state=config.seed
        )

        print(f">>> 训练集: {len(train_split)}, 验证集: {len(val_split)}, 测试集: {len(test_data)}")

        # 训练并获取指标
        metrics = train(config, train_split, val_split, test_data)

        # 绘制训练曲线图
        plot_metrics(config, *metrics)

        # all模式下执行测试
        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = CGRMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
