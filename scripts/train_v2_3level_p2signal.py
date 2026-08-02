"""v2 3level概念向量分类器 — 基于rel_enact稳健架构+P2反模式增量信号。

策略：保守增量，一次只加一个改动
  - 基底：rel_enact (264维 = relevance(132) + enactment(132))
          已验证：seed1~3 F1 = 0.8071 / 0.8092 / 0.8083，均值0.8082，最稳健
  - 增量：P2_raw (132维) — 显式"涉及但未实施"反模式信号
          → 提升Non-Toxic recall（当前0.79→目标0.81+）
  - 门控：TypeAugmentedGated风格（全矩阵门控 + 类型级统计特征注入分类层）
          → 不激进引入类型独立门控（已证402维过拟合）
  - 训练：完全标准配置（AdamW/OneCycleLR/batch16/dropout0.5/patience20）
          → 不使用Label Smoothing，不使用MixUp（已证0.05/0.2过强正则化）

特征维度 = relevance(132) + enactment(132) + P2_raw(132) = 396维
          + 5类型 × 3统计(rel_mean/enact_mean/p2_mean) = 15维
          = 411维 （类型统计仅15维增量，轻量不膨胀）

目标：Non-Toxic recall 0.79 → 0.81+，F1从0.8082→0.812+

使用示例：
    python scripts/train_v2_3level_p2signal.py --dataset_name TOXICN --model_name glm-4-9b-chat
    python scripts/train_v2_3level_p2signal.py --dataset_name TOXICN --model_name glm-4-9b-chat --n_trials 5
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


# =============================================================================
# P2SignalAugmentedGatedMLP
# =============================================================================
class P2SignalAugmentedGatedMLP(nn.Module):
    """基于rel_enact+P2反模式信号的增强门控MLP。

    与标准GatedMLP的关键区别：
      1. 输入三路：relevance + enactment + P2_raw
      2. 全矩阵门控（396维×396维，含P2信号的调节）
      3. 分类层额外拼接类型级三路mean统计（5类型×3统计=15维）
         → 鸟瞰视角：某类型整体"高P2低enact"则应判无毒

    结构：
      gate = sigmoid(W @ x)  ← 全矩阵门控（x含P2信息）
      x_gated = x * gate
      summary = [mean_t(rel), mean_t(enact), mean_t(p2) for t in 5types]  ← 15维
      h = FC(concat[x_gated, summary] → 96) → ReLU → Dropout → FC(2)

    Args:
        n_concepts: 132 for v2
        concept_types: 概念类型列表
        dropout_rate: 标准0.5
        hidden_features: 标准96
    """

    def __init__(self, n_concepts, concept_types, dropout_rate=0.5, hidden_features=96):
        super().__init__()
        self.n_concepts = n_concepts
        # 三路特征总维度：rel + enact + p2
        self.main_dim = n_concepts * 3  # 396

        type_names = sorted(set(concept_types))
        self.n_types = len(type_names)  # 5 for v2
        # 记录每个类型的索引
        self.type_indices = {}
        for t in type_names:
            self.type_indices[t] = [i for i, ct in enumerate(concept_types) if ct == t]

        # 每类型3统计(rel_mean, enact_mean, p2_mean) → 汇总维度
        n_summary = self.n_types * 3  # 15 for v2

        # 全矩阵门控（含P2信号调节）
        self.gate_layer = nn.Linear(self.main_dim, self.main_dim)

        # 分类层：主特征(396) + 汇总(15) = 411 → hidden → 2
        total_dim = self.main_dim + n_summary
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(total_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, 396+15=411) = [rel(132), enact(132), p2(132), 5类型×3统计(15)]
               注意：汇总特征放在末尾，门控只作用于前三路(396)
        """
        main_feats = x[:, 0:self.main_dim]       # (B, 396)
        summary_feats = x[:, self.main_dim:]       # (B, 15)

        # 全矩阵门控（含P2的信号，sigmoid可学习反模式抑制）
        gate = torch.sigmoid(self.gate_layer(main_feats))
        x_gated = main_feats * gate

        # 拼接门控主特征 + 类型汇总统计
        fused = torch.cat([x_gated, summary_feats], dim=1)
        fused = self.dropout(fused)
        h = self.relu(self.fc1(fused))
        h = self.dropout(h)
        return self.fc2(h)


# =============================================================================
# 特征提取
# =============================================================================
def extract_p2signal_features(data, concept_types):
    """提取 rel + enact + p2 + 类型级汇总特征。

    Returns (X, y)
    """
    n_samples = len(data)
    n_concepts = len(data[0]["concept"])
    type_names = sorted(set(concept_types))
    type_indices = {t: [i for i, ct in enumerate(concept_types) if ct == t] for t in type_names}

    relevances = np.zeros((n_samples, n_concepts))
    enactments = np.zeros((n_samples, n_concepts))
    p2_raw = np.zeros((n_samples, n_concepts))
    y = np.zeros(n_samples, dtype=int)

    for si, item in enumerate(data):
        lp = item["level_probs"]
        for ci in range(n_concepts):
            p1, p2, p3 = lp[ci][0], lp[ci][1], lp[ci][2]
            relevances[si, ci] = p2 + p3
            enactments[si, ci] = p3 / (p2 + p3 + 1e-8)
            p2_raw[si, ci] = p2
        y[si] = item["toxic"]

    # 类型级三路mean汇总: 15维
    n_types = len(type_names)
    summary = np.zeros((n_samples, n_types * 3))
    for si in range(n_samples):
        for t_idx, t in enumerate(type_names):
            idx = type_indices[t]
            base = t_idx * 3
            summary[si, base]     = np.mean(relevances[si, idx])
            summary[si, base + 1] = np.mean(enactments[si, idx])
            summary[si, base + 2] = np.mean(p2_raw[si, idx])

    X = np.concatenate([relevances, enactments, p2_raw, summary], axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =============================================================================
# train_one_trial
# =============================================================================
def train_one_trial(train_X, train_y, test_X, test_y, concept_types,
                    seed=1, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_concepts = 132
    in_features = train_X.shape[1]  # 411

    split = train_test_split(
        train_X.numpy(), train_y.numpy(),
        test_size=0.1, stratify=train_y.numpy(), random_state=seed,
    )
    tr_X = torch.tensor(split[0])
    va_X = torch.tensor(split[1])
    tr_y = torch.tensor(split[2])
    va_y = torch.tensor(split[3])

    train_loader = DataLoader(TensorDataset(tr_X, tr_y),
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_X, va_y),
                            batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y),
                             batch_size=config.batch_size, shuffle=False)

    model = P2SignalAugmentedGatedMLP(
        n_concepts=n_concepts, concept_types=concept_types,
        dropout_rate=config.dropout_rate, hidden_features=config.hidden_features,
    ).to(device)

    # 完全标准配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr / config.div_factor)
    total_steps = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr, total_steps=total_steps,
        pct_start=config.pct_start, anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor, final_div_factor=config.final_div_factor,
    )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0
    history = {'epochs': [], 'val_f1': [], 'test_f1': []}

    pbar = tqdm(range(config.epochs), desc=f"Training (seed={seed})")
    for epoch in pbar:
        model.train()
        for batch in train_loader:
            bx, by = batch
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # val
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch
                out = model(bx.to(device))
                vp.extend(torch.argmax(out, dim=1).cpu().numpy())
                vl.extend(by.numpy())
        val_f1 = f1_score(vl, vp, average='macro')

        # test观察（不参与模型选择）
        tp, tl = [], []
        with torch.no_grad():
            for batch in test_loader:
                bx, by = batch
                out = model(bx.to(device))
                tp.extend(torch.argmax(out, dim=1).cpu().numpy())
                tl.extend(by.numpy())
        test_f1 = f1_score(tl, tp, average='macro')

        history['epochs'].append(epoch + 1)
        history['val_f1'].append(val_f1)
        history['test_f1'].append(test_f1)
        pbar.set_postfix({'val_f1': f'{val_f1:.4f}', 'test_f1': f'{test_f1:.4f}', 'best': f'{best_f1:.4f}'})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config.patience:
            pbar.close()
            print(f">>> 早停: 连续{config.patience}个epoch未提升")
            break

    # 最终测试（仅一次）
    model.load_state_dict(best_state_dict)
    model.eval()
    ap, al, apr = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            bx, by = batch
            out = model(bx.to(device))
            probs = torch.softmax(out, dim=1)
            ap.extend(torch.argmax(out, dim=1).cpu().numpy())
            al.extend(by.numpy())
            apr.extend(probs.cpu().numpy())

    final_f1 = f1_score(al, ap, average='macro')
    final_p = precision_score(al, ap, average='macro', zero_division=0)
    final_r = recall_score(al, ap, average='macro', zero_division=0)
    report = classification_report(al, ap, target_names=["Non-Toxic", "Toxic"])
    print(f"\n>>> Trial seed={seed}: best_epoch={best_epoch}, val_f1={best_f1:.4f}, test_f1={final_f1:.4f}")
    print(f"    Precision={final_p:.4f}, Recall={final_r:.4f}")
    print(f"    Non-Toxic Recall = {recall_score(al, ap, labels=[0], average=None)[0]:.4f}")

    return {
        'seed': seed, 'best_epoch': best_epoch, 'best_val_f1': best_f1,
        'test_f1': final_f1, 'test_precision': final_p, 'test_recall': final_r,
        'classification_report': report,
        'predictions': {'labels': al, 'preds': ap, 'probs': apr},
        'history': history, 'model_state_dict': best_state_dict,
    }


def plot_metrics(experiment_dir, history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['epochs'], history['val_f1'], label='Val F1', color='tab:blue')
    ax.plot(history['epochs'], history['test_f1'], label='Test F1 (obs)',
            color='tab:red', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Macro F1')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('v2 3level P2Signal Training Metrics')
    plt.tight_layout()
    plt.savefig(experiment_dir / 'metrics.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="v2 3level 保守增量：rel_enact(264)+P2反模式(132)+轻量TypeAug汇总(15)=411维",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=1)
    args = parser.parse_args()

    config = MLPConfig()
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name

    concept_train_path = (config.processed_path / args.dataset_name / args.model_name
                          / f"concept_train_{args.model_name}_v2_3level.json")
    concept_test_path = (config.processed_path / args.dataset_name / args.model_name
                         / f"concept_test_{args.model_name}_v2_3level.json")
    if not concept_train_path.exists() or not concept_test_path.exists():
        raise FileNotFoundError(f"概念向量不存在: {concept_train_path}")
    print(f">>> 加载v2 3level概念向量: {concept_train_path.parent}")

    with open(concept_train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(concept_test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    n_concepts = len(train_data[0]["concept"])
    print(f">>> train={len(train_data)}, test={len(test_data)}, n_concepts={n_concepts}")

    type_mapping_path = config.raw_data_path / "adjective" / "toxic_adjectives_v2_types.json"
    with open(type_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    concept_types = [item["type"] for item in mapping]
    from collections import Counter
    type_dist = dict(Counter(concept_types))
    print(f">>> 类型分布: {type_dist}")

    train_X, train_y = extract_p2signal_features(train_data, concept_types)
    test_X, test_y = extract_p2signal_features(test_data, concept_types)
    in_features = train_X.shape[1]
    n_summary = len(type_dist) * 3
    print(f">>> 特征维度: {in_features} = relevance({n_concepts}) + enactment({n_concepts}) "
          f"+ P2_raw({n_concepts}) + 类型级汇总({n_summary})")
    print(f">>> 模型: P2SignalAugmentedGatedMLP (全矩阵门控 + 类型汇总注入分类层)")
    print(f">>> 训练: 标准配置 AdamW+OneCycleLR batch={config.batch_size} dropout={config.dropout_rate} patience={config.patience}")

    all_results = []
    for trial_idx in range(args.n_trials):
        seed = trial_idx + 1
        print("\n" + "=" * 55)
        print(f"  Trial {trial_idx + 1}/{args.n_trials} | seed={seed}")
        print("=" * 55)
        r = train_one_trial(train_X, train_y, test_X, test_y,
                            concept_types=concept_types, seed=seed, config=config)
        all_results.append(r)

    all_results.sort(key=lambda r: r["best_val_f1"], reverse=True)
    best_trial = all_results[0]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = config.experiment_path / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "timestamp": timestamp,
        "pipeline": "v2_3level_p2signal",
        "model_type": "P2SignalAugmentedGatedMLP",
        "n_features": in_features,
        "n_concepts": n_concepts,
        "type_distribution": type_dist,
        "dataset_name": args.dataset_name, "model_name": args.model_name,
        "seed": config.seed, "batch_size": config.batch_size,
        "epochs": config.epochs, "max_lr": config.max_lr,
        "dropout_rate": config.dropout_rate, "hidden_features": config.hidden_features,
        "patience": config.patience,
        "label_smoothing": 0.0, "mixup_alpha": 0.0,
        "n_trials": args.n_trials, "best_trial_seed": best_trial["seed"],
    }
    with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    torch.save(best_trial["model_state_dict"], experiment_dir / "best_model.pth")
    plot_metrics(experiment_dir, best_trial['history'])

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "pipeline": "v2_3level_p2signal",
        "model_type": "P2SignalAugmentedGatedMLP",
        "n_features": in_features,
        "best_trial_seed": best_trial['seed'],
        "best_epoch": best_trial['best_epoch'],
        "best_val_f1": round(best_trial['best_val_f1'], 4),
        "precision_macro": round(best_trial['test_precision'], 4),
        "recall_macro": round(best_trial['test_recall'], 4),
        "f1_macro": round(best_trial['test_f1'], 4),
    }
    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("v2 3level P2反模式 分类器评估结果\n")
        f.write("=" * 35 + "\n")
        f.write(f"模型: P2SignalAugmentedGatedMLP (rel+enact+P2 + 类型汇总)\n")
        f.write(f"特征维度: {in_features}\n")
        f.write(f"最佳Trial Seed: {best_trial['seed']}\n")
        f.write(f"最佳Epoch: {best_trial['best_epoch']}\n")
        f.write(f"Val F1: {best_trial['best_val_f1']:.4f}\n")
        f.write(f"Test F1: {best_trial['test_f1']:.4f}\n")
        f.write("-" * 35 + "\n")
        f.write(best_trial['classification_report'])

    preds_obj = best_trial['predictions']
    predictions = []
    for i in range(len(preds_obj['labels'])):
        predictions.append({
            "index": i,
            "true_label": int(preds_obj['labels'][i]),
            "pred_label": int(preds_obj['preds'][i]),
            "correct": bool(preds_obj['labels'][i] == preds_obj['preds'][i]),
            "probabilities": {
                "non_toxic": round(float(preds_obj['probs'][i][0]), 4),
                "toxic": round(float(preds_obj['probs'][i][1]), 4),
            },
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  v2 3level P2Signal 汇总 ({args.n_trials} trials)")
    print("=" * 60)
    print(f"{'Seed':<8}{'Best Ep':<10}{'Val F1':<10}{'Test F1':<10}")
    print("-" * 38)
    for r in all_results:
        print(f"{r['seed']:<8}{r['best_epoch']:<10}{r['best_val_f1']:<10.4f}{r['test_f1']:<10.4f}")
    test_f1s = [r['test_f1'] for r in all_results]
    print("-" * 38)
    print(f"  Test F1 mean = {np.mean(test_f1s):.4f}  std = {np.std(test_f1s):.4f}  best = {np.max(test_f1s):.4f}")
    print(f"  选中seed={best_trial['seed']}: F1={best_trial['test_f1']:.4f}")
    print(best_trial['classification_report'])
    print(f"\n>>> 结果: {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
