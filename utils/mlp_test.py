import torch
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
from models.mlp import MLP
from configs.MLP_config import MLPConfig
from utils.mlp_train import load_data


def evaluate_best_model(model_timestamp: str, output_dir: str = None):
    """
    评估最佳模型并保存结果

    :param model_timestamp: 模型参数的时间戳
    :param output_dir: 结果保存目录 (可选，如果为 None 则使用实验目录)
    """
    # 1. 初始化配置
    mlp_config = MLPConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备：{device}")

    # 2. 加载测试数据
    print(">>> 正在加载测试数据...")
    test_x, test_y = load_data(mlp_config, f"test_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json")
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    # 3. 初始化模型并加载权重
    model = MLP(in_features=test_x.shape[1])
    model_path = mlp_config.experiment_path / model_timestamp /"best_model.pth"

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        print(f">>> 成功加载模型权重：{model_path}")
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {model_path}，请先运行训练脚本。")
        return

    # 4. 推理 (同时保存预测概率)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 有毒类别的概率

    # 5. 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 6. 输出详细报告
    print("\n" + "=" * 30)
    print("      MLP 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print("详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"]))
    print("-" * 30)
    print("混淆矩阵:")
    print(conf_matrix)
    print("=" * 30)

    # 7. 保存结果到指定目录
    if output_dir is None:
        save_dir = mlp_config.experiment_path / model_timestamp / "test_results"
    else:
        save_dir = Path(output_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> 正在保存结果到：{save_dir}")

    # 7.1 保存详细预测结果 (CSV 格式)
    csv_path = save_dir / "predictions.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("index,true_label,predicted_label,toxic_probability\n")
        for i, (true_lbl, pred_lbl, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
            f.write(f"{i},{true_lbl},{pred_lbl},{prob:.6f}\n")
    print(f">>> 预测详情已保存至：{csv_path}")

    # 7.2 保存评估指标 (JSON 格式)
    metrics = {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"], output_dict=True)
    }
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f">>> 评估指标已保存至：{metrics_path}")

    # 7.3 保存错误分析 (仅保存预测错误的样本)
    errors = []
    for i, (true_lbl, pred_lbl, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
        if true_lbl != pred_lbl:
            errors.append({
                "index": i,
                "true_label": int(true_lbl),
                "predicted_label": int(pred_lbl),
                "toxic_probability": float(prob)
            })
    
    if errors:
        error_path = save_dir / "errors.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f">>> 错误分析 ({len(errors)} 个样本) 已保存至：{error_path}")
    else:
        print(">>> 没有预测错误的样本！")

    print("=" * 30)
    print("所有结果保存完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP 模型评估脚本")
    parser.add_argument("--timestamp", type=str, required=True, help="模型时间戳 (例如：20260410-164833)")
    parser.add_argument("--output_dir", type=str, default=None, help="结果保存目录 (可选)")
    args = parser.parse_args()
    
    evaluate_best_model(args.timestamp, args.output_dir)
