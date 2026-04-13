import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from models.mlp import MLP
from configs.MLP_config import MLPConfig
from utils.mlp_train import load_data


def evaluate_best_model(model_timestamp: str):
    """

    :param model_timestamp: 模型参数的时间戳
    :return:
    """
    # 1. 初始化配置
    mlp_config = MLPConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    # 2. 加载测试数据
    print(">>> 正在加载测试数据...")
    test_x, test_y = load_data(mlp_config, f"test_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json")
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    # 加载原始文本内容（用于逐条保存预测结果）
    concept_path = mlp_config.processed_path / f"test_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json"
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)
    contents = [item["content"] for item in raw_concept_data]

    # 3. 初始化模型并加载权重
    model = MLP(in_features=test_x.shape[1])
    experiment_dir = mlp_config.experiment_path / model_timestamp
    model_path = experiment_dir / "best_model.pth"

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        print(f">>> 成功加载模型权重: {model_path}")
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 {model_path}，请先运行训练脚本。")
        return

    # 4. 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 5. 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 6. 输出详细报告到控制台
    print("\n" + "=" * 30)
    print("      MLP 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print("详细分类报告:")
    print(report)
    print("=" * 30)

    # 7. 持久化保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # 7.1 保存评估指标 JSON
    metrics_dict = {
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
    }
    metrics_path = test_results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f">>> 评估指标已保存至: {metrics_path}")

    # 7.2 保存分类报告 TXT
    report_path = test_results_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MLP 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")
    print(f">>> 分类报告已保存至: {report_path}")

    # 7.3 保存逐条预测结果 JSON
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
    predictions_path = test_results_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f">>> 逐条预测结果已保存至: {predictions_path}")


if __name__ == '__main__':
    evaluate_best_model("20260413-235111")
