import torch
import json
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
    test_x, test_y = load_data(mlp_config, "test_with_concepts(TOXICN)(Qwen2.5-1.5B-Instruct).json")
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    # 3. 初始化模型并加载权重
    model = MLP(in_features=test_x.shape[1])
    model_path = mlp_config.experiment_path / model_timestamp /"best_model.pth"

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
    print("=" * 30)


if __name__ == '__main__':
    evaluate_best_model("20260410-164833")