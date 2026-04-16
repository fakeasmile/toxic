"""MLP训练与测试。

整合训练和测试功能,实现训练完成后自动测试的流水线。
支持命令行参数配置,确保训练-测试配置一致性。

使用示例:
    # 1. 训练+测试
    python utils/mlp_pipeline.py --mode all
    
    # 2. 仅训练模式
    python utils/mlp_pipeline.py --mode train
    
    # 3. 仅测试模式 (必须指定实验时间戳)
    python utils/mlp_pipeline.py --mode test --timestamp 20260415-085433

    # 4. 自定义数据集和超参数（完整命令）
    python utils/mlp_pipeline.py --mode all \\
        --dataset_name COLD \\
        --model_name Qwen2.5-1.5B-Instruct \\
        --batch_size 32 \\
        --epochs 100 \\
        --max_lr 5e-4 \\
        --pct_start 0.2 \\
        --div_factor 25 \\
        --final_div_factor 10000 \\
        --anneal_strategy cos \\
        --dropout_rate 0.4 \\
        --hidden_features 128 \\
        --use_deterministic \\
        --seed 42
    
    # 5. 启用确定性模式 (确保实验可复现)
    python utils/mlp_pipeline.py --mode all --use_deterministic --seed 42

命令行参数说明:
    运行模式:
        --mode              运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)
        --timestamp         测试模式时的实验时间戳 (如: 20260415-085433)
    
    数据集配置:
        --dataset_name      数据集名称 (TOXICN/COLD, 默认: TOXICN)
        --model_name        LLM模型名称 (默认: Qwen2.5-1.5B-Instruct)
    
    随机种子:
        --seed              随机种子 (默认: 1)
        --use_deterministic 启用确定性模式 (确保实验可复现，默认：False)
    
    训练超参数:
        --batch_size        批次大小 (默认: 16)
        --epochs            训练轮数 (默认: 200)
        --max_lr            峰值学习率 (默认: 1e-3)
        --pct_start         Warmup比例 (默认: 0.2)
        --div_factor        初始学习率除数 (默认: 25.0)
        --final_div_factor  最终学习率除数 (默认: 10000.0)
        --anneal_strategy   衰减策略: cos (余弦) 或 linear (线性), 默认: cos
    
    MLP模型结构参数:
        --dropout_rate      Dropout比率 (默认: 0.3)
        --hidden_features   隐藏层维度 (默认: 96)

参数优先级:
    - 训练模式: 命令行参数 > MLP_config.py（命令行参数覆盖MLP_config参数）
    - 测试模式: 强制使用实验目录的 config.json (忽略命令行超参数)

输出文件:
    实验目录结构 (experiments/<timestamp>/):
        ├── config.json              # 实验配置快照
        ├── best_model.pth           # 最佳模型权重
        ├── metrics.png              # 训练曲线图
        └── test_results/            # 测试结果目录 (仅 all/test 模式)
            ├── metrics.json         # 测试集评估指标
            ├── classification_report.txt  # 详细分类报告
            └── predictions.json     # 逐条预测结果

注意事项:
    1. 运行前需确保已生成概念向量文件 (使用scripts/generate_adjective_c_r.py)
    2. 测试模式必须指定有效的实验时间戳
"""

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

from configs.MLP_config import MLPConfig
from models.mlp import MLP

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MLP 训练与测试统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流水线 (训练+测试)
  python mlp_pipeline.py --mode all
  
  # 仅训练
  python mlp_pipeline.py --mode train
  
  # 仅测试
  python mlp_pipeline.py --mode test --timestamp 20260415-085433
  
  # 自定义超参数
  python mlp_pipeline.py --mode all --dataset_name COLD --epochs 100 --hidden_features 128
        """
    )

    # 运行模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'train', 'test'],
        default='all',
        help='运行模式: all (训练+测试, 默认), train (仅训练), test (仅测试)'
    )

    # 测试模式必需参数
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='测试模式时的实验时间戳 (如: 20260415-085433)'
    )

    # 数据集配置
    parser.add_argument('--dataset_name', type=str, default=None, help='数据集名称 (TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, default=None, help='LLM模型名称')

    # 随机种子
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--use_deterministic', action='store_true', default=False, help='启用确定性模式')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_lr', type=float, default=None, help='峰值学习率')
    parser.add_argument('--pct_start', type=float, default=None, help='Warmup比例')
    parser.add_argument('--div_factor', type=float, default=None, help='初始学习率除数')
    parser.add_argument('--final_div_factor', type=float, default=None, help='最终学习率除数')
    parser.add_argument('--anneal_strategy', type=str, default=None, help='衰减策略 (cos/linear)')

    # 模型结构参数
    parser.add_argument('--dropout_rate', type=float, default=None, help='Dropout比率')
    parser.add_argument('--hidden_features', type=int, default=None, help='隐藏层维度')

    return parser.parse_args()


def update_MLPConfig(args):
    """基于MLP_config参数，根据命令行参数更新配置对象

    优先级: 命令行参数 > MLPConfig默认值
    """
    mlp_config = MLPConfig()  # MLP_config.py中的配置对象

    # 用命令行参数更新配置对象
    if args.dataset_name is not None:
        mlp_config.dataset_name = args.dataset_name

    if args.model_name is not None:
        # 更新LLM模型
        mlp_config.model_name = args.model_name

    mlp_config.train_path = mlp_config.base_path / "data" / "raw" / mlp_config.dataset_name / "train.json"
    mlp_config.test_path = mlp_config.base_path / "data" / "raw" / mlp_config.dataset_name / "test.json"
    mlp_config.train_concept_path = mlp_config.processed_path / f"train_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json"
    mlp_config.test_concept_path = mlp_config.processed_path / f"test_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json"

    # 随机种子
    if args.seed is not None:
        mlp_config.seed = args.seed

    # 确定性模式
    if args.use_deterministic:  # store_true默认为False，只有显式传入才为True
        mlp_config.use_deterministic = True

    # 训练超参数
    if args.batch_size is not None:
        mlp_config.batch_size = args.batch_size
    if args.epochs is not None:
        mlp_config.epochs = args.epochs
    if args.max_lr is not None:
        mlp_config.max_lr = args.max_lr
    if args.pct_start is not None:
        mlp_config.pct_start = args.pct_start
    if args.div_factor is not None:
        mlp_config.div_factor = args.div_factor
    if args.final_div_factor is not None:
        mlp_config.final_div_factor = args.final_div_factor
    if args.anneal_strategy is not None:
        mlp_config.anneal_strategy = args.anneal_strategy

    # 模型结构参数
    if args.dropout_rate is not None:
        mlp_config.dropout_rate = args.dropout_rate
    if args.hidden_features is not None:
        mlp_config.hidden_features = args.hidden_features

    return mlp_config


def load_data(config, mode):
    """加载指定训练或测试的概念向量和标签。

    Args:
        config: 配置文件
        mode: train/test，区分加载训练或实验数据集

    Returns:
        tuple: (concepts, labels) 概念向量和标签张量
    """

    if mode == "train":
        concept_path = config.train_concept_path
        raw_data_path = config.train_path

    elif mode == "test":
        concept_path = config.test_concept_path
        raw_data_path = config.test_path
    else:
        raise ValueError("in load_data, mode must be 'train' or 'test'")

    # 加载概念向量文件
    with open(concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)

    # 加载原始数据集
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    concepts, labels = [], []
    if len(raw_data) != len(raw_concept_data):
        raise ValueError(
            f"数据长度不匹配:\n"
            f"  原始数据 ({mode}): {len(raw_data)} 条\n"
            f"  概念向量 ({mode}): {len(raw_concept_data)} 条\n"
            f"  文件路径:\n"
            f"    原始数据: {raw_data_path}\n"
            f"    概念向量: {concept_path}\n"
            f"  请重新运行 scripts/generate_adjective_c_r.py 生成概念向量"
        )

    # 提取形容词概念向量和对应的标签
    for i in range(0, len(raw_concept_data)):
        concepts.append(raw_concept_data[i]["concept"])
        labels.append(raw_data[i]["toxic"])

    return torch.tensor(concepts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def plot_metrics(config, epochs, losses, f1_scores, precisions, recalls):
    """绘制损失与各项评价指标曲线图。

    Args:
        config: MLPConfig 配置对象
        epochs: 轮次列表
        losses: 损失列表
        f1_scores: F1分数列表
        precisions: 精确率列表
        recalls: 召回率列表
    """
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制验证集损失 (左轴)
    lns1 = ax1.plot(epochs, losses, color='tab:red', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 绘制验证集 F1, Precision, Recall (右轴)
    lns2 = ax2.plot(epochs, f1_scores, color='tab:blue', label='Val F1')
    lns3 = ax2.plot(epochs, precisions, color='tab:green', linestyle='--', label='Val Precision')
    lns4 = ax2.plot(epochs, recalls, color='tab:orange', linestyle=':', label='Val Recall')

    ax2.set_ylabel('Score', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('MLP Training Metrics with Gating Mechanism')

    # 合并图例
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')

    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train(config, train_data):
    """训练MLP模型。

    Args:
        config: MLPConfig 配置对象
        train_data: 训练数据 (concepts, labels)
    """
    batch_size = config.batch_size
    epochs = config.epochs

    # 从训练集中按9:1比例划分验证集(分层抽样)
    full_train_x, full_train_y = train_data
    train_x_np, val_x_np, train_y_np, val_y_np = train_test_split(
        full_train_x.numpy(), full_train_y.numpy(),
        test_size=0.1,
        stratify=full_train_y.numpy(),
        random_state=config.seed
    )
    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    val_x = torch.tensor(val_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.long)
    val_y = torch.tensor(val_y_np, dtype=torch.long)

    print(f">>> 训练集样本数: {len(train_x)}, 验证集样本数: {len(val_x)}")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        in_features=train_x.shape[1],
        dropout_rate=config.dropout_rate,
        hidden_features=config.hidden_features
    ).to(device)

    # 损失函数,优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr / config.div_factor)

    max_lr = config.max_lr
    pct_start = config.pct_start
    total_steps = len(train_loader) * epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=config.anneal_strategy,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        three_phase=False
    )

    # 指标
    best_f1 = 0.0
    best_mlp_status_dict = None
    epoch_list, loss_history, f1_history, precision_history, recall_history = [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 验证集上验证
        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                v_loss = criterion(val_outputs, val_y)
                total_val_loss += v_loss.item()

                preds = torch.softmax(val_outputs, dim=1)
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # 计算指标
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        current_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        current_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # 保存结果
        epoch_list.append(epoch + 1)
        loss_history.append(avg_val_loss)
        f1_history.append(current_f1)
        precision_history.append(current_precision)
        recall_history.append(current_recall)

        print(f"Epoch {epoch + 1}: \n>>>Loss = {avg_val_loss:.4f}, \n>>>F1 = {current_f1:.4f}, \n>>>P = {current_precision:.4f}, "
              f"\n>>>R = {current_recall:.4f}")

        if current_f1 > best_f1:
            print(f">>> 发现更优模型 (F1: {current_f1:.4f}),提升:{current_f1 - best_f1:.4f}")
            best_f1 = current_f1
            best_mlp_status_dict = model.state_dict()

    # 保存模型
    if best_mlp_status_dict is not None:
        model_save_path = config.experiment_path / "best_model.pth"
        torch.save(best_mlp_status_dict, model_save_path)
        print(f">>> 最佳模型已保存至: {model_save_path}")

    # 调用绘图函数
    plot_metrics(config, epoch_list, loss_history, f1_history, precision_history, recall_history)


def load_config_only_from_experiment(timestamp, base_path):
    """从timestamp实验目录加载训练时保存的config.json配置。

    Args:
        timestamp: 实验时间戳
        base_path: 项目根路径

    Returns:
        tuple: (config_dict, experiment_dir) 配置字典和实验目录路径
    """
    experiment_dir = base_path / "experiments" / timestamp  # 某次实验的目录

    # 检查实验目录存在性
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"❌ 实验目录不存在: {experiment_dir}\n"
            f"   请检查时间戳是否正确,或先运行训练模式。"
        )

    # 检查配置文件存在性
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ 配置文件缺失: {config_path}\n"
            f"   实验目录不完整,无法进行测试。"
        )

    # 读取并验证配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"❌ 配置文件格式错误: {config_path}\n"
            f"   错误详情: {e}"
        )

    return saved_config, experiment_dir


def evaluate_best_model(base_path, timestamp):
    """评估最佳模型"""
    # 只从实验目录的config.json中加载参数配置
    saved_config, experiment_dir = load_config_only_from_experiment(timestamp, base_path)
    saved_config = SimpleNamespace(**saved_config)  # 将字典转换为支持.属性访问

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用设备: {device}")

    # 恢复训练时的随机种子设置，确保可复现性
    set_seed(saved_config)

    # 检查模型文件存在性
    model_path = experiment_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ 模型文件缺失: {model_path}"
        )

    # 加载测试数据
    print(">>> 正在加载测试数据...")
    test_x, test_y = load_data(saved_config, "test")
    batch_size = int(saved_config.batch_size)  # 确保类型为 int
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    # 从概念向量文件中加载原始文本内容（逐条保存预测结果）
    with open(saved_config.test_concept_path, "r", encoding="utf-8") as f:
        raw_concept_data = json.load(f)
    contents = [item["content"] for item in raw_concept_data]

    # 初始化模型 (使用训练时的配置)
    model = MLP(
        in_features=test_x.shape[1],
        dropout_rate=saved_config.dropout_rate,
        hidden_features=saved_config.hidden_features
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        print(f">>> 成功加载模型权重: {model_path}")
    except RuntimeError as e:
        raise RuntimeError(
            f"❌ 模型文件损坏或版本不兼容: {model_path}\n"
            f"   错误详情: {e}"
        )
    except KeyError as e:
        raise KeyError(
            f"❌ 模型状态字典键不匹配: {model_path}\n"
            f"   缺少键: {e}"
        )

    # 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    # 输出详细报告到控制台
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

    # 持久化保存结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # 保存评估指标 JSON
    metrics_dict = {
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
    }
    metrics_path = test_results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f">>> 评估指标已保存至: {metrics_path}")

    # 保存分类报告 TXT
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

    # 保存逐条预测结果 JSON
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


def load_dynamic_config(args):
    """
    1. 加载MLP_config，并依据命令行参数构建最终参数
    2. 生成时间戳实验目录，并更新experiment_path（在experiment_path基础上加时间戳后缀）
    3. 完整参数配置保存到config.json中
    """
    # 1. 加载MLP_config，并依据命令行参数构建最终参数
    updated_config = update_MLPConfig(args)  # 用解析的命令行参数更新MLPConfig()中的参数

    # 2. 生成时间戳并创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = updated_config.experiment_path / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    updated_config.experiment_path = experiment_dir

    # 打印关键配置参数到控制台
    print("\n" + "=" * 60)
    print("MLP 训练配置参数")
    print("=" * 60)
    print(f"实验目录: {updated_config.experiment_path}")
    print(f"时间戳: {timestamp}")
    print("\n--- 数据集配置 ---")
    print(f"数据集名称: {updated_config.dataset_name}")
    print(f"模型名称: {updated_config.model_name}")
    print(f"训练集路径: {updated_config.train_path}")
    print(f"测试集路径: {updated_config.test_path}")
    print("\n--- 训练超参数 ---")
    print(f"批次大小 (batch_size): {updated_config.batch_size}")
    print(f"训练轮数 (epochs): {updated_config.epochs}")
    print(f"峰值学习率 (max_lr): {updated_config.max_lr}")
    print(f"Warmup比例 (pct_start): {updated_config.pct_start}")
    print(f"初始学习率除数 (div_factor): {updated_config.div_factor}")
    print(f"最终学习率除数 (final_div_factor): {updated_config.final_div_factor}")
    print(f"衰减策略 (anneal_strategy): {updated_config.anneal_strategy}")
    print("\n--- 模型结构参数 ---")
    print(f"Dropout比率: {updated_config.dropout_rate}")
    print(f"隐藏层维度: {updated_config.hidden_features}")
    print("\n--- 随机种子配置 ---")
    print(f"随机种子 (seed): {updated_config.seed}")
    print(f"确定性模式: {updated_config.use_deterministic}")
    print("=" * 60 + "\n")

    # 3. 保存完整配置到config.json文件
    config_dict = {
        "timestamp": timestamp,
        "experiment_path": str(updated_config.experiment_path),
        "dataset_name": updated_config.dataset_name,
        "model_name": updated_config.model_name,
        "train_path": str(updated_config.train_path),
        "test_path": str(updated_config.test_path),
        "train_concept_path": str(updated_config.train_concept_path),
        "test_concept_path": str(updated_config.test_concept_path),
        "processed_path": str(updated_config.processed_path),
        "seed": updated_config.seed,
        "use_deterministic": updated_config.use_deterministic,
        "batch_size": updated_config.batch_size,
        "epochs": updated_config.epochs,
        "max_lr": updated_config.max_lr,
        "pct_start": updated_config.pct_start,
        "div_factor": updated_config.div_factor,
        "final_div_factor": updated_config.final_div_factor,
        "anneal_strategy": updated_config.anneal_strategy,
        "dropout_rate": updated_config.dropout_rate,
        "hidden_features": updated_config.hidden_features
    }

    config_file = updated_config.experiment_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f">>> 配置文件已保存至: {config_file}\n")

    return updated_config


def set_seed(config):
    # 确定性模式
    if config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(config)
        print(">>> 已启用确定性模式 (Reproducibility Enabled)")
    else:
        print(">>> 已禁用确定性模式 (Randomness Enabled),结果将不可复现")

def main():
    """
    参数加载逻辑：只要涉及到训练模型，基于MLP_config.py，使用命令行参数更新配置，并保存到config.json中
    只要涉及到评估（无论是all模式下还是test模式下），都从实验目录的config.json中加载参数配置
    """
    args = parse_args()  # 解析命令行参数

    # 训练
    if args.mode in ['all', 'train']:
        # 获取完整参数配置
        final_config = load_dynamic_config(args)

        # 是否启用确定性模式
        set_seed(final_config)

        print("\n>>> 开始训练流程...")
        train_data = load_data(final_config, "train")  # 加载训练数据
        train(final_config, train_data)
        print("\n>>> 训练流程完成!")

        # all模式下执行测试
        if args.mode == 'all':
            print("\n>>> 开始测试流程...")
            timestamp = final_config.experiment_path.name  # 获取实验时间戳
            evaluate_best_model(final_config.base_path, timestamp)
            print("\n>>> 测试执行完成!")

    # 测试模式:从实验目录读取配置，不接受任何命令行参数
    if args.mode == 'test':
        if args.timestamp is None:
            print("❌ 错误: 测试模式必须指定 --timestamp 参数")
            print("   示例: python mlp_pipeline.py --mode test --timestamp 20260415-085433")
            sys.exit(1)
        
        print(f"\n>>> 测试模式: 加载实验 {args.timestamp} 的配置")
        config = MLPConfig()  # 仅用于获取base_path
        evaluate_best_model(config.base_path, args.timestamp)


if __name__ == '__main__':
    main()
