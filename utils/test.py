import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader
from models.fc import FC
from models.bert import ModifiedBert
from utils.data_preprocess import ToxicDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def test(experiment_dir):
    """
    测试模型文件（best_model.pth + config.json）
    所有配置从 config.json 读取

    :param experiment_dir: 实验目录路径（如 experiments/20260410-184300）
    """
    print(f"{'='*20} 测试模型 {'='*20}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 验证实验目录是否存在
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    # 1. 从 config.json 加载所有配置
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"=>加载配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 提取所需配置
    bert_path = config["bert_path"]
    attack_stance_dict_path = config["attack_stance_dict_path"]
    dirty_lexicon_path = config["dirty_lexicon_path"]
    test_path = config["test_path"]
    max_len = config["max_len"]
    batch_size = config["batch_size"]
    freeze_bert_layers = config["freeze_bert_layers"]
    num_toxic_types = config["num_toxic_types"]
    dropout_rate = config["dropout_rate"]
    
    print(f"   数据集: {config['dataset_name']}")
    print(f"   max_len: {max_len}, batch_size: {batch_size}")
    print(f"   freeze_bert_layers: {freeze_bert_layers}, dropout: {dropout_rate}")

    # 2. 加载模型权重
    model_path = experiment_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"=>加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    bert_model = ModifiedBert(
        bert_path, 
        freeze_bert_layers, 
        num_toxic_types
    ).to(device)
    bert_model.load_state_dict(checkpoint['bert_state_dict'])
    
    fnn = FC(dropout_rate).to(device)
    fnn.load_state_dict(checkpoint['fnn_state_dict'])

    # 切换为评估模式呢
    bert_model.eval()
    fnn.eval()

    # 3. 加载测试集
    print(f"=>加载测试集")
    test_data = ToxicDataset(
        data_path=test_path,
        bert_path=bert_path,
        attack_stance_dict_path=attack_stance_dict_path,
        dirty_dict_path=dirty_lexicon_path,
        max_len=max_len,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 4. 在测试集上评估
    all_predicts = []
    all_labels = []

    print(f"=>开始评估")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", colour="GREEN"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            toxic_ids = batch["toxic_ids"].to(device)
            stance_ids = batch["stance_ids"].to(device)
            labels = batch["toxic"].to(device)

            # BERT特征提取
            feature_extraction = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                toxic_ids=toxic_ids,
                stance_ids=stance_ids,
            )
            
            # FNN分类
            logits = fnn(feature_extraction)
            pred = torch.softmax(logits, dim=1)
            max_idx = torch.argmax(pred, dim=1)

            all_predicts.extend(max_idx.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 5. 计算指标（使用 macro 平均）
    f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predicts, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predicts)

    print(f"\n{'='*20} 测试结果 {'='*20}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"{'='*50}\n")

    # 6. 保存测试结果
    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    test_metrics = {
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4)
        }
    }
    
    metrics_file = test_results_dir / "test_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    
    print(f">>> 测试结果已保存至: {metrics_file}")


if __name__ == '__main__':
    """
    加载新格式模型，在测试集上测试
    """
    from pathlib import Path
    
    # 获取项目根目录（utils 的父目录）
    project_root = Path(__file__).parent.parent
    
    # 指定测试的实验目录（使用绝对路径）
    experiment_dir = project_root / "experiments" / "20260410-233449"
    
    test(experiment_dir)
