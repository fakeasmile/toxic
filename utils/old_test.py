"""
旧版本测试脚本（适配 checkpoint.pth 格式）
用于测试旧格式的模型文件（包含 config、hyperparameters、model 等嵌套结构）
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from models.fc import FC
from models.bert import ModifiedBert, Pure_Bert
from utils.data_preprocess import ToxicDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from configs.base_config import BaseConfig


def test_old(base_config, model_name):
    """
    测试旧格式的模型文件（checkpoint.pth）

    :param base_config: 基础配置
    :param model_name: checkpoint.pth 路径
    :return:
    """
    print(f"{'='*10}测试模型{'='*10}")
    # 在验证集上测试训练的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_name, weights_only=False)

    config = checkpoint["config"]

    best_search_hyperparameters = checkpoint["best_search_hyperparameters"]
    print(f"best_search_hyperparameters: {best_search_hyperparameters}")

    hyperparameters = checkpoint["hyperparameters"]

    max_len = hyperparameters["max_len"]
    batch_size = hyperparameters["batch_size"]

    # print(f"description: {checkpoint['extra']['descriptions']}")

    # 1. 加载测试集
    print(f"=>加载测试集")
    test_data = ToxicDataset(
        data_path=base_config.test_path,
        bert_path=base_config.bert_path,
        attack_stance_dict_path=base_config.attack_stance_dict_path,
        dirty_dict_path=base_config.dirty_lexicon_path,
        max_len=max_len,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 2. 加载微调的模型
    print(f"=>加载模型")
    bert_model = ModifiedBert(base_config.bert_path, base_config.freeze_bert_layers, base_config.num_toxic_types).to(device)
    # bert_model = Pure_Bert(base_config.bert_path, base_config.freeze_bert_layers).to(device)
    bert_model.load_state_dict(checkpoint["model"]["bert"])
    fnn = FC(base_config.dropout_rate).to(device)
    fnn.load_state_dict(checkpoint["model"]["fnn"])

    fnn.eval()
    bert_model.eval()

    all_predicts = []  # 存储所有预测标签
    all_labels = []  # 存储所有真实标签

    with torch.no_grad():
        for batch in tqdm(test_loader, colour="RED"):
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
            logits = fnn(feature_extraction)  # FNN

            pred = torch.softmax(logits, dim=1)
            max_idx = torch.argmax(pred, dim=1)

            # 收集所有的预测标签和真实标签
            all_predicts.extend(max_idx.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    """
    精确率：模型预测为有害的样本中，真正有害的比例
    召回率：实际有害的样本中，被模型成功识别为有害的比例
    准确率：所有样本中，预测正确的比例
    """

    f1 = f1_score(all_labels, all_predicts, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_predicts, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predicts, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predicts)
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")
    print(f"准确率（Accuracy）: {accuracy:.4f}")


if __name__ == '__main__':
    """
    加载旧格式模型，在测试集上测试
    """
    base_config = BaseConfig()

    # 指定要测试的 checkpoint.pth 路径
    model_path = base_config.experiment_path / "20260407-140745" / "checkpoint.pth"
    
    test_old(base_config, model_path)
