import torch
import json
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

from models.fc import FC
from utils.data_preprocess import ToxicDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from models.bert import ModifiedBert, Pure_Bert
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from configs.base_config import BaseConfig

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


def init():
    """
    初始化配置、创建实验目录、打印参数、保存配置文件
    """
    # ------ 初始化配置 -----
    base_config = BaseConfig()

    # 生成时间戳并创建实验结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # 时间戳
    experiment_dir = base_config.experiment_path / timestamp  # 实验结果保存目录
    experiment_dir.mkdir(parents=True, exist_ok=True)
    base_config.experiment_path = experiment_dir

    # 检查是否启用确定性模式
    if base_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(base_config)
        print(">>> 已启用确定性模式 (Reproducibility Enabled)")
    else:
        print(">>> 已禁用确定性模式 (Randomness Enabled)，结果将不可复现")

    # 打印关键配置参数到控制台
    print("\n" + "="*60)
    print("有害言论检测(BERT+FNN)训练配置参数")
    print("="*60)
    print(f"实验目录: {base_config.experiment_path}")
    print(f"开始时间戳: {timestamp}")
    print("\n--- 数据集配置 ---")
    print(f"数据集名称: {base_config.dataset_name}")
    print(f"训练集路径: {base_config.train_path}")
    print(f"测试集路径: {base_config.test_path}")
    print(f"BERT模型路径: {base_config.bert_path}")
    print("\n--- 训练超参数 ---")
    print(f"批次大小 (batch_size): {base_config.batch_size}")
    print(f"训练轮数 (epochs): {base_config.epochs}")
    print(f"学习率 (lr): {base_config.lr}")
    print(f"最大序列长度 (max_len): {base_config.max_len}")
    print(f"Dropout比率: {base_config.dropout_rate}")
    print(f"冻结BERT层数: {base_config.freeze_bert_layers}")
    print(f"毒性类别数: {base_config.num_toxic_types}")
    print("\n--- 随机种子配置 ---")
    print(f"随机种子 (seed): {base_config.seed}")
    print(f"确定性模式: {base_config.use_deterministic}")
    print("="*60 + "\n")

    # 保存完整配置到JSON文件
    config_dict = {
        "timestamp": timestamp,
        "experiment_path": str(base_config.experiment_path),
        "dataset_name": base_config.dataset_name,
        "train_path": str(base_config.train_path),
        "test_path": str(base_config.test_path),
        "bert_path": str(base_config.bert_path),
        "attack_stance_dict_path": str(base_config.attack_stance_dict_path),
        "dirty_lexicon_path": str(base_config.dirty_lexicon_path),
        "seed": base_config.seed,
        "use_deterministic": base_config.use_deterministic,
        "batch_size": base_config.batch_size,
        "epochs": base_config.epochs,
        "lr": base_config.lr,
        "max_len": base_config.max_len,
        "dropout_rate": base_config.dropout_rate,
        "freeze_bert_layers": base_config.freeze_bert_layers,
        "num_toxic_types": base_config.num_toxic_types
    }

    config_file = base_config.experiment_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f">>> 配置文件已保存至: {config_file}\n")

    return base_config


def plot_metrics(base_config, epochs, losses, f1_scores, precisions, recalls):
    """
    绘制损失与各项评价指标曲线图（对齐 mlp_train）
    """
    plt.figure(figsize=base_config.fig_size)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制验证集损失 (左轴)
    lns1 = ax1.plot(epochs, losses, color='tab:red', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 绘制验证集 F1, Precision, Recall (右轴)
    lns2 = ax2.plot(epochs, f1_scores, color='tab:blue', label='Test F1')
    lns3 = ax2.plot(epochs, precisions, color='tab:green', linestyle='--', label='Test Precision')
    lns4 = ax2.plot(epochs, recalls, color='tab:orange', linestyle=':', label='Test Recall')

    ax2.set_ylabel('Score', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title('BERT + FNN Training Metrics')

    # 合并图例
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')

    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = base_config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, optimizer, loss_fn, scheduler=None):
    # 模型切换为训练模式
    fnn.train()
    bert_model.train()

    train_loss_epoch = 0  # 当前epoch的总损失
    train_sample_nums = 0  # 训练集总样本数

    pbar = tqdm(train_iter, desc=f"Epoch {epoch + 1}/{epochs}", colour="YELLOW")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # 词索引序列
        attention_mask = batch["attention_mask"].to(device)  # 注意力掩码
        token_type_ids = batch["token_type_ids"].to(device)  #
        toxic_ids = batch["toxic_ids"].to(device)  # 毒性序列
        stance_ids = batch["stance_ids"].to(device)
        labels = batch["toxic"].to(device)  # 真实标签

        # bert_optimizer.zero_grad()
        # fnn_optimizer.zero_grad()
        optimizer.zero_grad()

        # 使用BERT特征提取
        feature_extraction = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            toxic_ids=toxic_ids,
            stance_ids=stance_ids,
        )
        # 映射到毒性分类
        logits = fnn(feature_extraction)

        loss_batch = loss_fn(logits, labels)  # 计算批次的总损失

        train_loss_epoch += loss_batch.item()
        train_sample_nums += batch["input_ids"].shape[0]  # 累加样本数

        loss_batch.backward()
        # torch.nn.utils.clip_grad_norm_(list(bert_model.parameters()) + list(fnn.parameters()), max_norm=1.0)  # 梯度裁剪
        # bert_optimizer.step()
        # fnn_optimizer.step()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()  # 调整学习率

    return train_loss_epoch, train_sample_nums, pbar


def evaluate(fnn, bert_model, dev_iter, device, loss_fn):
    # 一个epoch完成后在验证集上验证
    fnn.eval()
    bert_model.eval()

    dev_loss_epoch = 0  # 验证集总损失
    dev_total_nums = 0  # 验证集总样本数

    all_predicts = []  # 存储所有预测标签
    all_labels = []  # 存储所有真实标签

    with torch.no_grad():
        for batch in dev_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            toxic_ids = batch["toxic_ids"].to(device)  # 毒性序列
            stance_ids = batch["stance_ids"].to(device)
            labels = batch["toxic"].to(device)

            feature_extraction = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                toxic_ids=toxic_ids,
                stance_ids=stance_ids,
            )
            logits = fnn(feature_extraction)

            loss = loss_fn(logits, labels)
            dev_loss_epoch += loss.item()

            pred = torch.softmax(logits, dim=1)
            max_idx = torch.argmax(pred, dim=1)

            dev_total_nums += batch["input_ids"].shape[0]

            # 收集所有的预测标签和真实标签
            all_predicts.extend(max_idx.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return dev_loss_epoch, dev_total_nums, all_predicts, all_labels


def final_train(base_config):
    """
    使用配置中的参数进行最终训练
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'=' * 20} 开始最终模型训练 {'=' * 20}")

    # ---------------- 1. 加载数据 ----------------
    max_len = base_config.max_len

    trn_data = ToxicDataset(
        base_config.train_path,
        base_config.bert_path,
        base_config.attack_stance_dict_path,
        base_config.dirty_lexicon_path,
        max_len=max_len
    )
    val_data = ToxicDataset(
        base_config.test_path,
        base_config.bert_path,
        base_config.attack_stance_dict_path,
        base_config.dirty_lexicon_path,
        max_len=max_len
    )

    generator = None
    if base_config.use_deterministic:
        generator = torch.Generator().manual_seed(base_config.seed)

    train_iter = DataLoader(
        trn_data,
        batch_size=base_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_iter = DataLoader(
        val_data,
        batch_size=base_config.batch_size,
        shuffle=False,
    )

    # ---------------- 2. 模型初始化 ----------------
    bert_model = ModifiedBert(base_config.bert_path, base_config.freeze_bert_layers, base_config.num_toxic_types).to(device)
    fnn = FC(base_config.dropout_rate).to(device)

    epochs = base_config.epochs

    # ---------------- 3. 优化器 ----------------
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # 损失函数

    optimizer = optim.AdamW(
        list(bert_model.parameters()) + list(fnn.parameters()),
        lr=base_config.lr,
        # weight_decay=base_config.weight_decay
    )  # 优化器
    # bert_optimizer = optim.AdamW(bert_model.parameters(), lr=base_config.lr)
    # fnn_optimizer = optim.AdamW(fnn.parameters(), lr=base_config.lr)

    pct_start = base_config.train_pct_start
    # cooldown_epochs = int(epochs * pct_start)
    # scheduler = OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=base_config.lr,
    #     steps_per_epoch=len(train_iter),
    #     epochs=epochs,
    #     pct_start=pct_start,  # 前pct_start%步用于Warmup
    #     anneal_strategy='cos',  # 使用余弦退火
    #     cycle_momentum=True
    # )
    scheduler = None  # 不使用scheduler，如需启用请取消上面的注释

    # ---------------- 4. 定义指标 ----------------
    train_losses = []  # 汇总每个epoch训练集上的损失
    dev_losses = []  # 汇总每个epoch验证集上的损失
    best_f1 = 0  # 验证集上最优F1
    best_epoch = 0  # 最优指标对应的epoch
    
    # 记录在测试集上指标最高时对应的模型
    best_fnn_state = None
    best_bert_state = None
    
    # 用于可视化的指标历史
    epoch_list = []
    f1_history = []
    precision_history = []
    recall_history = []

    for epoch in range(epochs):
        # --------------------训练阶段----------------------
        # train_loss_epoch, train_sample_nums, pbar = train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, bert_optimizer, fnn_optimizer, loss_fn)
        train_loss_epoch, train_sample_nums, pbar = train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, optimizer, loss_fn, scheduler)

        dev_loss_epoch, dev_total_nums, all_predicts, all_labels = evaluate(fnn, bert_model, val_iter, device, loss_fn)

        # ---------------- 计算本轮指标 ----------------
        train_loss = train_loss_epoch / train_sample_nums  # 训练集平均损失
        dev_loss = dev_loss_epoch / dev_total_nums  # 验证集平均损失

        f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_predicts, average='macro', zero_division=0)  # 精确率
        recall = recall_score(all_labels, all_predicts, average='macro', zero_division=0)  # 召回率
        accuracy = accuracy_score(all_labels, all_predicts)

        pbar.write(f"\nepochs={epoch + 1}[指标]\n{'>' * 3}训练集：平均损失：{train_loss}")
        pbar.write(f"{'>' * 3}验证集：平均损失：{dev_loss}")
        pbar.write(f"{'>' * 3}精确率 (Precision): {precision:.4f}")
        pbar.write(f"{'>' * 3}召回率 (Recall): {recall:.4f}")
        pbar.write(f"{'>' * 3}F1分数 (F1 Score): {f1:.4f}")
        pbar.write(f"{'>' * 3}准确率（Accuracy）: {accuracy:.4f}")

        # ---------------- 记录需要可视化的指标 ----------------
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        epoch_list.append(epoch + 1)
        f1_history.append(f1)
        precision_history.append(precision)
        recall_history.append(recall)

        # 保存最佳模型（基于F1）
        if f1 > best_f1:
            improvement = f1 - best_f1
            best_f1 = f1
            best_epoch = epoch + 1
            print(f">>> 发现更优模型 (F1: {f1:.4f})，提升：{improvement:.4f}")
            best_bert_state = bert_model.state_dict()
            best_fnn_state = fnn.state_dict()
    
    print(f"\n{'='*20} 训练完成 {'='*20}")
    print(f"最佳模型: epoch={best_epoch}, F1={best_f1:.4f}")
    
    # ---------------- 5. 保存最佳模型 ----------------
    if best_fnn_state is not None and best_bert_state is not None:
        model_save_path = base_config.experiment_path / "best_model.pth"
        torch.save({
            'fnn_state_dict': best_fnn_state,
            'bert_state_dict': best_bert_state,
        }, model_save_path)
        print(f">>> 最佳模型已保存至: {model_save_path}")
    
    # ---------------- 6. 生成可视化图表 ----------------
    plot_metrics(base_config, epoch_list, dev_losses, f1_history, precision_history, recall_history)


if __name__ == '__main__':
    # 初始化配置
    base_config = init()
    # 开始训练
    final_train(base_config)
