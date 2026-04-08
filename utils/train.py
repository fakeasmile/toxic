import torch

from models.fc import FC
from .data_preprocess import ToxicDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from models.bert import ModifiedBert, Pure_Bert
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging
log = logging.getLogger("final_train")


def train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, optimizer, loss_fn):
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

        # scheduler.step()  # 调整学习率

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

def final_train(base_config, best_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'=' * 20} 开始最终模型训练 {'=' * 20}")
    print(f"使用最优参数: {best_params}")
    log.info(f"{'=' * 20} 开始最终模型训练 {'=' * 20}")
    log.info(f"使用最优参数: {best_params}")

    # ---------------- 1. 加载数据 ----------------
    max_len = best_params['max_len']

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
        batch_size=best_params["batch_size"],
        shuffle=True,
        generator=generator,
    )
    val_iter = DataLoader(
        val_data,
        batch_size=best_params["batch_size"],
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
        lr=best_params["lr"],
        # weight_decay=best_params["weight_decay"]
    )  # 优化器
    # bert_optimizer = optim.AdamW(bert_model.parameters(), lr=best_params["lr"])
    # fnn_optimizer = optim.AdamW(fnn.parameters(), lr=best_params["lr"])

    # pct_start = base_config.final_train_pct_start
    # cooldown_epochs = int(epochs * pct_start)
    # scheduler = OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=best_params["lr"],
    #     steps_per_epoch=len(train_iter),
    #     epochs=epochs,
    #     pct_start=pct_start,  # 前pct_start%步用于Warmup
    #     anneal_strategy='cos',  # 使用余弦退火
    #     cycle_momentum=True
    # )

    # ---------------- 4. 定义指标 ----------------
    train_losses = []  # 汇总每个epoch训练集上的损失
    dev_losses = []  # 汇总每个epoch验证集上的损失
    best_target = 0  # 验证集上最优指标
    best_epoch = 0  # 最优指标对应的epoch
    dev_accuracies = []  # 汇总每个epoch验证集上的正确率

    # 记录在测试集上指标最高时对应的模型
    best_fnn_state = None
    best_bert_state = None

    for epoch in range(epochs):
        # --------------------训练阶段----------------------
        # train_loss_epoch, train_sample_nums, pbar = train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, bert_optimizer, fnn_optimizer, loss_fn)
        train_loss_epoch, train_sample_nums, pbar = train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, optimizer, loss_fn)

        dev_loss_epoch, dev_total_nums, all_predicts, all_labels = evaluate(fnn, bert_model, val_iter, device, loss_fn)

        # ---------------- 计算本轮指标 ----------------
        train_loss = train_loss_epoch / train_sample_nums  # 训练集平均损失
        dev_loss = dev_loss_epoch / dev_total_nums  # 验证集平均损失

        f1 = f1_score(all_labels, all_predicts, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predicts, average='weighted', zero_division=0)  # 精确率
        recall = recall_score(all_labels, all_predicts, average='weighted', zero_division=0)  # 召回率
        accuracy = accuracy_score(all_labels, all_predicts)

        pbar.write(f"\nepochs={epoch + 1}[指标]\n{'>' * 3}训练集：平均损失：{train_loss}")
        pbar.write(f"{'>' * 3}验证集：平均损失：{dev_loss}")
        pbar.write(f"{'>' * 3}精确率 (Precision): {precision:.4f}")
        pbar.write(f"{'>' * 3}召回率 (Recall): {recall:.4f}")
        pbar.write(f"{'>' * 3}F1分数 (F1 Score): {f1:.4f}")
        pbar.write(f"{'>' * 3}准确率（Accuracy）: {accuracy:.4f}")

        log.info(f"\nepochs={epoch + 1}[指标]\n{'>' * 3}训练集：平均损失：{train_loss}")
        log.info(f"{'>' * 3}验证集：平均损失：{dev_loss}")
        log.info(f"{'>' * 3}精确率 (Precision): {precision:.4f}")
        log.info(f"{'>' * 3}召回率 (Recall): {recall:.4f}")
        log.info(f"{'>' * 3}F1分数 (F1 Score): {f1:.4f}")
        log.info(f"{'>' * 3}准确率（Accuracy）: {accuracy:.4f}")

        # ---------------- 记录需要可视化的指标 ----------------
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        dev_accuracies.append(accuracy)

        # 保存模型
        if f1 > best_target:
            best_target = f1
            best_epoch = epoch + 1
            print(f"best_target: {best_target:.4f}, in epoch {epoch + 1}")
            log.info(f"best_target: {best_target:.4f}, in epoch {epoch + 1}")
            best_bert_state = bert_model.state_dict()
            best_fnn_state = fnn.state_dict()
    print(f"保存模型，epoch={best_epoch}, best_target={best_target}")
    log.info(f"保存模型，epoch={best_epoch}, best_target={best_target}")
    return train_losses, dev_losses, dev_accuracies, best_fnn_state, best_bert_state