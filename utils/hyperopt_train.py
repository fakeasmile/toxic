import optuna
import torch
import numpy as np

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
log = logging.getLogger("hyperopt_train_logger")


def train_one_epoch(fnn, bert_model, train_iter, epoch, epochs, device, bert_optimizer, fnn_optimizer, loss_fn):
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

        bert_optimizer.zero_grad()
        fnn_optimizer.zero_grad()

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
        bert_optimizer.step()
        fnn_optimizer.step()

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
            labels = batch["toxic"].to(device)

            feature_extraction = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                toxic_ids=toxic_ids,
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


def train(trial, train_iter, dev_iter, base_config, device, hyper_params):
    """

    :param trial:
    :param train_iter: 训练DataLoader
    :param dev_iter: 验证DataLoader
    :param base_config:
    :param device:
    :param hyper_params: 超参数配置
    :return:
    """

    # ---------------------模型-----------------------------
    bert_model = ModifiedBert(base_config.bert_path, base_config.freeze_bert_layers, base_config.num_toxic_types).to(device)  # BERT
    fnn = FC(base_config.dropout_rate).to(device)  # FNN

    # ---------------------优化器-----------------------------
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # 损失函数
    # loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    optimizer = optim.AdamW(
        list(bert_model.parameters()) + list(fnn.parameters()),
        lr=hyper_params["lr"],
        weight_decay=hyper_params["weight_decay"]
    )  # 优化器

    # total_steps = len(train_iter) * epochs  # 由于OneCycleLR在每个batch之后调度学习率，所以学习率被调度的总次数=批次数*epochs
    pct_start = base_config.train_pct_start  # Warmup占比
    cooldown_epochs = int(base_config.epochs * pct_start)  # 前pct_start%个epoch（学习率上升期）不触发早停

    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=hyper_params["lr"],
        steps_per_epoch=len(train_iter),
        epochs=base_config.epochs,
        pct_start=pct_start,
        anneal_strategy='cos',  # 使用余弦退火
        cycle_momentum=True
    )

    # ---------------------定义指标-----------------------------
    best_target = 0  # 验证集上最优指标
    patience_counter = 0  # 早停计数器
    patience = base_config.train_patience  # 耐心值

    for epoch in range(base_config.epochs):
        # --------------------训练阶段----------------------
        train_loss_epoch, train_sample_nums, pbar = train_one_epoch(fnn, bert_model, train_iter, epoch, base_config.epochs, device, optimizer, scheduler, loss_fn)

        # --------------------验证阶段----------------------
        dev_loss_epoch, dev_total_nums, all_predicts, all_labels = evaluate(fnn, bert_model, dev_iter, device, loss_fn)

        # ---------------------计算本轮epoch的指标-----------------------------
        train_loss = train_loss_epoch / train_sample_nums  # 训练集平均损失
        dev_loss = dev_loss_epoch / dev_total_nums  # 验证集平均损失

        f1 = f1_score(all_labels, all_predicts, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predicts, average='weighted', zero_division=0)  # 精确率
        recall = recall_score(all_labels, all_predicts, average='weighted', zero_division=0)  # 召回率
        accuracy = accuracy_score(all_labels, all_predicts)  # 准确率

        # ---------------------输出控制台并记录日志-----------------------------
        pbar.write(f"\n[Trial {trial.number}] epochs={epoch + 1}[指标]\n{'>'*3}训练集：平均损失：{train_loss}")
        pbar.write(f"{'>'*3}验证集：平均损失：{dev_loss}")
        pbar.write(f"{'>'*3}精确率 (Precision): {precision:.4f}")
        pbar.write(f"{'>'*3}召回率 (Recall): {recall:.4f}")
        pbar.write(f"{'>'*3}F1分数 (F1 Score): {f1:.4f}")
        pbar.write(f"{'>'*3}准确率（Accuracy）: {accuracy:.4f}")

        log.info(f"\n[Trial {trial.number}] epochs={epoch + 1}[指标]\n{'>' * 3}训练集：平均损失：{train_loss}")
        log.info(f"{'>' * 3}验证集：平均损失：{dev_loss}")
        log.info(f"{'>' * 3}精确率 (Precision): {precision:.4f}")
        log.info(f"{'>' * 3}召回率 (Recall): {recall:.4f}")
        log.info(f"{'>' * 3}F1分数 (F1 Score): {f1:.4f}")
        log.info(f"{'>' * 3}准确率（Accuracy）: {accuracy:.4f}")

        # 向Optuna报告当前epoch的指标（指标为需要优化的指标）
        trial.report(f1, epoch)

        # 判断本次试验是否应该被剪枝
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1} with target: {f1:.4f}")
            log.info(f"Trial {trial.number} pruned at epoch {epoch + 1} with target: {f1:.4f}")
            raise optuna.TrialPruned()

        # early stopped：当在验证集上的指标连续patience次epoch没有提升时触发早停（在学习率上升期不触发早停）
        if epoch < cooldown_epochs:
            if f1 > best_target:
                best_target = f1
        else:
            if f1 > best_target:
                best_target = f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}, 指标: {best_target:.4f}(cooldown_epochs={cooldown_epochs})")
                log.info(f"Early stopping triggered at epoch {epoch + 1}, 指标: {best_target:.4f}(cooldown_epochs={cooldown_epochs})")
                raise optuna.TrialPruned()

    return best_target