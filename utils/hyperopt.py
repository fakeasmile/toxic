import optuna
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from utils.data_preprocess import ToxicDataset
from utils.hyperopt_train import train

# 禁用optuna冗余日志
optuna.logging.set_verbosity(optuna.logging.WARNING)
import logging
log = logging.getLogger("hyperopt_train_logger")

def build_search_space(trial, search_space_config):
    """
    根据配置构建超参数搜索空间
    :param trial: optuna.Trial对象
    :param search_space_config: HyperOptConfig中的search_space，type=dict
    :return: 超参数字典
    """
    hyper_params = {}
    for key, param_config in search_space_config.items():
        if param_config["type"] == "float":
            hyper_params[key] = trial.suggest_float(
                key,
                low=param_config["low"],
                high=param_config["high"],
                log=param_config.get("log", False)
            )
        elif param_config["type"] == "categorical":
            hyper_params[key] = trial.suggest_categorical(
                key,
                choices=param_config["choices"]
            )
        elif param_config["type"] == "int":
            hyper_params[key] = trial.suggest_int(
                key,
                low=param_config["low"],
                high=param_config["high"],
                step=param_config.get("step", 1)
            )
    return hyper_params

def dynamic_load_data(base_config, current_max_len):
    """
    动态加载数据集
    :param current_max_len:
    :return:
    """
    # 加载训练集
    trn_data_raw = ToxicDataset(
        base_config.train_path,
        base_config.bert_path,
        base_config.attack_stance_dict_path,
        base_config.dirty_lexicon_path,
        max_len=current_max_len
    )
    # 获取训练集样本的标签
    labels = []
    indices = list(range(len(trn_data_raw)))
    for idx in indices:
        labels.append(trn_data_raw.raw_data[idx]["toxic"])
    labels = np.array(labels)

    # 根据正负样本标签按照比例分层抽样（防止类别不平衡），把原始训练集划分为训练集和验证集
    # 分别获取训练集中所有的正负样本的索引
    positive_indices_all = np.where(labels == 1)[0]
    negative_indices_all = np.where(labels == 0)[0]
    # print(f"正样本总数: {len(positive_indices_all)}, 负样本总数: {len(negative_indices_all)}")

    np.random.shuffle(positive_indices_all)
    np.random.shuffle(negative_indices_all)

    positive_count = int(len(positive_indices_all) * base_config.train_ratio)  # 训练集中正样本总数
    negative_count = int(len(negative_indices_all) * base_config.train_ratio)  # 训练集中负样本总数

    # 截取前n个作为训练集的正负样本，剩余作为验证集的样本
    train_idx = np.concatenate([positive_indices_all[:positive_count], negative_indices_all[:negative_count]])
    val_idx = np.concatenate([positive_indices_all[positive_count:], negative_indices_all[negative_count:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    trn_data = Subset(trn_data_raw, train_idx.tolist())
    val_data = Subset(trn_data_raw, val_idx.tolist())

    return trn_data, val_data



def objective(trial, base_config, hyperopt_config):
    # 1. 构建超参数搜索空间
    hyper_params = build_search_space(trial, hyperopt_config.search_space)

    # 动态提取本次试验的max_len
    current_max_len = hyper_params["max_len"]

    # 2. 动态加载数据集
    print(f"[Trial {trial.number}] （max_len={current_max_len}）\n超参数：{hyper_params}")
    log.info(f"\n[Trial {trial.number}] （max_len={current_max_len}）\n超参数：{hyper_params}")
    trn_data, val_data = dynamic_load_data(base_config, current_max_len)

    # 3. 构建DataLoader
    generator = None
    if base_config.use_deterministic:
        generator = torch.Generator().manual_seed(base_config.seed)

    train_iter = DataLoader(
        trn_data,
        batch_size=hyper_params["batch_size"],
        shuffle=True,
        generator=generator,
    )
    dev_iter = DataLoader(
        val_data,
        batch_size=hyper_params["batch_size"],
        shuffle=False
    )

    # 4. 训练并返回最优验证集上指标
    best_target = train(
        trial=trial,
        train_iter=train_iter,
        dev_iter=dev_iter,
        hyper_params=hyper_params,
        base_config=base_config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # print(f"[trial {trial.number}],best target: {best_target}")

    return best_target


def run_hyperopt(base_config, hyperopt_config):
    """
    运行Optuna超参数搜索
    :param base_config: Base_Config对象
    :param hyperopt_config: HyperOptConfig对象
    :return: 最优超参数、最优准确率
    """

    print(f"{'-'*10}开始超参数搜索（共{hyperopt_config.n_trials}轮）{'-'*10}")
    log.info(f"{'-'*10}开始超参数搜索（共{hyperopt_config.n_trials}轮）{'-'*10}")
    # 创建Optuna研究对象
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    # 执行搜索
    study.optimize(
        lambda trial: objective(trial, base_config, hyperopt_config),
        n_trials=hyperopt_config.n_trials,
    )

    # 获取最优超参数组合
    best_params = study.best_params  # 格式：{"lr": ..., "batch_size": ..., ...}

    # 获取超参数搜索过程中的最优指标
    best_score = study.best_value

    # 获取最优试验的详细信息
    best_trial = study.best_trial

    print(f"{'=' * 20} 搜索结束 {'=' * 20}")
    print(f"🏆 最佳试验编号: {best_trial.number}")
    print(f"📈 最佳F1分数: {best_score:.4f}")
    print(f"⚙️ 最佳超参数:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")

    log.info("超参数搜索结束 - 最终结果汇总")
    log.info(f"🏆 最佳试验编号: {best_trial.number}，📈 最佳F1分数: {best_score:.4f}")
    log.info("⚙️ 最佳超参数配置:")
    for key, value in best_params.items():
        log.info(f"   {key:<30}: {value}")

    return best_params

