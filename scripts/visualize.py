"""
可视化
"""
from configs.base_config import BaseConfig
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']

def visualize_train_dev_losses(model_name: Path, if_save = True):
    """
    可视化训练过程中训练集和验证集上损失变化趋势
    :param if_save: 是否保存图表
    :param model_name: .pth文件的路径
    :return:
    """

    checkpoint = torch.load(Path(model_name), weights_only=True)
    loss_history = checkpoint["loss_history"]

    train_losses = loss_history["train"]
    dev_losses = loss_history["dev"]
    dev_accuracies = loss_history["dev_accuracies"]

    plt.figure(figsize=(10,6))
    epochs = range(1, len(train_losses)+1)

    plt.plot(epochs, train_losses, "o-", label="Train Loss")
    plt.plot(epochs, dev_losses, "o-", label="Dev Loss")
    plt.plot(epochs, dev_accuracies, "o-", label="Dev accuracies")

    plt.title(f"训练集和验证集损失变化趋势({model_name.stem})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    if if_save:
        plt.savefig(model_name.parent / f"{model_name.stem}.png")

    plt.show()


if __name__ == '__main__':
    visualize_train_dev_losses(Path(__file__).parent.parent / "experiments" / "20260315-035855.pth")
