"""分析经过data_preprocess预处理之后的数据集"""

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
from tqdm import tqdm
from collections import defaultdict, Counter
from utils.data_preprocess import ToxicDataset
from configs.base_config import BaseConfig

def analyze_toxic_dataset(dataset, name):
    print(f"数据集：{name}，总样本数: {len(dataset)}")

    # 初始化统计容器
    stats = {
        'toxic_count': 0,  # 统计整个数据集中，有毒的样本总数
        'topic_dist': defaultdict(int),  # 统计整个数据集中，不同毒性分类的数量
        'toxic_by_topic': defaultdict(int),  # 统计整个数据集中，不同毒性分类，确认有毒的数量
        'valid_seq_lengths': [],  # 有效序列长度
        'truncate_count': 0,  # 被max_len截断样本的数量

        'stance_in_toxic': 0,  # 真实标签为有毒的样本中有攻击立场样本的数量
        'dirty_in_toxic': 0,  # 真实标签为有毒的样本中有脏词样本的数量
        'stance_dirty_in_toxic': 0,  # 真实标签为有毒的样本中同时有攻击立场和脏词的样本的数量

        'stance_in_no_toxic': 0,  # 真实标签为无毒的样本中有攻击立场样本的数量
        'dirty_in_no_toxic': 0,  # 真实标签为无毒的样本中有脏词样本的数量
        'stance_dirty_in_no_toxic': 0,  # 真实标签为无毒的样本中同时有攻击立场和脏词的样本的数量
    }

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    max_len = dataset.max_len

    print("开始遍历数据集进行分析...")
    for i, sample in enumerate(tqdm(dataloader)):
        # 提取数据
        label = sample['toxic'].item()  # 是否具有毒性0/1
        topic = dataset.raw_data[i]['topic']  # 毒性类型race/gender/region
        stance_ids = sample['stance_ids'].squeeze(0)  # 攻击立场,[max_len]（使用squeeze目的在于DataLoader加载批次数据[1, max_len]）
        toxic_ids = sample['toxic_ids'].squeeze(0)  # 脏词序列,[max_len]
        attention_mask = sample['attention_mask'].squeeze(0)  # 注意力掩码（pad的部分为0）,[max_len]

        has_stance = (stance_ids > 0).any().item()  # 检测是否具有攻击立场
        has_dirty = (toxic_ids > 0).any().item()  # 检测是否有脏词

        if label == 1:
            stats['toxic_count'] += 1  # 统计数据集中有毒样本的数量
            stats['toxic_by_topic'][topic] += 1  # 统计不同毒性类别中，有毒样本的数量
            # 统计真实标签为有毒的样本中，有攻击立场和有脏词序列的数量，以及同时有两者的数量
            if has_stance:
                stats["stance_in_toxic"]+=1
            if has_dirty:
                stats["dirty_in_toxic"]+=1
            if has_stance and has_dirty:
                stats["stance_dirty_in_toxic"]+=1

        # 统计不同毒性类别的数量
        stats["topic_dist"][topic] += 1

        # 计算有效长度 (非 padding 部分)
        valid_seq_len = attention_mask.sum().item()
        if valid_seq_len == max_len:
            stats['truncate_count'] += 1
        stats['valid_seq_lengths'].append(valid_seq_len)

        # 统计真实标签为无毒的样本中，有攻击立场和有脏词序列的数量，以及同时有两者的数量
        if label == 0:
            if has_stance:
                stats["stance_in_no_toxic"]+=1
            if has_dirty:
                stats["dirty_in_no_toxic"]+=1
            if has_stance and has_dirty:
                stats["stance_dirty_in_no_toxic"]+=1

    # --- 输出分析报告 ---
    print("\n" + "=" * 30)
    print("数据分析报告")
    print("=" * 30)

    print(f"\n[毒性分布]")
    all_nums = len(dataset)  # 样本总数
    toxic_nums = stats['toxic_count']  # 有毒样本总数
    print(f"总样本数量: {len(dataset)}, 有毒样本数量: {toxic_nums}；毒性比例: {toxic_nums / all_nums:.2%}")

    print(f"在所有样本中，有攻击立场占比：{(stats['stance_in_toxic']+stats['stance_in_no_toxic']) / all_nums:.2%},"
          f"有脏词占比：{(stats['dirty_in_toxic']+stats['dirty_in_no_toxic']) / all_nums:.2%}")
    print(f"\t在真实标签为无毒的样本中，有攻击立场样本占比：{stats['stance_in_no_toxic'] / (all_nums-toxic_nums):.2%}，"
          f"有脏词样本占比：{stats['dirty_in_no_toxic'] / (all_nums-toxic_nums):.2%}，同时有两者占比：{stats['stance_dirty_in_no_toxic'] / (all_nums-toxic_nums):.2%}")
    print(f"\t在真实标签为有毒的样本中，有攻击立场样本占比：{stats['stance_in_toxic'] / toxic_nums:.2%}，"
          f"有脏词样本占比：{stats['dirty_in_toxic'] / toxic_nums:.2%}，同时有两者占比：{stats['stance_dirty_in_toxic'] / toxic_nums:.2%}")

    print(f"\n[毒性样本类别分布]")
    for topic, count in stats['topic_dist'].items():
        print(
            f"毒性类别：{topic}，占总样本比例：{count / all_nums:.2%}；该类别中有毒样本占比：{stats['toxic_by_topic'][topic] / count:.2%}")


    # 绘制有效长度与数量的柱状图
    valid_len_count = Counter(stats['valid_seq_lengths'])
    fig, ax = plt.subplots(figsize = (10,6))
    ax.bar(valid_len_count.keys(), valid_len_count.values())
    ax.set_xlabel("有效长度")
    ax.set_ylabel("数量")
    ax.set_title(f"数据集：{name}中有效长度-数量关系")
    ax.text(0.95, 0.95, f"被截断的样本占比：{stats['truncate_count'] / all_nums:.2%}",
            transform=ax.transAxes,  # 使用轴坐标（0-1）
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_config = BaseConfig()

    # 加载训练集
    trn_data_raw = ToxicDataset(
        base_config.train_path,
        base_config.bert_path,
        base_config.attack_stance_dict_path,
        base_config.dirty_lexicon_path,
        max_len=100
    )
    analyze_toxic_dataset(trn_data_raw, "train")

    # 加载测试集
    trn_data_raw = ToxicDataset(
        base_config.test_path,
        base_config.bert_path,
        base_config.attack_stance_dict_path,
        base_config.dirty_lexicon_path,
        max_len=100
    )
    analyze_toxic_dataset(trn_data_raw, "test")