import json

import torch
from torch.utils.data import Dataset
from modelscope import AutoTokenizer
import re
from pathlib import Path


def load_attack_stance_dict(attack_stance_dict_path):
    """
    加载攻击立场词典
    attack_stance_dict结构：
        {
            "region":{...},
            "race":{...},
            "gender":{...},
            "LGBT":{...},
            "political":{...},
            "other":{...}
        }
    其中{...}结构为：
        "key":[word1, word2, ...]，例如："南京": ["歧视","弱化唱白历史"]
        "key"表示主词，[...]表示主词对应的关联词
    :param attack_stance_dict_path: 攻击立场词典的路径
    :return:
    """
    with open(attack_stance_dict_path, 'r', encoding='utf-8') as f:
        attack_stance_dict = json.load(f)
    return attack_stance_dict

def load_dirty_dict(dirty_dict_path):
    """
    返回的脏词词典结构：
    {"general": [...], "LGBT": [...], "racism": [...], "region": [...], "sexism": [...]}
    :param dirty_dict_path: 脏词词典目录
    :return:
    """
    dirty_dict = {}  # 脏词词典
    for dirty_dict_json_path in Path(dirty_dict_path).glob("*.json"):
        with open(dirty_dict_json_path, "r", encoding="utf-8") as f:
            dirty = json.load(f)

        dirty_word_list = []
        for dirty_word in list(dirty.keys()):
            dirty_word_list.append(dirty_word.strip())
        dirty_dict[dirty_dict_json_path.stem] = dirty_word_list

    return dirty_dict


class ToxicDataset(Dataset):
    def __init__(self, data_path, bert_path, attack_stance_dict_path, dirty_dict_path, max_len):
        """

        :param data_path: 要加载的数据集的路径
        :param bert_path: Bert模型路径
        :param attack_stance_dict_path: 攻击立场词典路径
        :param dirty_dict_path: 脏词词典目录
        :param max_len: 样本最大词序列长度
        """
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)  # Bert分词器
        self.attack_stance_dict = load_attack_stance_dict(attack_stance_dict_path)  # 攻击立场词典
        self.dirty_dict = load_dirty_dict(dirty_dict_path)  # 脏词词典
        self.max_len = max_len  # 最大填充/截断长度

        '''
        raw_data格式：
        [{"topic":..., "content":..., "toxic":..., "toxic_one_hot":...}, {...}, ...]
        '''
        with open(data_path, "r", encoding="utf-8") as f:  # 加载数据集
            self.raw_data = json.load(f)

    def __len__(self):
        return len(self.raw_data)

    def detect_stance(self, input_ids):
        """
        当文本中存在包含攻击立场的「主词 + 至少一个关联词」时，判定存在攻击立场，把主词和关联词位置设置为1，表示具有攻击立场
        eg:如果文本中出现了主词：北京，关联词：打工，教育，这些词在input_ids中的位置对应的stance_ids位置应该设置为1
                input_ids=['[CLS]', '我', '认', '为', '北', '京', '高', '考', '的', '打', '工', '的', '北', '京', '的', '教', '育', '[SEP]']
                stance_ids=[  0,      0,   0,    0,    1,   1,    0,   0,    0,    1,   1,    0,   1,    1,   0,    1,    1,    0, ]
        :param input_ids: 原始文本经过BERT分词器分词后的索引序列
        :return:
        """
        stance_ids = [0]*input_ids.size(1)  # 初始每个词都没有攻击立场

        text = "".join(self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True).split(" "))  # 原始文本（可能被截断处理）
        tokens_seq = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze())  # token序列

        for _, primary_dict in self.attack_stance_dict.items():
            # primary_dict表示{key:[word1, word2, ...]}形式的主词和关联词的对应关系
            for primary_word, secondary_word_list in primary_dict.items():

                # 检查ori_text中是否出现主词，若没有出现则跳过
                primary_word_regex = re.compile(rf"{re.escape(primary_word)}")  # 查找主词的正则表达式
                primary_word_matches = list(primary_word_regex.finditer(text))

                skip = True
                for match in primary_word_matches:
                    skip = False
                    break

                if skip:
                    continue

                # 确认text中肯定含有主词

                # 检查是否至少存在该主词对应的一个关联词，若存在，把所有出现主词和关联词的位置设为有攻击立场；否则跳过
                regex_pattern_str = "|".join([re.escape(w) for w in secondary_word_list])
                secondary_words_regex = re.compile(rf"{regex_pattern_str}")  # 查找所有关联词的正则表达式
                secondary_words_matches = list(secondary_words_regex.finditer(text))

                skip = True
                for match in secondary_words_matches:
                    skip = False
                    break

                if skip:
                    continue

                # 此时文本中含有主词和该主词对应的至少一个关联词，设置主词位置有攻击立场
                for match in primary_word_matches:
                    # 能匹配上说明文本中一定含有这个主词
                    pri_word = match.group()  # 匹配上的主词
                    pri_length = len(pri_word)
                    for index, w in enumerate(tokens_seq):
                        if w == pri_word[:1] and "".join(tokens_seq[index:index + pri_length]) == pri_word:
                            stance_ids[index:index + pri_length] = [1]*pri_length
                # 设置关联词位置有攻击立场
                for match in secondary_words_matches:
                    sec_word = match.group()
                    sec_length = len(sec_word)
                    for index, w in enumerate(tokens_seq):
                        if w == sec_word[:1] and "".join(tokens_seq[index:index + sec_length]) == sec_word:
                            stance_ids[index:index + sec_length] = [1] * sec_length
        return stance_ids

    def detect_dirty(self, input_ids):
        """
        标记脏词（可以处理txl，但是无法处理未登入词[UNK]）
        :param input_ids:
        :return:
        """
        toxic_ids = [0]*input_ids.size(1)

        text = "".join(self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True).split(" "))  # 原始文本（截断？）
        tokens_seq = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze())  # token序列

        for dirty_index, (dirty_category, all_dirty_word_list) in enumerate(self.dirty_dict.items()):
            # 查找文本中脏词的正则表达式
            dirty_word_pattern_str = "|".join([re.escape(w) for w in all_dirty_word_list])
            dirty_word_regex = re.compile(rf"{dirty_word_pattern_str}")
            dirty_word_matches = list(dirty_word_regex.finditer(text))

            for match in dirty_word_matches:
                dirty_word = match.group()
                dirty_word_length = len(dirty_word)

                for index, w in enumerate(tokens_seq):
                    if w == dirty_word[:1]:
                        sub_dirty_word_list = tokens_seq[index: index+dirty_word_length]
                        sub_str = ""
                        for s in sub_dirty_word_list:
                            if s.startswith("##"):
                                sub_str+=s[2:]
                            else:
                                sub_str+=s
                        if sub_str == dirty_word:
                            toxic_ids[index: index+dirty_word_length] = [dirty_index+1]*dirty_word_length


        return toxic_ids


    def __getitem__(self, index):
        """
        raw_data格式：
        [{"topic":..., "content":..., "toxic":..., "toxic_one_hot":...}, {...}, ...]
        """
        sample = self.raw_data[index]
        content = str(sample["content"])

        # 使用bert分词器对文本分词
        inputs = self.tokenizer(
            content,
            return_tensors="pt",
            padding="max_length",  # 以最大长度补齐
            max_length=self.max_len,  # 最大长度
            truncation=True,  # 超长截断
            return_offsets_mapping=True
        )

        # 检测攻击立场
        stance_ids = self.detect_stance(inputs["input_ids"])
        # 检测脏词序列
        toxic_ids = self.detect_dirty(inputs["input_ids"])

        sample_new = {
            # "topic": sample["topic"],
            # "content": content,
            "toxic": torch.tensor(sample["toxic"], dtype=torch.long),  # 是否具有毒性，0/1
            # "toxic_one_hot": torch.tensor(sample["toxic_one_hot"], dtype=torch.long),  # 独热编码
            "input_ids": inputs["input_ids"].clone().detach().squeeze(dim=0),  # 词索引序列
            "token_type_ids": inputs["token_type_ids"].clone().detach().squeeze(dim=0),  #
            "attention_mask": inputs["attention_mask"].clone().detach().squeeze(dim=0),  # 注意力掩码
            "stance_ids": torch.tensor(stance_ids, dtype=torch.long),  # 攻击立场,[max_len]
            "toxic_ids": torch.tensor(toxic_ids, dtype=torch.long),  # 脏词序列,[max_len]
            # "toxic_type_one_hot": torch.zeros(3, dtype=torch.long),
            # "expression_one_hot": torch.zeros(4, dtype=torch.long),
            # "target": torch.zeros(2, dtype=torch.long),
        }

        return sample_new


if __name__ == '__main__':
    base_path = Path(__file__).parent.parent
    dataset = ToxicDataset(
        base_path / "data" / "raw" / "test.json",
        base_path / "models" / "bert-base-chinese",
        base_path / "data" / "raw" / "attack_stance.json",
        base_path / "data" / "raw" / "lexicon",
        30,
    )
    y = dataset[0]
    print(y)
