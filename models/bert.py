import torch
import torch.nn as nn
from transformers import BertModel


class BERTBaseline(nn.Module):
    """
    基于bert-base-chinese的二分类基线模型。
    结构: BERT → [CLS]池化输出 → Dropout → 线性分类头
    全参数微调，不冻结任何BERT层。
    """

    def __init__(self, bert_path, num_classes=2, dropout_rate=0.1):
        """
        :param bert_path: bert-base-chinese 模型路径
        :param num_classes: 分类类别数，默认2（Toxic/Non-Toxic）
        :param dropout_rate: Dropout比率
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        :param input_ids: token ID序列
        :param attention_mask: 注意力掩码
        :param token_type_ids: 句子类型ID
        :return: 分类logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(self.dropout(outputs.pooler_output))
