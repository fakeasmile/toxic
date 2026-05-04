from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTBaseline(nn.Module):
    """
    BERT基线模型，仅使用BERT的[CLS]向量进行分类。
    """

    def __init__(self, bert_path, dropout_rate=0.1):
        super(BERTBaseline, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))
        return logits
