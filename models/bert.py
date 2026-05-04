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
        cls_output = outputs.pooler_output
        logits = self.classifier(self.dropout(cls_output))
        return logits


class BERTMultiTask(nn.Module):
    """
    BERT多任务学习模型。

    核心设计：概念向量不作为输入特征，而是作为辅助训练目标。
    BERT同时学习两个任务：
    1. 主任务：毒性分类（二分类）
    2. 辅助任务：概念向量预测（回归）

    通过多任务学习，概念信息作为正则化信号引导BERT的内部表示，
    让BERT学到更有结构的语义表示。

    结构:
        BERT → [CLS] [B, 768]
                  │
                  ├──► 分类头 ──► 毒性标签 (主任务)
                  │
                  └──► 概念预测头 ──► 概念向量 (辅助任务)

    损失函数:
        L = L_cls + λ * L_concept
        L_cls: 交叉熵损失（毒性分类）
        L_concept: MSE损失（概念向量预测）
        λ: 概念损失权重（控制辅助任务的影响）
    """

    def __init__(self, bert_path, concept_dim, dropout_rate=0.1, concept_loss_weight=0.1):
        super(BERTMultiTask, self).__init__()

        self.concept_dim = concept_dim
        self.concept_loss_weight = concept_loss_weight

        # BERT
        self.bert = BertModel.from_pretrained(bert_path)

        # 主任务：毒性分类
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 2)

        # 辅助任务：概念向量预测
        self.concept_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, concept_dim),
        )

    def forward(self, input_ids, attention_mask, labels=None, concept_vector=None, token_type_ids=None):
        """
        :param input_ids: [B, seq_len]
        :param attention_mask: [B, seq_len]
        :param labels: [B] 毒性标签 (0或1)
        :param concept_vector: [B, concept_dim] 概念向量
        :param token_type_ids: [B, seq_len]
        :return: (loss, cls_logits, concept_pred)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.pooler_output

        # 主任务：毒性分类
        cls_logits = self.classifier(self.dropout(cls_output))

        # 辅助任务：概念向量预测
        concept_pred = self.concept_head(cls_output)

        # 计算损失
        loss = None
        if labels is not None and concept_vector is not None:
            cls_loss = F.cross_entropy(cls_logits, labels)
            concept_loss = F.mse_loss(concept_pred, concept_vector)
            loss = cls_loss + self.concept_loss_weight * concept_loss

        return loss, cls_logits, concept_pred
