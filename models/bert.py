import torch
from torch import nn
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


class DualChannelFusion(nn.Module):
    """
    双通道融合模型（BERT语义通道 + 形容词概念向量通道）。
    改进版：BERT不降维 + 残差连接 + LayerNorm。
    结构: BERT → [CLS] (768维) ──┐
                                  ├→ 残差连接: H(x) = x + F(x) → LayerNorm → 分类头
          概念向量 → 投影 → 门控 →┘
    其中 x = BERT输出, F(x) = 门控后的概念向量
    全参数微调BERT，使用分层学习率。
    """

    def __init__(self, bert_path, concept_dim, proj_dim=768, dropout_rate=0.3):
        """
        :param bert_path: bert-base-chinese 模型路径
        :param concept_dim: 概念向量维度（由形容词词典大小决定，从数据中自动推断）
        :param proj_dim: 投影维度，默认768（与BERT输出一致，不降维）
        :param dropout_rate: Dropout比率
        """
        super(DualChannelFusion, self).__init__()

        # ========== BERT语义通道（作为残差连接的identity） ==========
        self.bert = BertModel.from_pretrained(bert_path)
        # BERT输出保持768维，不降维，直接作为x

        # ========== 概念向量通道（作为残差连接的F(x)） ==========
        self.concept_proj = nn.Sequential(
            nn.Linear(concept_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.concept_gate = nn.Linear(proj_dim, proj_dim)  # Sigmoid门控，独立控制每个维度

        # ========== 融合与分类 ==========
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(proj_dim, 2)

    def forward(self, input_ids, attention_mask, concept_vector, token_type_ids=None):
        """
        :param input_ids: token ID序列
        :param attention_mask: 注意力掩码
        :param concept_vector: 形容词概念向量
        :param token_type_ids: 句子类型ID
        :return: 分类logits
        """
        # BERT通道：[CLS] 作为identity (x)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = bert_out.pooler_output  # [B, 768]  ← 这是x

        # 概念向量通道：投影 → Sigmoid门控 作为F(x)
        concept_proj = self.concept_proj(concept_vector)  # [B, 768]
        concept_gate_weights = torch.sigmoid(self.concept_gate(concept_proj))  # [B, 768]
        concept_feat = concept_proj * concept_gate_weights  # [B, 768]  ← 这是F(x)

        # 真正的残差连接: H(x) = x + F(x)
        fused = cls_output + concept_feat  # [B, 768]

        # LayerNorm + Dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        # 分类
        logits = self.classifier(fused)
        return logits


class ConceptGuidedBERT(nn.Module):
    """
    概念引导的BERT模型。
    结构: 概念向量 → 注意力生成器 → 层权重
          BERT → 所有隐藏层 → 按层权重加权求和 → [CLS] → 与概念向量拼接 → 分类头
    利用概念向量动态决定各BERT层的重要性权重，实现概念引导的层级注意力。
    """

    def __init__(self, bert_path, concept_dim, dropout_rate=0.3):
        """
        :param bert_path: bert-base-chinese 模型路径
        :param concept_dim: 概念向量维度（由形容词词典大小决定，从数据中自动推断）
        :param dropout_rate: Dropout比率
        """
        super(ConceptGuidedBERT, self).__init__()

        # ========== BERT ==========
        self.bert = BertModel.from_pretrained(bert_path)
        self.num_layers = self.bert.config.num_hidden_layers

        # ========== 概念向量注意力生成器 ==========
        self.attention_generator = nn.Sequential(
            nn.Linear(concept_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_layers),
            nn.Sigmoid(),
        )

        # ========== Dropout与分类头 ==========
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768 + concept_dim, 2)

    def forward(self, input_ids, attention_mask, concept_vector, token_type_ids=None):
        """
        :param input_ids: token ID序列
        :param attention_mask: 注意力掩码
        :param concept_vector: 形容词概念向量
        :param token_type_ids: 句子类型ID
        :return: 分类logits
        """
        # 概念向量输入注意力生成器 → 层权重 [B, num_layers]
        layer_weights = self.attention_generator(concept_vector)

        # BERT前向传播获取所有隐藏层
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # [num_layers+1, B, seq_len, 768]

        # 去掉embedding层，保留num_layers个隐藏层
        hidden_states = hidden_states[1:]  # [num_layers, B, seq_len, 768]

        # stack后维度: [B, num_layers, seq_len, 768]
        stacked = torch.stack(hidden_states, dim=1)

        # layer_weights扩展维度: [B, num_layers, 1, 1]
        weights = layer_weights.unsqueeze(-1).unsqueeze(-1)

        # 加权求和: [B, seq_len, 768]
        weighted_sum = (stacked * weights).sum(dim=1)

        # 取[CLS]位置（索引0）
        cls_output = weighted_sum[:, 0, :]  # [B, 768]

        # 与concept_vector拼接: [B, 768 + concept_dim]
        fused = torch.cat([cls_output, concept_vector], dim=-1)

        # 分类
        logits = self.classifier(self.dropout(fused))
        return logits
