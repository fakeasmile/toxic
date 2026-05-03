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


class DualChannelFusion(nn.Module):
    """
    双通道融合模型（BERT语义通道 + 形容词概念向量通道）。
    
    核心设计：概念向量作为BERT的"语义增强器"，逐层与BERT交互。
    
    结构:
        BERT → 逐层隐藏状态 [num_layers, B, seq_len, 768]
                  │
                  ├──► 每层与概念向量交互 ──► 增强后的隐藏状态
                  │
        概念向量 ─┴──► 投影到768维 ──► 生成逐层增强信号
                  
        最终[CLS] → LayerNorm → Dropout → 分类
    
    特点：
    - 概念向量不直接参与分类，而是增强BERT的语义表示
    - 逐层交互让BERT的每一层都能感知有害概念
    - 残差连接保留BERT原始能力
    """

    def __init__(self, bert_path, concept_dim, dropout_rate=0.3):
        super(DualChannelFusion, self).__init__()

        # BERT
        self.bert = BertModel.from_pretrained(bert_path)
        self.num_layers = self.bert.config.num_hidden_layers

        # 概念向量投影：将概念向量投影到与BERT隐藏状态相同的维度
        self.concept_proj = nn.Sequential(
            nn.Linear(concept_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 逐层交互门控：为每一层生成一个门控信号
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.Sigmoid(),
            ) for _ in range(self.num_layers)
        ])

        # 输出层
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, concept_vector, token_type_ids=None):
        # 获取BERT所有隐藏层
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[1:]  # [num_layers, B, seq_len, 768]

        # 投影概念向量
        concept_feat = self.concept_proj(concept_vector)  # [B, 768]

        # 逐层交互：每层BERT输出与概念向量融合
        enhanced_states = []
        for i, hidden_state in enumerate(hidden_states):
            # hidden_state: [B, seq_len, 768]
            # concept_feat: [B, 768]
            
            # 将概念向量扩展到与hidden_state相同的seq_len
            concept_expanded = concept_feat.unsqueeze(1).expand(-1, hidden_state.size(1), -1)  # [B, seq_len, 768]
            
            # 拼接后生成门控
            concat = torch.cat([hidden_state, concept_expanded], dim=-1)  # [B, seq_len, 1536]
            gate = self.layer_gates[i](concat)  # [B, seq_len, 768]
            
            # 门控融合：保留BERT信息，增强概念相关部分
            enhanced = hidden_state + gate * concept_expanded  # [B, seq_len, 768]
            enhanced_states.append(enhanced)

        # 使用最后一层的[CLS]进行分类
        final_cls = enhanced_states[-1][:, 0, :]  # [B, 768]

        # LayerNorm + Dropout + 分类
        final_cls = self.layer_norm(final_cls)
        final_cls = self.dropout(final_cls)
        logits = self.classifier(final_cls)
        return logits


class ConceptGuidedBERT(nn.Module):
    """
    概念引导的BERT模型。

    核心设计：概念向量作为语义增强信号，通过门控机制与BERT各层交互，
    并动态调整每层的重要性权重。

    结构:
        BERT → 所有隐藏层 [num_layers, B, seq_len, 768]
                  │
                  ├──► 每层与概念增强信号交互
                  │
        概念向量 ─┴──► 投影到768维 ──► 生成逐层增强信号
                         │
                         └──► 层权重生成器 ──► 动态加权各层

        最终[CLS] → LayerNorm → 与概念向量拼接 → 分类

    特点：
    - 概念向量投影到BERT维度，与每层隐藏状态逐元素交互
    - 层权重动态调整，让模型自动学习哪层BERT输出更重要
    - 最终拼接保留原始概念信息，增强分类能力
    """

    def __init__(self, bert_path, concept_dim, dropout_rate=0.3):
        super(ConceptGuidedBERT, self).__init__()

        # BERT
        self.bert = BertModel.from_pretrained(bert_path)
        self.num_layers = self.bert.config.num_hidden_layers

        # 概念向量投影：将概念向量投影到与BERT隐藏状态相同的维度
        self.concept_proj = nn.Sequential(
            nn.Linear(concept_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 逐层门控：为每一层生成一个门控信号
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.Sigmoid(),
            ) for _ in range(self.num_layers)
        ])

        # 层权重生成器：决定每层BERT的重要性（使用softmax归一化）
        self.layer_weights = nn.Sequential(
            nn.Linear(concept_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_layers),
        )

        # 输出层
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768 + concept_dim, 2)

    def forward(self, input_ids, attention_mask, concept_vector, token_type_ids=None):
        # 投影概念向量
        concept_feat = self.concept_proj(concept_vector)  # [B, 768]

        # 生成层权重 [B, num_layers]
        layer_logits = self.layer_weights(concept_vector)  # [B, num_layers]
        layer_weights = F.softmax(layer_logits, dim=-1)  # 归一化，和为1

        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[1:]  # [num_layers, B, seq_len, 768]

        # 逐层交互：每层BERT输出与概念向量融合
        enhanced_states = []
        for i, hidden_state in enumerate(hidden_states):
            # hidden_state: [B, seq_len, 768]
            # concept_feat: [B, 768]

            # 将概念向量扩展到与hidden_state相同的seq_len
            concept_expanded = concept_feat.unsqueeze(1).expand(-1, hidden_state.size(1), -1)  # [B, seq_len, 768]

            # 拼接后生成门控
            concat = torch.cat([hidden_state, concept_expanded], dim=-1)  # [B, seq_len, 1536]
            gate = self.layer_gates[i](concat)  # [B, seq_len, 768]

            # 门控融合：保留BERT信息，增强概念相关部分
            enhanced = hidden_state + gate * concept_expanded  # [B, seq_len, 768]
            enhanced_states.append(enhanced)

        # stack后按层权重加权求和: [B, num_layers, seq_len, 768]
        stacked = torch.stack(enhanced_states, dim=1)  # [B, num_layers, seq_len, 768]
        weights = layer_weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_layers, 1, 1]
        weighted_sum = (stacked * weights).sum(dim=1)  # [B, seq_len, 768]

        # 取[CLS]位置
        cls_output = weighted_sum[:, 0, :]  # [B, 768]

        # LayerNorm + Dropout
        cls_output = self.layer_norm(cls_output)
        cls_output = self.dropout(cls_output)

        # 与原始概念向量拼接
        fused = torch.cat([cls_output, concept_vector], dim=-1)  # [B, 768 + concept_dim]

        # 分类
        logits = self.classifier(fused)
        return logits
