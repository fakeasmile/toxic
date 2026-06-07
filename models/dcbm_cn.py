"""DCBM-CN: 解耦概念瓶颈模型用于中文有害言论检测

架构:
    输入文本 → LLM → 显式概念向量 (K_e维)
    输入文本 → RoBERTa → [CLS] → VAE → 隐式概念 (K_i维, IB约束)
    隐式概念 → 话题判别器 (GRL) → 话题分类 (对抗训练)
    [显式 ⊕ 隐式] → 稀疏门控分类器 → 有毒/无毒

训练目标:
    L_total = L_cls + alpha * L_IB + beta * L_adv + gamma * L_sparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 (GRL)

    前向传播: 恒等变换
    反向传播: 梯度乘以 -lambda_adv
    """

    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


class VAEEncoder(nn.Module):
    """VAE编码器: RoBERTa [CLS] → 隐式概念

    结构: [CLS](768) → FC(768,512) → ReLU → FC(512,256) → ReLU
          → mu_layer(256, K_i)
          → logvar_layer(256, K_i)
          → z = mu + eps * exp(logvar)  (重参数化)
    """

    def __init__(self, roberta_hidden=768, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(roberta_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, cls_hidden, use_mean=True):
        """
        Args:
            cls_hidden: RoBERTa [CLS] 表示, shape [B, 768]
            use_mean: 推理时使用均值，不采样
        Returns:
            z: 隐式概念, shape [B, K_i]
            mu: 均值, shape [B, K_i]
            logvar: 对数方差, shape [B, K_i]
        """
        h = self.encoder(cls_hidden)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        if use_mean:
            z = mu
        else:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)

        return z, mu, logvar

    def kl_loss(self, mu, logvar):
        """计算KL散度: KL(q(z|x) || N(0, I))"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


class TopicDiscriminator(nn.Module):
    """话题判别器: 隐式概念 → 话题分类 (含GRL)

    结构: z_i(K_i) → GRL → FC(K_i, 64) → ReLU → FC(64, num_topics)
    """

    def __init__(self, latent_dim=32, num_topics=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_topics),
        )
        self.num_topics = num_topics

    def forward(self, z, lambda_adv=1.0):
        """
        Args:
            z: 隐式概念, shape [B, K_i]
            lambda_adv: GRL反转强度
        Returns:
            topic_pred: 话题预测logits, shape [B, num_topics]
        """
        reversed_z = GradientReversalLayer.apply(z, lambda_adv)
        topic_pred = self.classifier(reversed_z)
        return topic_pred


class SparseGateClassifier(nn.Module):
    """稀疏门控分类器: 概念向量 → 门控 → 分类

    结构: [显式 ⊕ 隐式](K_e+K_i) → sigmoid门控 → 逐元素乘
          → Dropout → FC(K_e+K_i, hidden) → ReLU → Dropout → FC(hidden, 2)
    """

    def __init__(self, in_features, hidden_features=128, dropout_rate=0.3):
        super().__init__()
        self.gate_layer = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, concept_vector):
        """
        Args:
            concept_vector: [显式 ⊕ 隐式] 概念向量, shape [B, K_e+K_i]
        Returns:
            logits: 分类logits, shape [B, 2]
            gate_weights: 门控权重, shape [B, K_e+K_i]
        """
        gate_weights = torch.sigmoid(self.gate_layer(concept_vector))
        x = concept_vector * gate_weights
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, gate_weights

    def l1_penalty(self, gate_weights):
        """L1稀疏惩罚"""
        return gate_weights.abs().sum(dim=1).mean()


class DCBM_CN(nn.Module):
    """DCBM-CN: 解耦概念瓶颈模型

    整合VAE编码器、话题判别器、稀疏门控分类器。
    RoBERTa编码器冻结，仅训练VAE、判别器、分类器。
    """

    def __init__(
        self,
        roberta_path,
        explicit_dim=197,
        latent_dim=32,
        num_topics=4,
        hidden_features=128,
        dropout_rate=0.3,
        freeze_roberta=True,
    ):
        super().__init__()
        self.explicit_dim = explicit_dim
        self.latent_dim = latent_dim

        # RoBERTa编码器（冻结）
        self.roberta = AutoModel.from_pretrained(roberta_path)
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # VAE编码器
        self.vae_encoder = VAEEncoder(
            roberta_hidden=self.roberta.config.hidden_size,
            latent_dim=latent_dim,
        )

        # 话题判别器
        self.topic_discriminator = TopicDiscriminator(
            latent_dim=latent_dim,
            num_topics=num_topics,
        )

        # 稀疏门控分类器
        total_concept_dim = explicit_dim + latent_dim
        self.classifier = SparseGateClassifier(
            in_features=total_concept_dim,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        )

    def forward(self, input_ids, attention_mask, explicit_concepts, lambda_adv=1.0, use_mean=True):
        """
        Args:
            input_ids: token ids, shape [B, seq_len]
            attention_mask: attention mask, shape [B, seq_len]
            explicit_concepts: 显式概念向量, shape [B, K_e]
            lambda_adv: GRL反转强度
            use_mean: VAE推理时是否使用均值
        Returns:
            dict: {
                'logits': 分类logits [B, 2],
                'gate_weights': 门控权重 [B, K_e+K_i],
                'mu': VAE均值 [B, K_i],
                'logvar': VAE对数方差 [B, K_i],
                'topic_pred': 话题预测 [B, num_topics],
                'kl_loss': KL散度标量,
                'l1_penalty': L1稀疏惩罚标量,
                'implicit_concepts': 隐式概念 [B, K_i],
            }
        """
        # RoBERTa提取[CLS]
        with torch.no_grad() if not self.roberta.training else torch.enable_grad():
            roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_hidden = roberta_output.last_hidden_state[:, 0, :]  # [B, 768]

        # VAE编码
        z_implicit, mu, logvar = self.vae_encoder(cls_hidden, use_mean=use_mean)
        kl_loss = self.vae_encoder.kl_loss(mu, logvar)

        # 话题判别
        topic_pred = self.topic_discriminator(z_implicit, lambda_adv=lambda_adv)

        # 拼接概念
        concept_vector = torch.cat([explicit_concepts, z_implicit], dim=1)  # [B, K_e+K_i]

        # 门控分类
        logits, gate_weights = self.classifier(concept_vector)
        l1_penalty = self.classifier.l1_penalty(gate_weights)

        return {
            'logits': logits,
            'gate_weights': gate_weights,
            'mu': mu,
            'logvar': logvar,
            'topic_pred': topic_pred,
            'kl_loss': kl_loss,
            'l1_penalty': l1_penalty,
            'implicit_concepts': z_implicit,
        }

    def get_explicit_gate_weights(self, gate_weights):
        """从门控权重中分离显式概念部分"""
        return gate_weights[:, :self.explicit_dim]

    def get_implicit_gate_weights(self, gate_weights):
        """从门控权重中分离隐式概念部分"""
        return gate_weights[:, self.explicit_dim:]
