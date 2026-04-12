import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings


# ========== 自定义BERT词嵌入层 ==========
class CustomEmbeddings(BertEmbeddings):
    """
    在原始BERT三种嵌入基础上，添加额外嵌入向量
    原始嵌入 = Token + Segment + Position
    最终嵌入 = 原始嵌入 + sigmoid(a) * toxic_emb + sigmoid(b) * stance_emb
    a、b为可学习的动态权重（向量，每维度独立），sigmoid约束到[0,1]
    """
    def __init__(self, config, pretrained_embeddings, num_toxic_types=6):
        """

        :param config: BERT配置对象
        :param pretrained_embeddings: 原始预训练的BERT嵌入层（带权重）
        :param num_toxic_types: lexicon中毒性类别数
        """
        super().__init__(config)

        # 复用原始BERT的预训练嵌入层权重
        self.word_embeddings = pretrained_embeddings.word_embeddings  # 预训练词嵌入
        self.position_embeddings = pretrained_embeddings.position_embeddings  # 预训练位置嵌入
        self.token_type_embeddings = pretrained_embeddings.token_type_embeddings  # 预训练句子类型嵌入
        self.LayerNorm = pretrained_embeddings.LayerNorm  # 预训练归一化层
        self.dropout = pretrained_embeddings.dropout  # 预训练dropout层

        # 毒性（脏词序列）嵌入
        self.toxic_embeddings = nn.Embedding(num_toxic_types, config.hidden_size)
        nn.init.normal_(self.toxic_embeddings.weight, mean=0.0, std=0.02)

        # 攻击立场嵌入层
        self.stance_embeddings = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.stance_embeddings.weight, std=0.02)

        # 动态融合权重（向量，每维度独立），sigmoid约束到[0,1]
        self.toxic_weight = nn.Parameter(torch.zeros(config.hidden_size))
        self.stance_weight = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, **kwargs):
        # 获取BERT原始嵌入词向量
        embeddings = super().forward(input_ids, token_type_ids, position_ids)

        toxic_ids = kwargs.get('toxic_ids', None)  # 获取毒性词嵌入
        stance_ids = kwargs.get('stance_ids', None)  # 获取攻击立场词嵌入

        # 动态权重融合：sigmoid约束到[0,1]
        if toxic_ids is not None:
            toxic_emb = self.toxic_embeddings(toxic_ids)
            embeddings = embeddings + torch.sigmoid(self.toxic_weight) * toxic_emb

        if stance_ids is not None:
            stance_emb = self.stance_embeddings(stance_ids)
            embeddings = embeddings + torch.sigmoid(self.stance_weight) * stance_emb

        return embeddings

# ============= BERT特征提取 ============
class ModifiedBert(nn.Module):

    def __init__(self, bert_path, freeze_layers=12, num_toxic_types=6):
        super(ModifiedBert, self).__init__()
        self.virgin_bert = BertModel.from_pretrained(bert_path)  # 加载原始BERT模型

        # 替换BERT嵌入层
        self.virgin_bert.embeddings = CustomEmbeddings(
            self.virgin_bert.config,
            self.virgin_bert.embeddings,  # 带原始权重的BERT嵌入层
            num_toxic_types
        )

        self.freeze_bert_layers(freeze_layers)  # 冻结BERT的指定层数

    def freeze_bert_layers(self, freeze_layers):
        """
        冻结BERT的前freeze_layers层
        :param freeze_layers: 冻结层数
        :return:
        """
        # Embedding嵌入层（让毒性嵌入可训练）
        for param in self.virgin_bert.embeddings.parameters():
            param.requires_grad = True

        # 冻结底层
        for i in range(freeze_layers):
            for param in self.virgin_bert.encoder.layer[i].parameters():
                param.requires_grad = False

        # 解冻顶层
        for i in range(freeze_layers, self.virgin_bert.config.num_hidden_layers):
            for param in self.virgin_bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Pooler层解冻
        for param in self.virgin_bert.pooler.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, toxic_ids=None, stance_ids=None):
        """

        :param input_ids: token序列
        :param attention_mask: 注意力掩码
        :param token_type_ids: 句子类型
        :param toxic_ids: 毒性序列
        :param stance_ids: 攻击立场
        :return: 经过BERT特征提取的向量
        """
        # 获取BERT词嵌入层融合后的词嵌入向量
        embedding_output = self.virgin_bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            toxic_ids=toxic_ids,
            stance_ids=stance_ids,
        )

        extended_attention_mask = self.virgin_bert.get_extended_attention_mask(
            attention_mask,
            input_ids.shape,
        )

        # encoder
        encoder_outputs = self.virgin_bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            return_dict=True
        )

        # 获取输出
        sequence_output = encoder_outputs.last_hidden_state  # 序列特征
        pooler_output = self.virgin_bert.pooler(sequence_output)    # 池化特征

        # return {
        #     'pooler_output': pooler_output,
        #     'last_hidden_state': sequence_output
        # }
        return pooler_output


class Pure_Bert(nn.Module):
    def __init__(self, bert_path, freeze_layers=12):
        super(Pure_Bert, self).__init__()
        self.virgin_bert = BertModel.from_pretrained(bert_path)
        self.freeze_bert_layers(freeze_layers)

    def freeze_bert_layers(self, freeze_layers):
        """
        冻结BERT的前freeze_layers层
        :param freeze_layers: 冻结层数
        :return:
        """
        # Embedding嵌入层（让毒性嵌入可训练）
        for param in self.virgin_bert.embeddings.parameters():
            param.requires_grad = True

        # 冻结底层
        for i in range(freeze_layers):
            for param in self.virgin_bert.encoder.layer[i].parameters():
                param.requires_grad = False

        # 解冻顶层
        for i in range(freeze_layers, self.virgin_bert.config.num_hidden_layers):
            for param in self.virgin_bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Pooler层解冻
        for param in self.virgin_bert.pooler.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, toxic_ids=None):

        outputs = self.virgin_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.pooler_output