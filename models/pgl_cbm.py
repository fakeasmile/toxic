import torch
import torch.nn as nn
from transformers import AutoModel


class PinyinEncoder(nn.Module):
    def __init__(self, pinyin_vocab_size=1800, pinyin_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(pinyin_vocab_size, 64)
        self.bilstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, pinyin_dim)

    def forward(self, pinyin_ids):
        emb = self.embedding(pinyin_ids)
        _, (h_n, _) = self.bilstm(emb)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc(h)


class GlyphEncoder(nn.Module):
    def __init__(self, glyph_input_dim=50, glyph_dim=128):
        super().__init__()
        self.fc_in = nn.Linear(glyph_input_dim, 64)
        self.bilstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(128, glyph_dim)

    def forward(self, glyph_features):
        x = self.fc_in(glyph_features)
        _, (h_n, _) = self.bilstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc_out(h)


class LexiconEncoder(nn.Module):
    def __init__(self, lexicon_size, lexicon_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(lexicon_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, lexicon_dim)

    def forward(self, lexicon_vec):
        x = self.relu(self.fc1(lexicon_vec))
        return self.fc2(x)


class ConceptBottleneck(nn.Module):
    def __init__(self, fusion_dim, num_concepts):
        super().__init__()
        self.fc = nn.Linear(fusion_dim, num_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fused):
        return self.sigmoid(self.fc(fused))


class PGLCBMModel(nn.Module):
    def __init__(
        self,
        plm_name="chinese-roberta-wwm-ext",
        pinyin_vocab_size=1800,
        pinyin_dim=128,
        glyph_input_dim=50,
        glyph_dim=128,
        lexicon_size=100,
        lexicon_dim=64,
        num_concepts=20,
        num_classes=2,
        use_pinyin=True,
        use_glyph=True,
        use_lexicon=True,
        concept_loss_weight=0.1,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name)
        plm_hidden = self.plm.config.hidden_size

        self.use_pinyin = use_pinyin
        self.use_glyph = use_glyph
        self.use_lexicon = use_lexicon
        self.concept_loss_weight = concept_loss_weight

        self.pinyin_encoder = PinyinEncoder(pinyin_vocab_size, pinyin_dim) if use_pinyin else None
        self.glyph_encoder = GlyphEncoder(glyph_input_dim, glyph_dim) if use_glyph else None
        self.lexicon_encoder = LexiconEncoder(lexicon_size, lexicon_dim) if use_lexicon else None

        fusion_dim = plm_hidden
        if use_pinyin:
            fusion_dim += pinyin_dim
        if use_glyph:
            fusion_dim += glyph_dim
        if use_lexicon:
            fusion_dim += lexicon_dim

        self.dropout = nn.Dropout(0.3)
        self.concept_bottleneck = ConceptBottleneck(fusion_dim, num_concepts)
        self.concept_to_label = nn.Linear(num_concepts, num_classes)

        self.ce_loss = nn.CrossEntropyLoss()
        self.concept_loss = nn.MSELoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        pinyin_ids=None,
        glyph_features=None,
        lexicon_vec=None,
        labels=None,
        concept_labels=None,
    ):
        plm_out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        h = plm_out.last_hidden_state[:, 0, :]

        feats = [h]
        if self.use_pinyin and pinyin_ids is not None:
            feats.append(self.pinyin_encoder(pinyin_ids))
        if self.use_glyph and glyph_features is not None:
            feats.append(self.glyph_encoder(glyph_features))
        if self.use_lexicon and lexicon_vec is not None:
            feats.append(self.lexicon_encoder(lexicon_vec))

        fused = torch.cat(feats, dim=1)
        fused = self.dropout(fused)
        concept_probs = self.concept_bottleneck(fused)
        logits = self.concept_to_label(concept_probs)

        loss = None
        if labels is not None:
            ce = self.ce_loss(logits, labels)
            loss = ce
            if concept_labels is not None:
                mse = self.concept_loss(concept_probs, concept_labels)
                loss = ce + self.concept_loss_weight * mse

        if loss is not None:
            return logits, concept_probs, loss
        return logits, concept_probs
