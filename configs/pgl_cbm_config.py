from pathlib import Path


class PGLCBMConfig():
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.experiment_path = self.base_path / "experiments"
        self.homo_graph_path = self.raw_data_path / "homo_graph.json"
        self.toxic_lexicon_path = self.raw_data_path / "toxic_lexicon.json"
        self.concept_path = self.raw_data_path / "adjective" / "discovered_concepts.csv"

        self.seed = 1
        self.use_deterministic = False

        self.plm_name = "chinese-roberta-wwm-ext"
        self.plm_hidden_size = 768
        self.max_length = 128
        self.freeze_plm = False
        self.plm_lr = 2e-5

        self.pinyin_vocab_size = 1800
        self.pinyin_emb_dim = 64
        self.pinyin_hidden_dim = 64
        self.pinyin_output_dim = 128

        self.glyph_input_dim = 50
        self.glyph_emb_dim = 64
        self.glyph_hidden_dim = 64
        self.glyph_output_dim = 128

        self.lexicon_dim = 64

        self.num_concepts = 56
        self.concept_loss_weight = 0.1

        self.batch_size = 16
        self.epochs = 30
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.patience = 5
        self.dropout = 0.3

        self.use_homophone_restore = True
        self.homophone_restore_threshold = 0.0

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = PGLCBMConfig()
    print(config)
