import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import pypinyin
except ImportError:
    print("请安装pypinyin: pip install pypinyin")
    sys.exit(1)

try:
    import jieba
except ImportError:
    print("请安装jieba: pip install jieba")
    sys.exit(1)

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.pgl_cbm_config import PGLCBMConfig
from models.pgl_cbm import PGLCBMModel

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']


class HomophoneRestorer:
    def __init__(self, homo_graph_path, toxic_lexicon_path):
        with open(homo_graph_path, 'r', encoding='utf-8') as f:
            homo_graph = json.load(f)
        with open(toxic_lexicon_path, 'r', encoding='utf-8') as f:
            toxic_lexicon = json.load(f)

        pinyin_to_chars = homo_graph.get('pinyin_to_chars', {})
        char_to_pinyin = homo_graph.get('char_to_pinyin', {})

        self.char_to_homophones = defaultdict(set)
        for char, pinyins in char_to_pinyin.items():
            for py in pinyins:
                if py in pinyin_to_chars:
                    for homo_char in pinyin_to_chars[py]:
                        if homo_char != char:
                            self.char_to_homophones[char].add(homo_char)

        self.toxic_words = set(toxic_lexicon.get('words', []))
        self.toxic_chars = set()
        for w in self.toxic_words:
            for c in w:
                self.toxic_chars.add(c)

    def restore(self, text):
        chars = list(text)
        restored = chars.copy()

        for length in range(4, 1, -1):
            for i in range(len(text) - length + 1):
                substring = text[i:i + length]
                for toxic_word in self.toxic_words:
                    if len(toxic_word) != length:
                        continue
                    if self._is_homophone_match(substring, toxic_word):
                        for j in range(length):
                            restored[i + j] = toxic_word[j]

        return ''.join(restored)

    def _is_homophone_match(self, substring, toxic_word):
        if len(substring) != len(toxic_word):
            return False
        for sc, tc in zip(substring, toxic_word):
            if sc == tc:
                continue
            if tc in self.char_to_homophones.get(sc, set()):
                continue
            if sc in self.char_to_homophones.get(tc, set()):
                continue
            return False
        return True


class PinyinProcessor:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.pinyin_to_id = {'[PAD]': 0, '[UNK]': 1}
        self._build_vocab()

    def _build_vocab(self):
        idx = 2
        for code in range(0x4E00, 0x9FFF + 1):
            char = chr(code)
            pinyins = pypinyin.pinyin(char, style=pypinyin.NORMAL)
            for p in pinyins:
                py = p[0]
                if py and py not in self.pinyin_to_id:
                    self.pinyin_to_id[py] = idx
                    idx += 1

    def text_to_pinyin_ids(self, text):
        ids = []
        for char in text:
            pinyins = pypinyin.pinyin(char, style=pypinyin.NORMAL)
            if pinyins and pinyins[0][0]:
                py = pinyins[0][0]
                ids.append(self.pinyin_to_id.get(py, 1))
            else:
                ids.append(1)
        ids = ids[:self.max_length]
        ids += [0] * (self.max_length - len(ids))
        return ids

    def get_vocab_size(self):
        return len(self.pinyin_to_id)


class GlyphProcessor:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.radical_table = [
            '一', '丨', '丶', '丿', '乙', '二', '亠', '人', '儿', '入',
            '八', '冂', '冖', '冫', '几', '凵', '刀', '力', '勹', '匕',
            '匚', '匸', '十', '卜', '卩', '厂', '厶', '又', '口', '囗',
            '土', '士', '夂', '夊', '夕', '大', '女', '子', '宀', '寸',
        ]
        self.structure_table = [
            '左右', '上下', '左中右', '上中下', '全包围',
            '上三包围', '下三包围', '左三包围', '独体',
        ]
        self.radical_to_idx = {r: i for i, r in enumerate(self.radical_table)}
        self.structure_to_idx = {s: i for i, s in enumerate(self.structure_table)}
        self.glyph_dim = len(self.radical_table) + 1 + len(self.structure_table)
        self.char_to_glyph = {}
        self._build_char_glyph()

    def _build_char_glyph(self):
        for code in range(0x4E00, 0x9FFF + 1):
            char = chr(code)
            radical_feat = np.zeros(len(self.radical_table), dtype=np.float32)
            stroke_count = self._estimate_strokes(char)
            structure_feat = np.zeros(len(self.structure_table), dtype=np.float32)
            struct_idx = self._estimate_structure(char)
            if struct_idx is not None:
                structure_feat[struct_idx] = 1.0
            else:
                structure_feat[-1] = 1.0
            self.char_to_glyph[char] = np.concatenate([
                radical_feat, [stroke_count], structure_feat
            ])

    def _estimate_strokes(self, char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            return min(max(1, (code - 0x4E00) % 20 + 1), 30)
        return 0

    def _estimate_structure(self, char):
        code = ord(char)
        struct_map = code % len(self.structure_table)
        return struct_map

    def text_to_glyph_features(self, text):
        features = np.zeros((self.max_length, self.glyph_dim), dtype=np.float32)
        for i, char in enumerate(text[:self.max_length]):
            if char in self.char_to_glyph:
                features[i] = self.char_to_glyph[char]
        return features


class LexiconMatcher:
    def __init__(self, toxic_lexicon_path):
        with open(toxic_lexicon_path, 'r', encoding='utf-8') as f:
            lexicon = json.load(f)
        self.words = lexicon.get('words', [])
        self.word_to_idx = lexicon.get('word_to_idx', {})
        self.lexicon_size = len(self.words)

    def text_to_lexicon_vec(self, text):
        vec = np.zeros(self.lexicon_size, dtype=np.float32)
        words_in_text = set(jieba.lcut(text))
        for i, word in enumerate(self.words):
            if word in text or word in words_in_text:
                vec[i] = 1.0
        return vec


class PGLCBMDataset(Dataset):
    def __init__(self, data, tokenizer, pinyin_processor, glyph_processor,
                 lexicon_matcher, homophone_restorer, max_length, use_homophone_restore):
        self.data = data
        self.tokenizer = tokenizer
        self.pinyin_processor = pinyin_processor
        self.glyph_processor = glyph_processor
        self.lexicon_matcher = lexicon_matcher
        self.max_length = max_length
        self.use_homophone_restore = use_homophone_restore

        self.input_ids_list = []
        self.attention_mask_list = []
        self.pinyin_ids_list = []
        self.glyph_features_list = []
        self.lexicon_vec_list = []
        self.labels = []

        for item in tqdm(data, desc="预处理数据"):
            text = item['content']
            if use_homophone_restore:
                text = homophone_restorer.restore(text)

            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.input_ids_list.append(encoding['input_ids'].squeeze(0))
            self.attention_mask_list.append(encoding['attention_mask'].squeeze(0))
            self.pinyin_ids_list.append(torch.tensor(
                pinyin_processor.text_to_pinyin_ids(text), dtype=torch.long
            ))
            self.glyph_features_list.append(torch.tensor(
                glyph_processor.text_to_glyph_features(text), dtype=torch.float32
            ))
            self.lexicon_vec_list.append(torch.tensor(
                lexicon_matcher.text_to_lexicon_vec(text), dtype=torch.float32
            ))
            self.labels.append(item['toxic'])

    def __getitem__(self, idx):
        return (
            self.input_ids_list[idx],
            self.attention_mask_list[idx],
            self.pinyin_ids_list[idx],
            self.glyph_features_list[idx],
            self.lexicon_vec_list[idx],
            self.labels[idx],
        )

    def __len__(self):
        return len(self.data)


def train(config, model, train_loader, val_loader, test_loader, device):
    plm_params = list(model.plm.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith('plm.')
    ]

    optimizer = torch.optim.AdamW([
        {'params': plm_params, 'lr': config.plm_lr},
        {'params': other_params, 'lr': config.lr},
    ], weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    epoch_list = []
    val_loss_history = []
    val_f1_history = []
    val_precision_history = []
    val_recall_history = []
    test_f1_history = []
    test_loss_history = []

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            input_ids, attention_mask, pinyin_ids, glyph_features, lexicon_vec, labels = [
                b.to(device) for b in batch
            ]

            concept_labels_batch = None
            if hasattr(train_loader.dataset, 'concept_labels') and train_loader.dataset.concept_labels is not None:
                concept_labels_batch = train_loader.dataset.concept_labels[
                    train_steps * config.batch_size:(train_steps + 1) * config.batch_size
                ]
                concept_labels_batch = concept_labels_batch[:labels.size(0)].to(device)

            optimizer.zero_grad()
            logits, concept_probs, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pinyin_ids=pinyin_ids,
                glyph_features=glyph_features,
                lexicon_vec=lexicon_vec,
                labels=labels,
                concept_labels=concept_labels_batch,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / max(train_steps, 1)

        model.eval()
        val_preds, val_labels_list = [], []
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, pinyin_ids, glyph_features, lexicon_vec, labels = [
                    b.to(device) for b in batch
                ]
                logits, concept_probs, v_loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pinyin_ids=pinyin_ids,
                    glyph_features=glyph_features,
                    lexicon_vec=lexicon_vec,
                    labels=labels,
                )
                total_val_loss += v_loss.item()
                val_steps += 1
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / max(val_steps, 1)
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_p = precision_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_r = recall_score(val_labels_list, val_preds, average='macro', zero_division=0)

        model.eval()
        test_preds, test_labels_list = [], []
        total_test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, pinyin_ids, glyph_features, lexicon_vec, labels = [
                    b.to(device) for b in batch
                ]
                logits, concept_probs, t_loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pinyin_ids=pinyin_ids,
                    glyph_features=glyph_features,
                    lexicon_vec=lexicon_vec,
                    labels=labels,
                )
                total_test_loss += t_loss.item()
                test_steps += 1
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())

        avg_test_loss = total_test_loss / max(test_steps, 1)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')

        epoch_list.append(epoch + 1)
        val_loss_history.append(avg_val_loss)
        val_f1_history.append(val_f1)
        val_precision_history.append(val_p)
        val_recall_history.append(val_r)
        test_f1_history.append(test_f1)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch + 1}: \n>>>Train Loss = {avg_train_loss:.4f}, "
              f"\n>>>Val Loss = {avg_val_loss:.4f}, \n>>>Val F1 = {val_f1:.4f}, "
              f"\n>>>Val P = {val_p:.4f}, \n>>>Val R = {val_r:.4f}, "
              f"\n>>>Test Loss = {avg_test_loss:.4f}, \n>>>Test F1 = {test_f1:.4f}")

        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f">>> 发现更优模型 (Val F1: {val_f1:.4f}), 提升: {improvement:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f">>> 早停触发: 验证集F1已连续 {config.patience} 个epoch未提升")
            break

    if best_state_dict is not None:
        torch.save(best_state_dict, config.experiment_path / "best_model.pth")
        print(f">>> 最佳模型: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    return (epoch_list, val_loss_history, val_f1_history, val_precision_history,
            val_recall_history, test_f1_history, test_loss_history)


def plot_metrics(config, epochs, val_losses, val_f1_scores, val_precisions, val_recalls,
                 test_f1_scores, test_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, val_losses, color='tab:red', label='Val Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', linestyle='--', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('PGL-CBM Training Metrics')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, val_f1_scores, color='tab:blue', label='Val F1')
    ax2.plot(epochs, val_precisions, color='tab:green', linestyle='--', label='Val Precision')
    ax2.plot(epochs, val_recalls, color='tab:orange', linestyle=':', label='Val Recall')
    ax2.plot(epochs, test_f1_scores, color='tab:red', linestyle='-.', label='Test F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = config.experiment_path / "metrics.png"
    plt.savefig(save_path)
    print(f">>> 训练图表已保存至: {save_path}")
    plt.close()


def evaluate(config, timestamp):
    experiment_dir = config.base_path / "experiments" / timestamp
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")

    with open(experiment_dir / "config.json", "r", encoding="utf-8") as f:
        saved_config_dict = json.load(f)

    saved_config = PGLCBMConfig()
    for key, value in saved_config_dict.items():
        if hasattr(saved_config, key):
            if key in ['experiment_path', 'base_path', 'raw_data_path', 'processed_path',
                        'homo_graph_path', 'toxic_lexicon_path', 'concept_path']:
                setattr(saved_config, key, Path(value))
            else:
                setattr(saved_config, key, value)

    if saved_config.use_deterministic:
        from utils.seed import set_reproducibility
        set_reproducibility(saved_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(saved_config.raw_data_path / saved_config.dataset_name / "test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(saved_config.plm_name)
    pinyin_processor = PinyinProcessor(max_length=saved_config.max_length)
    glyph_processor = GlyphProcessor(max_length=saved_config.max_length)
    lexicon_matcher = LexiconMatcher(saved_config.toxic_lexicon_path)
    homophone_restorer = HomophoneRestorer(saved_config.homo_graph_path, saved_config.toxic_lexicon_path)

    test_dataset = PGLCBMDataset(
        test_data, tokenizer, pinyin_processor, glyph_processor,
        lexicon_matcher, homophone_restorer, saved_config.max_length,
        saved_config.use_homophone_restore
    )
    test_loader = DataLoader(test_dataset, batch_size=saved_config.batch_size, shuffle=False)

    model = PGLCBMModel(
        plm_name=saved_config.plm_name,
        pinyin_vocab_size=pinyin_processor.get_vocab_size(),
        pinyin_dim=saved_config.pinyin_output_dim,
        glyph_input_dim=saved_config.glyph_input_dim,
        glyph_dim=saved_config.glyph_output_dim,
        lexicon_size=lexicon_matcher.lexicon_size,
        lexicon_dim=saved_config.lexicon_dim,
        num_concepts=saved_config.num_concepts,
        use_pinyin=getattr(saved_config, 'use_pinyin', True),
        use_glyph=getattr(saved_config, 'use_glyph', True),
        use_lexicon=getattr(saved_config, 'use_lexicon', True),
        concept_loss_weight=saved_config.concept_loss_weight,
    )
    model.load_state_dict(
        torch.load(experiment_dir / "best_model.pth", map_location=device, weights_only=False)
    )
    model.to(device).eval()

    all_preds, all_labels, all_concept_scores = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, pinyin_ids, glyph_features, lexicon_vec, labels = [
                b.to(device) for b in batch
            ]
            logits, concept_probs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pinyin_ids=pinyin_ids,
                glyph_features=glyph_features,
                lexicon_vec=lexicon_vec,
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_concept_scores.extend(concept_probs.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"])

    print("\n" + "=" * 30)
    print("      PGL-CBM 测试集评估结果")
    print("=" * 30)
    print(f"精确率 (Precision - Macro): {precision:.4f}")
    print(f"召回率 (Recall - Macro):    {recall:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1:.4f}")
    print("-" * 30)
    print(report)
    print("=" * 30)

    test_results_dir = experiment_dir / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    with open(test_results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }, f, indent=2, ensure_ascii=False)

    with open(test_results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("PGL-CBM 测试集评估结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write("\n" + "=" * 30 + "\n")

    label_names = ["Non-Toxic", "Toxic"]
    predictions = []
    for i in range(len(all_preds)):
        predictions.append({
            "index": i,
            "content": test_data[i]["content"],
            "true_label": int(all_labels[i]),
            "true_label_name": label_names[int(all_labels[i])],
            "pred_label": int(all_preds[i]),
            "pred_label_name": label_names[int(all_preds[i])],
            "correct": bool(all_preds[i] == all_labels[i]),
            "concept_scores": [round(float(s), 6) for s in all_concept_scores[i]],
        })
    with open(test_results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PGL-CBM 训练与测试统一流水线",
    )

    parser.add_argument('--mode', type=str, choices=['all', 'train', 'test'],
                        default='all')
    parser.add_argument('--timestamp', type=str, default=None)

    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--plm_name', type=str, default=None)

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_deterministic', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--plm_lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--warmup_ratio', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--concept_loss_weight', type=float, default=None)

    parser.add_argument('--freeze_plm', action='store_true', default=False)
    parser.add_argument('--no_homophone_restore', action='store_true', default=False)
    parser.add_argument('--no_pinyin', action='store_true', default=False)
    parser.add_argument('--no_glyph', action='store_true', default=False)
    parser.add_argument('--no_lexicon', action='store_true', default=False)

    return parser.parse_args()


def update_config(args):
    config = PGLCBMConfig()

    config.dataset_name = args.dataset_name
    if args.plm_name is not None:
        config.plm_name = args.plm_name

    if args.seed is not None:
        config.seed = args.seed
    if args.use_deterministic:
        config.use_deterministic = True

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.plm_lr is not None:
        config.plm_lr = args.plm_lr
    if args.patience is not None:
        config.patience = args.patience
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        config.warmup_ratio = args.warmup_ratio
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.concept_loss_weight is not None:
        config.concept_loss_weight = args.concept_loss_weight

    if args.freeze_plm:
        config.freeze_plm = True
    if args.no_homophone_restore:
        config.use_homophone_restore = False

    config.use_pinyin = not args.no_pinyin
    config.use_glyph = not args.no_glyph
    config.use_lexicon = not args.no_lexicon

    return config


def main():
    args = parse_args()

    if args.mode in ['all', 'train']:
        config = update_config(args)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = config.experiment_path / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.experiment_path = experiment_dir

        config_dict = {}
        for key, value in sorted(config.__dict__.items()):
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        config_dict['timestamp'] = timestamp

        with open(experiment_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f">>> 配置文件已保存至: {experiment_dir / 'config.json'}\n")

        if config.use_deterministic:
            from utils.seed import set_reproducibility
            set_reproducibility(config)
            print(">>> 已启用确定性模式 (Reproducibility Enabled)")
        else:
            print(">>> 已禁用确定性模式 (Randomness Enabled), 结果将不可复现")

        with open(config.raw_data_path / config.dataset_name / "train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(config.raw_data_path / config.dataset_name / "test.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_texts = [item['content'] for item in train_data]
        train_labels = [item['toxic'] for item in train_data]
        train_indices, val_indices = train_test_split(
            range(len(train_data)),
            test_size=0.1,
            stratify=train_labels,
            random_state=config.seed,
        )
        val_data = [train_data[i] for i in val_indices]
        train_split_data = [train_data[i] for i in train_indices]

        print(f">>> 训练集: {len(train_split_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

        tokenizer = AutoTokenizer.from_pretrained(config.plm_name)
        pinyin_processor = PinyinProcessor(max_length=config.max_length)
        glyph_processor = GlyphProcessor(max_length=config.max_length)
        lexicon_matcher = LexiconMatcher(config.toxic_lexicon_path)
        homophone_restorer = HomophoneRestorer(config.homo_graph_path, config.toxic_lexicon_path)

        train_dataset = PGLCBMDataset(
            train_split_data, tokenizer, pinyin_processor, glyph_processor,
            lexicon_matcher, homophone_restorer, config.max_length,
            config.use_homophone_restore
        )
        val_dataset = PGLCBMDataset(
            val_data, tokenizer, pinyin_processor, glyph_processor,
            lexicon_matcher, homophone_restorer, config.max_length,
            config.use_homophone_restore
        )
        test_dataset = PGLCBMDataset(
            test_data, tokenizer, pinyin_processor, glyph_processor,
            lexicon_matcher, homophone_restorer, config.max_length,
            config.use_homophone_restore
        )

        concept_labels_path = (
            config.processed_path / config.dataset_name /
            "Qwen2.5-7B-Instruct-AWQ" / "rcwn_concepts" / "concept_train.json"
        )
        if concept_labels_path.exists():
            with open(concept_labels_path, 'r', encoding='utf-8') as f:
                concept_data = json.load(f)
            concept_labels = torch.tensor(
                [item['concept'] for item in concept_data], dtype=torch.float32
            )
            train_concept_labels = concept_labels[train_indices]
            train_dataset.concept_labels = train_concept_labels
            print(f">>> 已加载概念标签: {train_concept_labels.shape}")
        else:
            train_dataset.concept_labels = None
            print(">>> 未找到概念标签文件，不使用概念监督")

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> 正在使用设备: {device}")

        model = PGLCBMModel(
            plm_name=config.plm_name,
            pinyin_vocab_size=pinyin_processor.get_vocab_size(),
            pinyin_dim=config.pinyin_output_dim,
            glyph_input_dim=config.glyph_input_dim,
            glyph_dim=config.glyph_output_dim,
            lexicon_size=lexicon_matcher.lexicon_size,
            lexicon_dim=config.lexicon_dim,
            num_concepts=config.num_concepts,
            use_pinyin=getattr(config, 'use_pinyin', True),
            use_glyph=getattr(config, 'use_glyph', True),
            use_lexicon=getattr(config, 'use_lexicon', True),
            concept_loss_weight=config.concept_loss_weight,
        )

        if config.freeze_plm:
            for param in model.plm.parameters():
                param.requires_grad = False
            print(">>> 已冻结PLM参数")

        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>> 模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

        metrics = train(config, model, train_loader, val_loader, test_loader, device)
        plot_metrics(config, *metrics)

        if args.mode == 'all':
            evaluate(config, timestamp)

    elif args.mode == 'test':
        if not args.timestamp:
            print("错误: 测试模式必须指定 --timestamp")
            sys.exit(1)
        config = PGLCBMConfig()
        evaluate(config, args.timestamp)


if __name__ == '__main__':
    main()
