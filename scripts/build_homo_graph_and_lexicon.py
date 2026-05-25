import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    return parser.parse_args()


def get_dialect_variants(pinyin):
    variants = set()
    if pinyin.startswith('n'):
        variants.add('l' + pinyin[1:])
    elif pinyin.startswith('l'):
        variants.add('n' + pinyin[1:])

    if pinyin.startswith('zh'):
        variants.add('z' + pinyin[2:])
    elif pinyin.startswith('z'):
        variants.add('zh' + pinyin[1:])

    if pinyin.startswith('ch'):
        variants.add('c' + pinyin[2:])
    elif pinyin.startswith('c'):
        variants.add('ch' + pinyin[1:])

    if pinyin.startswith('sh'):
        variants.add('s' + pinyin[2:])
    elif pinyin.startswith('s'):
        variants.add('sh' + pinyin[1:])

    if pinyin.endswith('ng'):
        variants.add(pinyin[:-1])

    return variants


def build_homo_graph():
    pinyin_to_chars = {}
    char_to_pinyin = {}

    for code in range(0x4E00, 0x9FFF + 1):
        char = chr(code)
        pinyins = pypinyin.pinyin(char, style=pypinyin.NORMAL)
        char_pinyins = list(set(p[0] for p in pinyins if p[0]))

        if not char_pinyins:
            continue

        all_pinyins = set(char_pinyins)
        for p in char_pinyins:
            all_pinyins.update(get_dialect_variants(p))

        char_to_pinyin[char] = sorted(all_pinyins)

        for p in all_pinyins:
            if p not in pinyin_to_chars:
                pinyin_to_chars[p] = set()
            pinyin_to_chars[p].add(char)

    pinyin_to_chars = {k: sorted(v) for k, v in pinyin_to_chars.items()}

    return pinyin_to_chars, char_to_pinyin


def build_toxic_lexicon(dataset_name):
    data_path = project_root / 'data' / 'raw' / dataset_name / 'train.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    toxic_texts = [item['content'] for item in data if item.get('toxic') == 1]

    word_counter = Counter()
    for text in toxic_texts:
        words = jieba.lcut(text)
        words = [w.strip() for w in words if len(w.strip()) >= 2]
        word_counter.update(words)

    toxic_words = [w for w, c in word_counter.most_common() if c >= 2]
    word_to_idx = {w: i for i, w in enumerate(toxic_words)}

    return {'words': toxic_words, 'word_to_idx': word_to_idx}


def main():
    args = parse_args()

    pinyin_to_chars, char_to_pinyin = build_homo_graph()
    homo_graph = {
        'pinyin_to_chars': pinyin_to_chars,
        'char_to_pinyin': char_to_pinyin,
    }

    toxic_lexicon = build_toxic_lexicon(args.dataset_name)

    output_dir = project_root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'homo_graph.json', 'w', encoding='utf-8') as f:
        json.dump(homo_graph, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'toxic_lexicon.json', 'w', encoding='utf-8') as f:
        json.dump(toxic_lexicon, f, ensure_ascii=False, indent=2)

    print(f"Homo-graph: {len(pinyin_to_chars)} pinyin groups, {len(char_to_pinyin)} characters")
    print(f"Toxic lexicon: {len(toxic_lexicon['words'])} words")


if __name__ == '__main__':
    main()
