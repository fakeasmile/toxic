import json
import torch


class CodedTermRecognizer:
    def __init__(self, lexicon_path, homo_graph_path=None):
        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base_terms = [t.strip() for t in data["words"] if t.strip()]

        self.terms = list(base_terms)
        self.base_term_count = len(base_terms)

        if homo_graph_path is not None:
            char_to_variants = self._build_char_variants(homo_graph_path)
            augmented = set(base_terms)
            for term in base_terms:
                variants = self._generate_variants(term, char_to_variants)
                augmented.update(variants)
            self.terms = list(augmented)

    def _build_char_variants(self, homo_graph_path):
        with open(homo_graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        char_to_variants = {}
        for pinyin, chars in data["pinyin_to_chars"].items():
            if len(chars) < 2:
                continue
            for char in chars:
                if char not in char_to_variants:
                    char_to_variants[char] = set(chars)
                else:
                    char_to_variants[char].update(chars)
        return char_to_variants

    def _generate_variants(self, term, char_to_variants, max_variants=5):
        if not term:
            return set()
        all_variants = [set() for _ in range(len(term))]
        for i, ch in enumerate(term):
            if ch in char_to_variants:
                all_variants[i].update(char_to_variants[ch])
            else:
                all_variants[i].add(ch)

        import itertools
        variants = set()
        for combo in itertools.islice(itertools.product(*all_variants), max_variants):
            v = "".join(combo)
            if v != term:
                variants.add(v)
        return variants

    def forward(self, texts):
        features = torch.zeros(len(texts), len(self.terms), dtype=torch.float)
        for i, text in enumerate(texts):
            for j, term in enumerate(self.terms):
                if term in text:
                    features[i, j] = 1.0
        return features

    def match_terms(self, text):
        matched = []
        for term in self.terms[:self.base_term_count]:
            if term in text:
                matched.append(term)
        return matched
