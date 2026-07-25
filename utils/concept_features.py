"""概念向量特征提取工具。

根据concept_feat_mode从level_probs中提取不同维度的特征：
  - "single": 二元P(2), 3级P(3) → 每概念1维
  - "conditional": 二元P(2), 3级[P(3), P(3)/(P(2)+P(3))] → 二元1维, 3级2维
  - "all_probs": 二元[P(1),P(2)], 3级[P(1),P(2),P(3)] → 二元2维, 3级3维
"""

import numpy as np

THREE_LEVEL_TYPES = {"evaluation", "discrimination", "neutral"}
BINARY_TYPES = {"behavior", "strategy", "intent", "effect"}


def extract_concept_features(data, concept_types, mode="conditional"):
    """从概念向量数据中提取特征矩阵。

    Args:
        data: list of dict, 每个item包含"level_probs"和"toxic"
        concept_types: list of str, 每个概念的prompt_template类型
        mode: 特征提取模式

    Returns:
        X: np.array of shape (n_samples, n_features)
        y: np.array of shape (n_samples,)
        feature_names: list of str, 特征名列表
    """
    n_samples = len(data)
    n_concepts = len(concept_types)

    features = []
    feature_names = []

    for ci, ptype in enumerate(concept_types):
        if ptype in THREE_LEVEL_TYPES:
            if mode == "single":
                feature_names.append(f"P3_{ci}")
            elif mode == "conditional":
                feature_names.append(f"P3_{ci}")
                feature_names.append(f"cond_{ci}")
            elif mode == "all_probs":
                feature_names.append(f"P1_{ci}")
                feature_names.append(f"P2_{ci}")
                feature_names.append(f"P3_{ci}")
        else:
            if mode == "single":
                feature_names.append(f"P2_{ci}")
            elif mode == "conditional":
                feature_names.append(f"P2_{ci}")
            elif mode == "all_probs":
                feature_names.append(f"P1_{ci}")
                feature_names.append(f"P2_{ci}")

    X = np.zeros((n_samples, len(feature_names)))

    for si, item in enumerate(data):
        col = 0
        for ci, ptype in enumerate(concept_types):
            probs = item["level_probs"][ci]

            if ptype in THREE_LEVEL_TYPES:
                p1, p2, p3 = probs[0], probs[1], probs[2]
                if mode == "single":
                    X[si, col] = p3
                    col += 1
                elif mode == "conditional":
                    X[si, col] = p3
                    col += 1
                    denom = p2 + p3 + 1e-8
                    X[si, col] = p3 / denom
                    col += 1
                elif mode == "all_probs":
                    X[si, col] = p1
                    col += 1
                    X[si, col] = p2
                    col += 1
                    X[si, col] = p3
                    col += 1
            else:
                p1, p2 = probs[0], probs[1]
                if mode == "single":
                    X[si, col] = p2
                    col += 1
                elif mode == "conditional":
                    X[si, col] = p2
                    col += 1
                elif mode == "all_probs":
                    X[si, col] = p1
                    col += 1
                    X[si, col] = p2
                    col += 1

    y = np.array([item["toxic"] for item in data])

    return X, y, feature_names


def get_feature_dim(concept_types, mode="conditional"):
    """计算给定概念类型和模式下的特征维度。"""
    dim = 0
    for ptype in concept_types:
        if ptype in THREE_LEVEL_TYPES:
            if mode == "single":
                dim += 1
            elif mode == "conditional":
                dim += 2
            elif mode == "all_probs":
                dim += 3
        else:
            if mode == "single":
                dim += 1
            elif mode == "conditional":
                dim += 1
            elif mode == "all_probs":
                dim += 2
    return dim
