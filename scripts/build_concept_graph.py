"""
构建HECD-Net概念层次图

基于discovered_concepts.csv中的56个概念，按4层结构组织：
- Layer 1 (情感): 愤怒的, 厌恶的, 仇恨的, 怨恨的, 恶意的
- Layer 2 (行为): 辱骂的, 威胁的, 歧视的, 贬低的, 嘲笑的, 蔑视的, 煽动性的, 挑拨的, 引战的, 侮辱性的, 攻击性的, 挑衅性的, 物化的, 标签化的, 带节奏的, 羞辱性的, 骚扰性的, 暴力的, 极端的, 非人化的, 恐吓的
- Layer 3 (目标): 针对个人的, 针对群体的, 地域黑的, 性别偏见的, 种族偏见的, 排外的, 刻板的, 排斥的, 性骚扰的, 物化女性的, 厌女的, 民族主义的, 受害者有罪论的
- Layer 4 (策略): 直接的, 隐晦的, 反讽的, 阴阳怪气的, 夸张的, 扣帽子的, 捧杀的, 站队的, 冒犯的, 粗俗的, 下流的, 破坏性的, 分裂性的, 误导性的, 伪关心的, 煤气灯操纵的

输出: data/processed/concept_graph.json
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


CONCEPT_LAYERS = {
    "affective": [
        "愤怒的", "厌恶的", "仇恨的", "怨恨的", "恶意的"
    ],
    "behavioral": [
        "攻击性的", "侮辱性的", "威胁性的", "辱骂的", "挑衅性的",
        "贬低的", "嘲笑的", "蔑视的", "物化的", "标签化的",
        "煽动性的", "挑拨的", "引战的", "带节奏的", "羞辱性的",
        "骚扰性的", "暴力的", "极端的", "非人化的", "恐吓的"
    ],
    "target": [
        "刻板的", "排斥的", "针对个人的", "针对群体的", "地域黑的",
        "性别偏见的", "性骚扰的", "物化女性的", "厌女的", "种族偏见的",
        "排外的", "民族主义的", "受害者有罪论的"
    ],
    "strategic": [
        "直接的", "隐晦的", "反讽的", "阴阳怪气的", "夸张的",
        "扣帽子的", "捧杀的", "站队的", "冒犯的", "粗俗的",
        "下流的", "破坏性的", "分裂性的", "误导性的", "伪关心的",
        "煤气灯操纵的"
    ]
}

# 概念间的先验边关系 (source -> [targets])
# 基于语义因果和层次依赖定义
CONCEPT_EDGES = {
    # 情感 -> 行为
    "愤怒的": ["辱骂的", "威胁的", "攻击性的", "挑衅性的"],
    "厌恶的": ["贬低的", "嘲笑的", "蔑视的", "侮辱性的"],
    "仇恨的": ["歧视性的", "煽动性的", "暴力的", "极端的", "恐吓的"],
    "怨恨的": ["挑拨的", "非人化的", "羞辱性的"],
    "恶意的": ["误导性的", "伪关心的", "煤气灯操纵的", "骚扰性的"],

    # 行为 -> 目标
    "辱骂的": ["针对个人的"],
    "威胁性的": ["针对个人的", "恐吓的"],
    "歧视性的": ["针对群体的", "性别偏见的", "种族偏见的", "排外的"],
    "贬低的": ["针对个人的", "针对群体的", "地域黑的"],
    "嘲笑的": ["针对个人的", "物化的"],
    "蔑视的": ["针对群体的", "刻板的", "排斥的"],
    "物化的": ["物化女性的", "性骚扰的"],
    "标签化的": ["扣帽子的", "刻板的"],
    "煽动性的": ["引战的", "带节奏的", "极端的"],
    "挑拨的": ["引战的", "分裂性的"],
    "非人化的": ["种族偏见的", "排外的", "民族主义的"],
    "恐吓的": ["针对个人的", "骚扰性的"],

    # 策略 -> 行为
    "直接的": ["辱骂的", "威胁性的", "攻击性的", "暴力的", "粗俗的"],
    "隐晦的": ["贬低的", "挑拨的", "误导性的", "伪关心的"],
    "反讽的": ["嘲笑的", "侮辱性的", "捧杀的"],
    "阴阳怪气的": ["嘲笑的", "蔑视的", "贬低的"],
    "夸张的": ["煽动性的", "误导性的", "极端的"],
    "扣帽子的": ["标签化的", "刻板的"],
    "捧杀的": ["反讽的", "贬低的"],
    "站队的": ["煽动性的", "分裂性的"],
    "破坏性的": ["攻击性的", "暴力的", "极端的"],
    "煤气灯操纵的": ["恶意的", "隐晦的", "羞辱性的"],
}


def build_concept_graph(concepts_csv_path: Path, output_path: Path):
    import csv
    concepts = []
    with open(concepts_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concepts.append(row["name"])

    # 建立概念名 -> 索引映射
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    num_concepts = len(concepts)

    # 验证所有概念都被覆盖
    all_layered = set()
    for layer_concepts in CONCEPT_LAYERS.values():
        all_layered.update(layer_concepts)

    missing = set(concepts) - all_layered
    extra = all_layered - set(concepts)
    if missing:
        print(f"Warning: concepts in CSV but not in layers: {missing}")
    if extra:
        print(f"Warning: concepts in layers but not in CSV: {extra}")

    # 构建 concept -> layer 映射
    concept_to_layer = {}
    for layer_name, layer_concepts in CONCEPT_LAYERS.items():
        for c in layer_concepts:
            if c in concept_to_idx:
                concept_to_layer[c] = layer_name

    # 为未分类的概念分配默认层
    for c in concepts:
        if c not in concept_to_layer:
            concept_to_layer[c] = "uncategorized"
            print(f"Note: '{c}' assigned to 'uncategorized'")

    # 构建边列表 (source_idx, target_idx, edge_type)
    edges = []

    # 1. 先验语义边
    for src_name, tgt_names in CONCEPT_EDGES.items():
        if src_name not in concept_to_idx:
            continue
        src_idx = concept_to_idx[src_name]
        for tgt_name in tgt_names:
            if tgt_name not in concept_to_idx:
                continue
            tgt_idx = concept_to_idx[tgt_name]
            edges.append({
                "source": src_idx,
                "target": tgt_idx,
                "source_name": src_name,
                "target_name": tgt_name,
                "edge_type": "semantic",
                "layer_relation": f"{concept_to_layer[src_name]}->{concept_to_layer[tgt_name]}"
            })

    # 2. 层次顺序边 (低层 -> 高层，用于GAT传播)
    layer_order = ["affective", "behavioral", "target", "strategic"]
    for i in range(len(layer_order) - 1):
        lower_layer = layer_order[i]
        higher_layer = layer_order[i + 1]
        lower_concepts = [c for c in CONCEPT_LAYERS[lower_layer] if c in concept_to_idx]
        higher_concepts = [c for c in CONCEPT_LAYERS[higher_layer] if c in concept_to_idx]
        for lc in lower_concepts:
            for hc in higher_concepts:
                # 避免重复添加已存在的语义边
                existing = any(
                    e["source"] == concept_to_idx[lc] and e["target"] == concept_to_idx[hc]
                    for e in edges
                )
                if not existing:
                    edges.append({
                        "source": concept_to_idx[lc],
                        "target": concept_to_idx[hc],
                        "source_name": lc,
                        "target_name": hc,
                        "edge_type": "hierarchical",
                        "layer_relation": f"{lower_layer}->{higher_layer}"
                    })

    # 3. 同层共现边 (可选，基于语义相似性)
    # 简化为：同层内所有概念两两相连（形成全连接子图，GAT会学习权重）
    for layer_name, layer_concepts in CONCEPT_LAYERS.items():
        valid_concepts = [c for c in layer_concepts if c in concept_to_idx]
        for i, c1 in enumerate(valid_concepts):
            for c2 in valid_concepts[i + 1:]:
                edges.append({
                    "source": concept_to_idx[c1],
                    "target": concept_to_idx[c2],
                    "source_name": c1,
                    "target_name": c2,
                    "edge_type": "intra_layer",
                    "layer_relation": f"{layer_name}<->{layer_name}"
                })
                # 双向
                edges.append({
                    "source": concept_to_idx[c2],
                    "target": concept_to_idx[c1],
                    "source_name": c2,
                    "target_name": c1,
                    "edge_type": "intra_layer",
                    "layer_relation": f"{layer_name}<->{layer_name}"
                })

    graph_data = {
        "num_concepts": num_concepts,
        "concepts": concepts,
        "concept_to_idx": concept_to_idx,
        "concept_to_layer": concept_to_layer,
        "layers": CONCEPT_LAYERS,
        "edges": edges,
        "num_edges": len(edges)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"Concept graph built: {num_concepts} concepts, {len(edges)} edges")
    print(f"Saved to: {output_path}")

    # 统计
    edge_type_counts = {}
    for e in edges:
        et = e["edge_type"]
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
    print(f"Edge type distribution: {edge_type_counts}")


if __name__ == "__main__":
    concepts_csv = project_root / "data" / "raw" / "adjective" / "discovered_concepts.csv"
    output = project_root / "data" / "processed" / "concept_graph.json"
    build_concept_graph(concepts_csv, output)
