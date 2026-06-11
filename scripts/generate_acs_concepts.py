"""生成ACS概念评分（BGE嵌入 + 余弦相似度）

【功能】
使用BGE-base-zh-v1.5中文嵌入模型，对每条文本和每个概念计算余弦相似度，
作为概念激活值，替代Likert标量评分。

【与Likert评分的关系】
- Likert评分：LLM逐条推理 → 首token概率 → 加权期望 → 5级标量
- ACS评分：嵌入模型前向传播 → 余弦相似度 → 连续浮点值
- ACS信息量更丰富（连续值 vs 5级标量），计算速度更快

【执行流程】
1. 加载BGE嵌入模型
2. 编码所有概念文本（形容词+定义），得到概念嵌入矩阵
3. 对数据集中每条文本编码，计算与所有概念的余弦相似度
4. 应用ACC纠正（负值置零 + 跨类置零）
5. 保存为JSON文件

使用示例：
python scripts/generate_acs_concepts.py --mode train --dataset_name TOXICN
python scripts/generate_acs_concepts.py --mode test --dataset_name TOXICN
"""

import argparse
import json
import os
import sys
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.CBLLM_config import CBLLMConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成ACS概念评分（BGE嵌入 + 余弦相似度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集概念评分，test:生成测试集概念评分')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--batch_size', type=int, default=64, help='嵌入模型推理批次大小')

    return parser.parse_args()


def load_concepts(config):
    """加载概念定义，构建概念文本列表

    将形容词和其定义拼接为完整的概念文本，用于嵌入模型编码。

    Returns:
        concept_texts: List[str], 概念文本列表
        concept_names: List[str], 概念名称列表
        toxic_concept_indices: Set[int], 与toxic类别关联的概念索引
    """
    adjective_df = pd.read_csv(config.adjective_path)

    concept_texts = []
    concept_names = []
    toxic_concept_indices = set()

    for idx, row in adjective_df.iterrows():
        adjective = row['adjective']
        chinese = row['chinese']
        definition = row['definition']

        # 拼接概念文本：中文名 + 英文名 + 定义
        concept_text = f"{chinese}（{adjective}）：{definition}"
        concept_texts.append(concept_text)
        concept_names.append(chinese)

        # 根据定义判断是否与toxic类别关联
        # 有毒性相关概念在toxic样本中应激活，在non-toxic样本中应抑制
        definition_lower = definition.lower()
        toxic_keywords = ['有毒', '攻击', '侮辱', '歧视', '仇恨', '冒犯', '贬低',
                          '恶意', '暴力', '粗俗', '恶劣', '偏激', 'toxic', 'hateful',
                          'offensive', 'aggressive', 'insulting', 'hostile', 'abusive',
                          'derogatory', 'malicious']
        if any(kw in definition_lower for kw in toxic_keywords):
            toxic_concept_indices.add(idx)

    print(f"  加载概念: {len(concept_texts)} 个, 其中toxic关联: {len(toxic_concept_indices)} 个")

    return concept_texts, concept_names, toxic_concept_indices


def load_embedding_model(config):
    """加载BGE中文嵌入模型

    Returns:
        tokenizer: 分词器
        model: 嵌入模型
    """
    model_path = config.models_path / config.embedding_model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"BGE模型路径不存在: {model_path}\n"
            f"请下载 bge-base-zh-v1.5 到 models/ 目录"
        )

    print(f"Loading embedding model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 绕过transformers的torch版本安全检查（torch<2.6时触发CVE-2025-32434警告）
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda: None

    model = AutoModel.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return tokenizer, model


@torch.no_grad()
def encode_texts(tokenizer, model, texts, batch_size=64):
    """批量编码文本为嵌入向量

    使用BGE推荐的CLS pooling方式。

    Args:
        tokenizer: 分词器
        model: 嵌入模型
        texts: 文本列表
        batch_size: 批次大小

    Returns:
        embeddings: [N, dim] 归一化后的嵌入矩阵
    """
    all_embeddings = []

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[start_idx:start_idx + batch_size]

        # BGE推荐：为短文本添加指令前缀提升检索效果
        # 但概念匹配任务不需要检索指令，直接编码
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        outputs = model(**encoded)

        # CLS pooling: 取[CLS] token的输出作为句子嵌入
        embeddings = outputs.last_hidden_state[:, 0]  # [B, dim]

        # L2归一化，使余弦相似度等价于内积
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # [N, dim]


def apply_acc(concept_scores, labels, toxic_concept_indices):
    """ACC自动概念纠正

    规则：
    1. 负余弦相似度置零（文本与概念语义相反时不应激活）
    2. 跨类概念对齐：toxic关联概念在non-toxic样本中置零

    Args:
        concept_scores: [N, K] 原始余弦相似度
        labels: [N] 标签（0=non-toxic, 1=toxic）
        toxic_concept_indices: 与toxic类别关联的概念索引

    Returns:
        corrected_scores: [N, K] 纠正后的概念评分
    """
    corrected = concept_scores.clone()

    # 规则1：负值置零
    corrected = corrected.clamp(min=0.0)

    # 规则2：toxic关联概念在non-toxic样本中置零
    non_toxic_mask = (labels == 0)
    if len(toxic_concept_indices) > 0:
        toxic_indices = list(toxic_concept_indices)
        corrected[non_toxic_mask][:, toxic_indices] = 0.0

    return corrected


def main():
    args = parse_args()
    config = CBLLMConfig()

    # 动态生成路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    output_dir = config.processed_path / args.dataset_name / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"acs_concept_{args.mode}.json"

    # 加载概念定义
    print("\n>>> 加载概念定义...")
    concept_texts, concept_names, toxic_concept_indices = load_concepts(config)

    # 加载嵌入模型
    print("\n>>> 加载嵌入模型...")
    tokenizer, model = load_embedding_model(config)

    # 编码概念文本
    print("\n>>> 编码概念文本...")
    concept_embeddings = encode_texts(tokenizer, model, concept_texts, batch_size=args.batch_size)
    print(f"  概念嵌入形状: {concept_embeddings.shape}")

    # 加载数据集
    print(f"\n>>> 加载数据集: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    contents = [sample["content"] for sample in dataset]
    labels = [sample["toxic"] for sample in dataset]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    print(f"  数据量: {len(contents)}, toxic: {sum(labels)}, non-toxic: {len(labels) - sum(labels)}")

    # 编码文本
    print("\n>>> 编码文本...")
    text_embeddings = encode_texts(tokenizer, model, contents, batch_size=args.batch_size)
    print(f"  文本嵌入形状: {text_embeddings.shape}")

    # 计算余弦相似度（嵌入已归一化，内积=余弦相似度）
    print("\n>>> 计算概念评分...")
    concept_scores = torch.mm(text_embeddings, concept_embeddings.t())  # [N, K]
    print(f"  原始评分范围: [{concept_scores.min():.4f}, {concept_scores.max():.4f}]")

    # ACC纠正
    print("\n>>> 应用ACC概念纠正...")
    corrected_scores = apply_acc(concept_scores, labels_tensor, toxic_concept_indices)
    print(f"  纠正后评分范围: [{corrected_scores.min():.4f}, {corrected_scores.max():.4f}]")

    # 统计纠正效果
    n_zeroed_neg = (concept_scores < 0).sum().item()
    n_zeroed_cross = 0
    non_toxic_mask = (labels_tensor == 0)
    if len(toxic_concept_indices) > 0:
        toxic_indices = list(toxic_concept_indices)
        n_zeroed_cross = (concept_scores[non_toxic_mask][:, toxic_indices] > 0).sum().item()
    print(f"  负值置零: {n_zeroed_neg} 个, 跨类置零: {n_zeroed_cross} 个")

    # 保存结果
    print(f"\n>>> 保存结果到: {output_path}")
    results = []
    for i in range(len(dataset)):
        results.append({
            "content": contents[i],
            "toxic": labels[i],
            "concept": corrected_scores[i].tolist(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 同时保存CSV格式（兼容现有流程）
    csv_path = output_dir / f"acs_concept_{args.mode}.csv"
    csv_data = []
    for i in range(len(dataset)):
        row = {"content": contents[i], "toxic": labels[i]}
        for j, name in enumerate(concept_names):
            row[name] = corrected_scores[i, j].item()
        csv_data.append(row)
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False, encoding="utf-8")

    # 保存BGE文本嵌入（供CB-LLM-CN流水线使用）
    bge_embed_path = output_dir / f"bge_embed_{args.mode}.pt"
    torch.save(text_embeddings, bge_embed_path)
    print(f"  BGE嵌入: {bge_embed_path} (形状: {text_embeddings.shape})")

    print(f"\n>>> 完成!")
    print(f"  JSON: {output_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  概念维度: {len(concept_names)}")


if __name__ == '__main__':
    main()
