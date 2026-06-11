"""将语用推理结果编码为BGE嵌入

读取 generate_pragmatic_reasoning.py 的输出，对每个维度的推理文本
（concept + reason）用BGE-base-zh-v1.5编码为768维向量。

输出格式：.pt文件，包含7个维度的BGE嵌入，shape=[N, 7, 768]

使用示例：
python scripts/encode_reasoning_bge.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.PCCG_config import PCCGConfig

DIMENSION_NAMES = [
    "expression_strategy", "implicit_intent", "encoding_strategy",
    "attack_target", "emotional_tone", "pragmatic_effect", "topic_distinction"
]


def encode_texts(texts, tokenizer, model, batch_size=64, device="cuda"):
    """批量编码文本为BGE嵌入

    Args:
        texts: 文本列表
        tokenizer: BGE tokenizer
        model: BGE model
        batch_size: 批次大小
        device: 设备

    Returns:
        Tensor: [N, 768] 嵌入矩阵
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            # 使用[CLS]的last hidden state作为句子嵌入
            cls_embeddings = outputs.last_hidden_state[:, 0]
            # BGE推荐：归一化
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="将语用推理结果编码为BGE嵌入")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    config = PCCGConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 路径
    reasoning_path = (config.processed_path / args.dataset_name / args.model_name
                      / f"pragmatic_reasoning_{args.mode}.json")
    output_path = (config.processed_path / args.dataset_name / args.model_name
                   / f"bge_reasoning_{args.mode}.pt")

    if not reasoning_path.exists():
        print(f"错误: 语用推理文件不存在: {reasoning_path}")
        print("请先运行 generate_pragmatic_reasoning.py")
        sys.exit(1)

    print(f"加载语用推理结果: {reasoning_path}")
    with open(reasoning_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 加载BGE模型
    bge_path = config.models_path / "bge-base-zh-v1.5"
    print(f"加载BGE模型: {bge_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(bge_path))
    model = AutoModel.from_pretrained(str(bge_path)).to(device).eval()

    # 为每个维度构建编码文本并编码
    all_dim_embeddings = []

    for dim_name in DIMENSION_NAMES:
        texts = []
        for item in data:
            reasoning = item["reasoning"]
            dim_result = reasoning.get(dim_name, {})
            concept = dim_result.get("concept", "未知")
            reason = dim_result.get("reason", "未知")
            # 拼接概念和理由作为编码文本
            encode_text = f"{concept}：{reason}"
            texts.append(encode_text)

        print(f"编码维度: {dim_name} ({len(texts)} 条)")
        embeddings = encode_texts(texts, tokenizer, model, args.batch_size, device)
        all_dim_embeddings.append(embeddings)

    # 堆叠为 [N, 7, 768]
    all_embeddings = torch.stack(all_dim_embeddings, dim=1)
    print(f"嵌入矩阵形状: {all_embeddings.shape}")

    # 保存
    torch.save(all_embeddings, output_path)
    print(f"BGE嵌入保存到: {output_path}")


if __name__ == '__main__':
    main()
