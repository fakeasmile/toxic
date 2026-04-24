"""生成形容词概念向量（基于肯定词概率和）

使用示例：
python scripts/generate_adjective_c_r.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template binary
"""
import argparse
import sys
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成形容词概念向量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="使用示例:"
               "python scripts/generate_adjective_c_r.py --mode {train/test} --dataset_name TOXICN --model_name ..."
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        default='test',
        help='train:生成训练集的形容词概念向量，test:生成测试集的形容词概念向量'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='数据集名称(TOXICN/COLD)'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='LLM模型名称'
    )

    parser.add_argument(
        '--template',
        type=str,
        choices=['binary', 'likert'],
        default='binary',
        help='提示词模板类型：binary=二元判断(原始), likert=Likert程度量化'
    )

    return parser.parse_args()

def load_qwen_model(model_path: Path, model_name: str):
    """加载模型和分词器"""
    llm_path = model_path / model_name  # LLM模型路径
    # 判断llm目录是否存在
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit量化
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  #
        bnb_4bit_compute_dtype=torch.float16,  # 半精度
    )

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",  # PAD右填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Qwen model from {llm_path}")
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def get_first_token_ids(word_list, tokenizer, device):
    """对词表中每个词，用分词器编码得到关于这个词的token id（可能某个词对应多个token id），只记录第一个token id（并且去重后返回）。"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return torch.tensor(sorted(set(token_ids)), device=device, dtype=torch.long)


def _expand_prefix_cache(base_cache, batch_size: int):
    """
    将单样本 prefix cache:[1, num_heads, L_prefix, head_dim] 扩展到 batch 维度。
    同时兼容 legacy tuple cache 与 DynamicCache。
    """
    if isinstance(base_cache, DynamicCache):
        legacy_cache = base_cache.to_legacy_cache()
        expanded = DynamicCache()
        for layer_idx, kv in enumerate(legacy_cache):
            k, v = kv[0], kv[1]
            k_expanded = k.expand(batch_size, -1, -1, -1).contiguous()  # [batch_size, num_heads, L_prefix, head_dim]
            v_expanded = v.expand(batch_size, -1, -1, -1).contiguous()
            expanded.update(k_expanded, v_expanded, layer_idx)
        return expanded

    expanded_layers = []
    for kv in base_cache:
        k, v = kv[0], kv[1]
        k_expanded = k.expand(batch_size, -1, -1, -1).contiguous()
        v_expanded = v.expand(batch_size, -1, -1, -1).contiguous()
        expanded_layers.append((k_expanded, v_expanded))
    return tuple(expanded_layers)


def generate_adj_concept(data_path, output_path, adjective_path, mode, template, tokenizer, model):
    device = next(model.parameters()).device

    # 根据模板类型定义verbalizer token（首token id集合）
    if template == "binary":
        affirmative_tokens = ["是", "是的", "对", "准确", "正确", "Yes", "yes"]
        negative_tokens = ["否", "不", "错误", "偏差", "No", "no"]
        affirmative_ids = get_first_token_ids(affirmative_tokens, tokenizer, device)
        negative_ids = get_first_token_ids(negative_tokens, tokenizer, device)
    elif template == "likert":
        likert_tokens = {1: ["1", "一"], 2: ["2", "二"], 3: ["3", "三"], 4: ["4", "四"], 5: ["5", "五"]}
        likert_ids = {}
        for level in range(1, 6):
            likert_ids[level] = get_first_token_ids(likert_tokens[level], tokenizer, device)

    # 加载形容词词典
    adjectives = pd.read_csv(adjective_path)["chinese"].tolist()

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # 根据模板类型构建提示词指令
    if template == "binary":
        instruction = "你是一个语义分析专家。请判断给定形容词是否准确描述文本，只回答'是'或'否'。"
    elif template == "likert":
        instruction = ("你是一位专业的文本特征分析专家。请评估以下文本具有给定形容词所描述特征的程度。\n"
                       "请严格按照1到5的等级进行评估：\n"
                       "1 = 完全不具有该特征\n"
                       "2 = 略微具有该特征\n"
                       "3 = 中等程度具有该特征\n"
                       "4 = 较强程度具有该特征\n"
                       "5 = 非常强烈地具有该特征")

    results = []
    batch_size = 16  # 形容词批量推理大小

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples"), start=1):
        content = sample["content"]
        concept_vector = []
        raw_probs = []

        # 构建公共前缀（指令+文本内容），计算KV缓存
        prefix_text = f"{instruction}\n文本内容：{content}\n"
        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).to(device)

        with torch.no_grad():
            prefix_outputs = model(**prefix_inputs, use_cache=True)
            base_cache = prefix_outputs.past_key_values
            prefix_len = prefix_inputs["input_ids"].shape[1]

        # 按批次遍历形容词，复用前缀KV缓存
        for i in range(0, len(adjectives), batch_size):
            adj_batch = adjectives[i: i + batch_size]
            curr_bsz = len(adj_batch)  # 当前批次大小

            # 构建当前模板的后缀
            suffix_texts = []
            for adj in adj_batch:
                if template == "binary":
                    suffix_texts.append(f"形容词：'{adj}'描述是否准确？回答：")
                elif template == "likert":
                    suffix_texts.append(f"形容词'{adj}'的程度等级（仅回答数字）：")
            suffix_inputs = tokenizer(
                suffix_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            suffix_len = suffix_inputs["input_ids"].shape[1]

            # position_ids：后缀token的位置从prefix_len开始递增
            position_ids = torch.arange(prefix_len, prefix_len + suffix_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(curr_bsz, -1).contiguous()

            # 扩展前缀缓存到当前batch大小，拼接完整attention_mask
            expanded_cache = _expand_prefix_cache(base_cache, curr_bsz)
            prefix_mask = torch.ones(
                (curr_bsz, prefix_len),
                device=device,
                dtype=suffix_inputs["attention_mask"].dtype,
            )
            full_mask = torch.cat([prefix_mask, suffix_inputs["attention_mask"]], dim=1)

            with torch.no_grad():
                outputs = model(
                    input_ids=suffix_inputs["input_ids"],
                    attention_mask=full_mask,
                    position_ids=position_ids,
                    past_key_values=expanded_cache,
                    use_cache=False,
                )

            logits = outputs.logits
            last_token_indices = suffix_inputs["attention_mask"].sum(dim=1) - 1

            # 对每个形容词，根据模板类型提取概率并打分
            for j, last_idx in enumerate(last_token_indices):
                target_logits = logits[j, last_idx, :]
                probs = torch.softmax(target_logits.float(), dim=-1)  # 得到当前形容词对第一个token的概率分布

                if template == "binary":
                    pos_prob = probs[affirmative_ids].sum()
                    neg_prob = probs[negative_ids].sum()
                    total = pos_prob + neg_prob + 1e-8
                    score = (pos_prob / total).item()
                    raw_probs.append([pos_prob.item(), neg_prob.item()])
                elif template == "likert":
                    weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
                    # level_probs = [0.07, 0.11, 0.20, 0.26, 0.22]
                    level_probs = torch.stack([probs[likert_ids[k]].sum() for k in range(1, 6)])
                    total_level_prob = level_probs.sum() + 1e-8
                    score = (weights * level_probs / total_level_prob).sum().item()
                    raw_probs.append([level_probs[k].item() for k in range(5)])

                concept_vector.append(score)

        # 防御性校验，确保每条文本输出的形容词概念长度与形容词数量一致
        if len(concept_vector) != len(adjectives):
            raise RuntimeError(
                f"concept_vector 长度异常：期望 {len(adjectives)}，实际 {len(concept_vector)}"
            )

        results.append({"content": content, "toxic": sample["toxic"], "concept": concept_vector, "raw_probs": raw_probs})

        # 间隔清理缓存
        if torch.cuda.is_available() and sample_idx % 128 == 0:
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"形容词概念向量保存到: {output_path}")


def main():
    args = parse_args()

    config = MLPConfig()

    # 动态生成依赖 dataset_name/model_name 的路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    concept_dir = config.processed_path / args.dataset_name / args.model_name / args.template
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}.json"

    # 打印配置信息
    print("\n" + "=" * 60)
    print("形容词概念向量生成 - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"提示词模板: {args.template}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    tokenizer, model = load_qwen_model(config.models_path, args.model_name)
    generate_adj_concept(data_path, output_path, config.adjective_path, args.mode, args.template, tokenizer, model)


if __name__ == '__main__':
    main()
