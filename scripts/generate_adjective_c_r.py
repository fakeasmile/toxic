"""生成形容词概念向量（Chat Template版本）

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
               "python scripts/generate_adjective_c_r.py --mode test --dataset_name TOXICN --model_name ... --template binary"
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
        choices=['binary', 'likert', 'ICL'],
        default='binary',
        help='提示词模板类型：binary=二元判断(原始), likert=Likert程度量化, ICL=引入形容词解释'
    )

    return parser.parse_args()


def load_qwen_model(model_path: Path, model_name: str):
    """加载模型和分词器"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
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
    """获取词表中每个词的首token id（目前已保证每个词只会对应一个token id）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return torch.tensor(list(dict.fromkeys(token_ids)), device=device, dtype=torch.long)


def build_chat_messages(template, instruction, content, adj, adj_definition=""):
    """
    根据模板类型构建Chat Template的messages列表。
    与手动拼接方式在语义上等价，但使用标准Chat Template格式。
    """
    if template == "binary":
        user_content = f"文本内容：{content}\n形容词：「{adj}」描述是否准确？回答： "
    elif template == "likert":
        user_content = f"文本内容：{content}\n形容词「{adj}」的程度等级（直接回答数字）： "
    elif template == "ICL":
        user_content = (
            f"文本内容：{content}\n"
            f"形容词「{adj}」的定义：{adj_definition}\n"
            f"根据上述定义，该文本是否表现出该形容词「{adj}」所描述的特征？回答： "
        )
    else:
        raise ValueError(f"不支持的模板类型: {template}")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_adj_concept(data_path, output_path, adjective_path, template, tokenizer, model):
    device = next(model.parameters()).device

    # 根据模板类型定义verbalizer token（首token id集合）和提示词指令
    if template in ["binary", "ICL"]:
        affirmative_tokens = ["是", " 是", "Yes", " Yes", "yes", " yes"]
        negative_tokens = ["否", "不", " 不", "不能", "无", "No", " No", "no", " no"]
        affirmative_ids = get_first_token_ids(affirmative_tokens, tokenizer, device)
        negative_ids = get_first_token_ids(negative_tokens, tokenizer, device)

        if template == "binary":
            instruction = "你是一位社会心理学专家。请判断给定形容词是否准确描述文本，只回答是或否。"
        elif template == "ICL":
            instruction = "请根据形容词的定义，判断该文本是否表现出该形容词所描述的特征，只回答是或否。"
            # 加载形容词解释
            definition = pd.read_csv(adjective_path)["definition"].tolist()

    elif template == "likert":
        likert_tokens = ["1", "2", "3", "4", "5"]
        likert_ids = get_first_token_ids(likert_tokens, tokenizer, device)

        instruction = ("你是一位社会心理学专家。请评估以下文本具有给定形容词所描述特征的程度。\n"
                       "请严格按照1到5的等级进行评估：\n"
                       "1 = 完全不具有该特征\n"
                       "2 = 略微具有该特征\n"
                       "3 = 中等程度具有该特征\n"
                       "4 = 较强程度具有该特征\n"
                       "5 = 非常强烈地具有该特征")

    # 加载形容词词典
    adjectives = pd.read_csv(adjective_path)["chinese"].tolist()

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    results = []
    batch_size = 16  # 形容词批量推理大小

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples"), start=1):
        content = sample["content"]
        concept_vector = []  # 当前批次的形容词向量
        raw_probs = []  # 当前批次的原始概率

        # 按批次遍历形容词，每个形容词使用独立的Chat Template prompt
        for i in range(0, len(adjectives), batch_size):
            adj_batch = adjectives[i: i + batch_size]
            curr_bsz = len(adj_batch)

            # ICL模板：同步获取当前批次的形容词定义
            if template == "ICL":
                def_batch = definition[i: i + batch_size]

            # 为每个形容词构建完整的Chat Template prompt
            prompt_texts = []
            for index, adj in enumerate(adj_batch):
                if template == "ICL":
                    messages = build_chat_messages(template, instruction, content, adj, def_batch[index])
                else:
                    messages = build_chat_messages(template, instruction, content, adj)

                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_texts.append(prompt_text)

            # Batch编码
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)

            # Batch推理
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

            logits = outputs.logits
            last_token_indices = inputs["attention_mask"].sum(dim=1) - 1

            # 对每个形容词，根据模板类型提取概率并打分
            for j, last_idx in enumerate(last_token_indices):
                target_logits = logits[j, last_idx, :]
                probs = torch.softmax(target_logits.float(), dim=-1)

                if template == "binary":
                    pos_prob = probs[affirmative_ids].sum()
                    neg_prob = probs[negative_ids].sum()
                    total = pos_prob + neg_prob + 1e-8
                    score = (pos_prob / total).item()
                    raw_probs.append([pos_prob.item(), neg_prob.item()])
                elif template == "likert":
                    weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
                    level_probs = probs[likert_ids]
                    total_level_prob = level_probs.sum() + 1e-8
                    score = (weights * level_probs / total_level_prob).sum().item()
                    raw_probs.append(level_probs.tolist())
                elif template == "ICL":
                    pos_prob = probs[affirmative_ids].sum()
                    neg_prob = probs[negative_ids].sum()
                    total = pos_prob + neg_prob + 1e-8
                    score = (pos_prob / total).item()
                    raw_probs.append([pos_prob.item(), neg_prob.item()])

                concept_vector.append(score)

        # 防御性校验，确保每条文本输出的形容词概念长度与形容词数量一致
        if len(concept_vector) != len(adjectives):
            raise RuntimeError(
                f"concept_vector 长度异常：期望 {len(adjectives)}，实际 {len(concept_vector)}"
            )

        results.append({"content": content, "toxic": sample["toxic"], "concept": concept_vector, "raw_probs": raw_probs})

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
    generate_adj_concept(data_path, output_path, config.adjective_path, args.template, tokenizer, model)


if __name__ == '__main__':
    main()
