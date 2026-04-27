"""生成形容词概念向量（vLLM版本）

使用vLLM加速批量推理，保持Chat Template格式不变。
利用vLLM的Prefix Caching自动复用同一样本不同形容词之间的共享前缀。

使用示例：
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template binary
"""
import argparse
import sys
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成形容词概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="使用示例:"
               "python scripts/generate_adjective_c_r_vllm.py --mode test --dataset_name TOXICN --model_name ... --template binary"
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


def load_vllm_model(model_path: Path, model_name: str):
    """加载vLLM模型和tokenizer"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM model from {llm_path}")
    # vLLM自动处理模型加载和量化
    # enable_prefix_caching=True是关键：自动复用共享前缀的KV Cache
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        max_model_len=4096,
    )

    return tokenizer, llm


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（目前已保证每个词只会对应一个token id）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return list(dict.fromkeys(token_ids))


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


def extract_verbalizer_probs(logprobs_dict, verbalizer_ids):
    """
    从vLLM的top-k logprobs中提取verbalizer概率。

    注意：vLLM返回的是top-k内的局部softmax概率，与全局softmax有轻微偏差。
    缺失的verbalizer token概率设为0。

    Args:
        logprobs_dict: vLLM返回的{token_id: logprob}字典
        verbalizer_ids: 需要提取的verbalizer token id列表

    Returns:
        list: 每个verbalizer token对应的概率
    """
    if not logprobs_dict:
        return [0.0] * len(verbalizer_ids)

    # 提取token ids和对应的logprobs
    token_ids = list(logprobs_dict.keys())
    logprobs = torch.tensor([logprobs_dict[tid] for tid in token_ids])

    # 在top-k内做softmax（局部概率）
    probs_in_topk = torch.softmax(logprobs, dim=-1)

    # 构建概率查找表
    prob_lookup = {}
    for tid, prob in zip(token_ids, probs_in_topk):
        tid_int = tid.item() if isinstance(tid, torch.Tensor) else tid
        prob_lookup[tid_int] = prob.item()

    # 提取verbalizer概率，缺失的设为0
    verbalizer_probs = []
    for tid in verbalizer_ids:
        tid_int = tid.item() if isinstance(tid, torch.Tensor) else tid
        verbalizer_probs.append(prob_lookup.get(tid_int, 0.0))

    return verbalizer_probs


def generate_adj_concept(data_path, output_path, adjective_path, template, tokenizer, llm):
    """生成形容词概念向量"""

    # 根据模板类型定义verbalizer token（首token id集合）和提示词指令
    if template in ["binary", "ICL"]:
        affirmative_tokens = ["是", " 是", "Yes", " Yes", "yes", " yes"]
        negative_tokens = ["否", "不", " 不", "不能", "无", "No", " No", "no", " no"]
        affirmative_ids = get_first_token_ids(affirmative_tokens, tokenizer)
        negative_ids = get_first_token_ids(negative_tokens, tokenizer)

        if template == "binary":
            instruction = "你是一位社会心理学专家。请判断给定形容词是否准确描述文本，只回答是或否。"
        elif template == "ICL":
            instruction = "请根据形容词的定义，判断该文本是否表现出该形容词所描述的特征，只回答是或否。"
            # 加载形容词解释
            definition = pd.read_csv(adjective_path)["definition"].tolist()

    elif template == "likert":
        likert_tokens = ["1", "2", "3", "4", "5"]
        likert_ids = get_first_token_ids(likert_tokens, tokenizer)

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

    # vLLM采样参数
    # max_tokens=1: 只生成1个token（我们只需要第一个token的概率分布）
    # logprobs=100: 返回top-100的logprobs，覆盖verbalizer token
    sampling_params = SamplingParams(
        max_tokens=1,
        logprobs=100,
    )

    results = []
    batch_size = 32  # vLLM内部会自动优化batch，可以设置较大

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples"), start=1):
        content = sample["content"]
        concept_vector = []  # 当前样本的形容词向量
        raw_probs = []  # 当前样本的原始概率

        # 按批次遍历形容词，每个形容词使用独立的Chat Template prompt
        for i in range(0, len(adjectives), batch_size):
            adj_batch = adjectives[i: i + batch_size]

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

            # vLLM批量推理
            # vLLM会自动处理batching和prefix caching
            outputs = llm.generate(prompt_texts, sampling_params)

            # 对每个形容词提取概率并打分
            for j, output in enumerate(outputs):
                # vLLM输出格式：output.outputs[0].logprobs是第一个生成位置的logprobs
                # 格式：{token_id: logprob, ...}
                if output.outputs and output.outputs[0].logprobs:
                    logprobs_dict = output.outputs[0].logprobs[0]
                else:
                    logprobs_dict = {}

                if template == "binary":
                    pos_probs = extract_verbalizer_probs(logprobs_dict, affirmative_ids)
                    neg_probs = extract_verbalizer_probs(logprobs_dict, negative_ids)

                    pos_prob = sum(pos_probs)
                    neg_prob = sum(neg_probs)
                    total = pos_prob + neg_prob + 1e-8
                    score = pos_prob / total
                    raw_probs.append([pos_prob, neg_prob])

                elif template == "likert":
                    level_probs = extract_verbalizer_probs(logprobs_dict, likert_ids)

                    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
                    total_level_prob = sum(level_probs) + 1e-8
                    score = sum(w * p for w, p in zip(weights, level_probs)) / total_level_prob
                    raw_probs.append(level_probs)

                elif template == "ICL":
                    pos_probs = extract_verbalizer_probs(logprobs_dict, affirmative_ids)
                    neg_probs = extract_verbalizer_probs(logprobs_dict, negative_ids)

                    pos_prob = sum(pos_probs)
                    neg_prob = sum(neg_probs)
                    total = pos_prob + neg_prob + 1e-8
                    score = pos_prob / total
                    raw_probs.append([pos_prob, neg_prob])

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
    print("形容词概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"提示词模板: {args.template}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm = load_vllm_model(config.models_path, args.model_name)
    generate_adj_concept(data_path, output_path, config.adjective_path, args.template, tokenizer, llm)


if __name__ == '__main__':
    main()
