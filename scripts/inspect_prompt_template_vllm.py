"""Qwen模型提示词模板调试工具（单样本切片分析，vLLM版本）

【定位】
本脚本是 generate_adjective_c_r_vllm.py 的"单样本切片"调试工具。
generate_adjective_c_r_vllm.py 负责为数据集中所有文本、所有形容词批量生成概念向量；
而本脚本只抽取"一个文本 + 一个形容词"进行单步推理，用于在批量生成前快速验证
提示词模板和 Verbalizer 词表的设计是否合理。

【核心功能】
1. 首 token 概率分布 Top-10
   观察模型在第一个输出位置的概率分布。如果 Top-10 中大部分是 verbalizer 词表中的词，
   说明提示词模板成功将模型输出约束到预期方向。
2. 模型实际生成序列（贪心解码，10个token）
   观察模型实际输出的文本是否通顺、是否符合模板要求（如是否直接回答"是/否"或数字）。
3. Verbalizer 概率分析
   统计预定义 verbalizer 词表中所有词的概率总和，评估约束强度。
   - 理想情况下，该总和应占模型首 token 概率质量的 70%~90% 以上。
   - 若过低（如 < 0.5），说明模型大量概率分散到非预期词，提示词模板或 verbalizer 词表需改进。

【与 generate_adjective_c_r_vllm.py 的关系】
- 本脚本的提示词构建逻辑、verbalizer 词表、分数计算逻辑与 generate_adjective_c_r_vllm.py 完全一致。
- 通过本脚本调试确认模板和 verbalizer 合理后，再运行 generate_adjective_c_r_vllm.py 进行批量生成，
  可确保生成的概念向量质量。

【支持的模板类型】
- binary: 二元判断（是/否），用于判断形容词是否准确描述文本；
- likert: 1-5 程度量化，用于评估文本具有形容词特征的程度；
- ICL: 基于形容词定义的判断（In-Context Learning），引入形容词定义进行推理。

【使用方法】
直接修改下方 CONFIG 区域的变量（模型名、模板类型、文本内容、形容词、形容词定义等），然后运行：
python scripts/inspect_prompt_template_vllm.py
"""
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "Qwen2.5-1.5B-Instruct"  # models目录下的模型文件夹名

# 提示词模板类型，可选: "binary", "likert", "ICL"
PROMPT_TEMPLATE = "binary"

# 文本内容和形容词（直接修改即可）
TEXT_CONTENT = "中国男性根本就配不上中国女性"
ADJECTIVE = "攻击性的"

# ICL模板专用：形容词定义（仅ICL模板使用，不需要时可留空）
ADJECTIVE_DEFINITION = "尊重和接纳不同群体或观点的态度，不对他人身份、地域或取向进行排斥或歧视的言论。"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
QUANTIZATION = None  # 量化方法：None/awq/fp8
# ===================================================================


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85, quantization: str = None):
    """加载vLLM模型和tokenizer（复用generate_adjective_c_r_vllm逻辑）"""
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
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )

    return tokenizer, llm


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return list(dict.fromkeys(token_ids))


def build_chat_messages(template, text_content, adjective, adj_definition=""):
    """
    根据模板类型构建Chat Template的messages列表。
    逻辑与 generate_adjective_c_r_vllm.py 中的模板构建保持一致。
    """
    if template in ["binary", "ICL"]:
        if template == "binary":
            instruction = "你是一位社会心理学专家。请判断给定形容词是否准确描述文本，只回答是或否。"
            user_content = f"文本内容：{text_content}\n形容词：「{adjective}」描述是否准确？回答： "
        elif template == "ICL":
            instruction = "请根据形容词的定义，判断该文本是否表现出该形容词所描述的特征，只回答是或否。"
            user_content = (
                f"文本内容：{text_content}\n"
                f"形容词「{adjective}」的定义：{adj_definition}\n"
                f"根据上述定义，该文本是否表现出该形容词「{adjective}」所描述的特征？回答： "
            )

        verbalizer_words = ["是", " 是", "Yes", " Yes", "yes", " yes",
                            "否", "不", " 不", "不能", "无", "No", " No", "no", " no"]
        score_tokens = {
            "affirmative": ["是", " 是", "Yes", " Yes", "yes", " yes"],
            "negative": ["否", "不", " 不", "不能", "无", "No", " No", "no", " no"],
        }

    elif template == "likert":
        instruction = ("你是一位社会心理学专家。请评估以下文本具有给定形容词所描述特征的程度。\n"
                       "请严格按照1到5的等级进行评估：\n"
                       "1 = 完全不具有该特征\n"
                       "2 = 略微具有该特征\n"
                       "3 = 中等程度具有该特征\n"
                       "4 = 较强程度具有该特征\n"
                       "5 = 非常强烈地具有该特征")
        user_content = f"文本内容：{text_content}\n形容词「{adjective}」的程度等级（直接回答数字）： "
        verbalizer_words = ["1", "2", "3", "4", "5"]
        score_tokens = {
            "likert": ["1", "2", "3", "4", "5"],
        }

    else:
        raise ValueError(f"不支持的模板类型: {template}，可选: binary, likert, ICL")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]

    return messages, verbalizer_words, score_tokens


def main():
    config = MLPConfig()

    tokenizer, llm_model = load_vllm_model(config.models_path, MODEL_NAME, GPU_MEMORY_UTILIZATION, QUANTIZATION)

    # 根据模板构建Chat Template messages
    messages, verbalizer_words, score_tokens = build_chat_messages(
        PROMPT_TEMPLATE, TEXT_CONTENT, ADJECTIVE, ADJECTIVE_DEFINITION
    )

    # 生成完整prompt文本
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n" + "=" * 60)
    print("模型推理调试（vLLM版本）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"模板类型: {PROMPT_TEMPLATE}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词: {ADJECTIVE}")
    print(f"量化方法: {QUANTIZATION if QUANTIZATION else '无量化'}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"提示词: {prompt}")

    print(f"\n提示词token数: {len(tokenizer.encode(prompt))}")

    # vLLM采样配置：获取logprobs用于分析首token分布，同时生成少量token观察输出
    sampling_params_analysis = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20
    )

    # 推理获取首token logprobs
    outputs = llm_model.generate([prompt], sampling_params_analysis, use_tqdm=False)
    output = outputs[0]

    # 提取首token的logprobs分布
    logprobs = output.outputs[0].logprobs
    first_token_logprobs = logprobs[0]  # {token_id: Logprob对象}

    # 转换为概率字典
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    # 输出概率最高的前10个token
    topk = 10
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
    print(f"\n首token概率分布 Top-{topk}:")
    print(f"{'排名':<4} {'Token ID':<10} {'Token文本':<12} {'概率':<12} {'累计概率':<10}")
    cumsum = 0.0
    for rank, (tid, prob) in enumerate(sorted_probs, 1):
        token_text = tokenizer.decode([tid])
        cumsum += prob
        print(f"{rank:<4} {tid:<10} {repr(token_text):<12} {prob:<12.6f} {cumsum:<10.6f}")

    # 模型生成的10个词（贪心解码）
    print(f"\n模型生成 Top-10（贪心解码，每次取概率最高的token）:")
    sampling_params_gen = SamplingParams(
        max_tokens=10,
        temperature=0,
        logprobs=None
    )
    outputs_gen = llm_model.generate([prompt], sampling_params_gen, use_tqdm=False)
    generated_text = outputs_gen[0].outputs[0].text
    generated_ids = outputs_gen[0].outputs[0].token_ids
    print(f"生成token序列: {generated_ids}")
    print(f"生成文本: {repr(generated_text)}")

    # Verbalizer分析
    if verbalizer_words:
        verbalizer_ids = get_first_token_ids(verbalizer_words, tokenizer)

        print(f"\nVerbalizer分析 ({len(verbalizer_words)}个词 -> {len(verbalizer_ids)}个唯一token):")
        print(f"{'词':<10} {'Token ID':<10} {'概率':<12}")
        verbalizer_prob_list = []
        for word in verbalizer_words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if encoded:
                tid = encoded[0]
                p = probs_dict.get(tid, 0.0)
                verbalizer_prob_list.append((word, tid, p))
                print(f"{word:<10} {tid:<10} {p:<12.6f}")

        # verbalizer概率统计
        total_vprob = sum(p for _, _, p in verbalizer_prob_list)
        print(f"\nVerbalizer概率总和: {total_vprob:.6f}")
        print(f"Verbalizer占总概率比例: {total_vprob:.2%}")

        # 分数计算（与generate_adjective_c_r_vllm.py保持一致）
        print(f"\n{'=' * 60}")
        print("概念向量分数计算")
        print(f"{'=' * 60}")

        if PROMPT_TEMPLATE == "binary" or PROMPT_TEMPLATE == "ICL":
            affirmative_ids = get_first_token_ids(score_tokens["affirmative"], tokenizer)
            negative_ids = get_first_token_ids(score_tokens["negative"], tokenizer)

            pos_prob = sum(probs_dict.get(tid, 0.0) for tid in affirmative_ids)
            neg_prob = sum(probs_dict.get(tid, 0.0) for tid in negative_ids)
            total = pos_prob + neg_prob + 1e-8
            score = pos_prob / total

            print(f"肯定词概率: {pos_prob:.6f}")
            print(f"否定词概率: {neg_prob:.6f}")
            print(f"归一化后概念分数: {score:.6f}")

        elif PROMPT_TEMPLATE == "likert":
            likert_ids = get_first_token_ids(score_tokens["likert"], tokenizer)
            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

            level_probs_list = [probs_dict.get(tid, 0.0) for tid in likert_ids]
            level_probs = torch.tensor(level_probs_list)
            total_level_prob = level_probs.sum() + 1e-8
            score = (weights * level_probs / total_level_prob).sum().item()

            print(f"Level 1 (权重0.00): {level_probs_list[0]:.6f}")
            print(f"Level 2 (权重0.25): {level_probs_list[1]:.6f}")
            print(f"Level 3 (权重0.50): {level_probs_list[2]:.6f}")
            print(f"Level 4 (权重0.75): {level_probs_list[3]:.6f}")
            print(f"Level 5 (权重1.00): {level_probs_list[4]:.6f}")
            print(f"Likert概率总和: {total_level_prob.item():.6f}")
            print(f"加权期望概念分数: {score:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
