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
MODEL_NAME = "Qwen2.5-7B-Instruct-AWQ"  # models目录下的模型文件夹名

# 文本内容和形容词（直接修改即可）
TEXT_CONTENT = "什么被害妄想猎巫man"
ADJECTIVE = "包容的"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
TEMPERATURE = 2.0  # 采样温度（默认2.0），用于控制概率分布的分散程度
# ===================================================================


# 模型加载配置表（与 generate_adjective_c_r_vllm.py 保持一致）
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct-AWQ": {
        "quantization": "awq",
        "is_qwen3": False,
        "is_multimodal": False,
    },
    "Qwen3.5-9B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": True,
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
    },
}


def get_model_loading_config(model_name: str) -> dict:
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置。未知模型将直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer（复用generate_adjective_c_r_vllm逻辑）

    所有模型差异（量化方式、多模态处理、Qwen3+ 标志）均从
    MODEL_LOADING_CONFIG 中读取，保证新增模型时只需改配置表。
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["max_num_seqs"] = 64
        llm_kwargs["max_model_len"] = 1024
        print(f"检测到多模态模型({model_name})，已启用纯文本推理优化（跳过视觉编码器+CUDA Graph+降低显存占用，保留前缀缓存）")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


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


def build_chat_messages(text_content, adjective):
    """
    构建Likert Chat Template的messages列表。
    逻辑与 generate_adjective_c_r_vllm.py 中的模板构建保持一致。
    """
    instruction = ("你是一位语言分析专家。请评估以下文本与形容词的相关程度。\n"
                   "评估等级：\n"
                   "1 = 完全不相关\n"
                   "2 = 不太相关\n"
                   "3 = 有点相关\n"
                   "4 = 比较相关\n"
                   "5 = 非常相关\n"
                   "直接回答数字。")
    user_content = f"文本内容：{text_content}\n形容词：{adjective}\n回答： "
    verbalizer_words = ["1", "2", "3", "4", "5"]
    score_tokens = {
        "likert": ["1", "2", "3", "4", "5"],
    }

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]

    return messages, verbalizer_words, score_tokens


def main():
    config = MLPConfig()

    tokenizer, llm_model, qwen3_flag = load_vllm_model(config.models_path, MODEL_NAME, GPU_MEMORY_UTILIZATION)

    # 构建Chat Template messages
    messages, verbalizer_words, score_tokens = build_chat_messages(
        TEXT_CONTENT, ADJECTIVE
    )

    # 生成完整prompt文本
    chat_template_kwargs = {"enable_thinking": False} if qwen3_flag else {}
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )

    print("\n" + "=" * 60)
    print("模型推理调试（vLLM版本）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词: {ADJECTIVE}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"采样温度: {TEMPERATURE}")
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

    # 手动应用temperature（vLLM的logprobs返回原始概率，不受temperature影响）
    if TEMPERATURE > 0:
        logits = {tid: math.log(p + 1e-10) for tid, p in probs_dict.items()}
        adjusted_logits = {tid: l / TEMPERATURE for tid, l in logits.items()}
        max_logit = max(adjusted_logits.values())
        exp_sum = sum(math.exp(l - max_logit) for l in adjusted_logits.values())
        probs_dict = {tid: math.exp(l - max_logit) / exp_sum for tid, l in adjusted_logits.items()}

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
    print(f"\n模型生成序列（贪心解码，max_tokens=10）:")
    sampling_params_gen = SamplingParams(
        max_tokens=10,
        temperature=0,
        logprobs=None
    )
    outputs_gen = llm_model.generate([prompt], sampling_params_gen, use_tqdm=False)
    generated_text = outputs_gen[0].outputs[0].text
    generated_ids = outputs_gen[0].outputs[0].token_ids
    print(f"生成token序列: {generated_ids}")
    print(f"生成token数量: {len(generated_ids)}")
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
