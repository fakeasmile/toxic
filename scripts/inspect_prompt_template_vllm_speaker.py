"""说话者视角提示词模板调试工具（单样本切片分析，vLLM版本）

【定位】
本脚本是 generate_adjective_c_r_vllm_speaker.py 的"单样本切片"调试工具。
只抽取"一个文本 + 一个形容词"进行单步推理，用于在批量生成前快速验证
说话者视角的提示词模板和 Verbalizer 词表是否合理。

与 Likert 版（inspect_prompt_template_vllm.py）的区别：
  - System instruction: 语言行为分析专家 vs 语言分析专家
  - User question: "说话者在多大程度上表现出X的行为特征" vs "文本在多大程度上体现了X的特征"

使用方法：
修改下方 CONFIG 区域的变量，然后运行：
python scripts/inspect_prompt_template_vllm_speaker.py
"""
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "glm-4-9b-chat"  # models目录下的模型文件夹名

# 文本内容和形容词（直接修改即可）
# 测试用例：对比"讨论toxic"和"实施toxic"的评分差异
TEXT_CONTENT = "他只会辱骂诬陷"  # 非有毒：描述他人的侮辱行为
ADJECTIVE = "侮辱性的"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85
# ===================================================================


# 模型加载配置表（与 generate_adjective_c_r_vllm_speaker.py 保持一致）
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",
        "is_qwen3": True,
        "is_multimodal": True,
        "prompt_suffix": "",
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "\n",
    },
    "deepseek-llm-7b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Baichuan2-7B-Chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3-8B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
}


def get_model_loading_config(model_name: str) -> dict:
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")
    return list(dict.fromkeys(token_ids))


# ★说话者视角 System instruction（与 generate_adjective_c_r_vllm_speaker.py 一致）
SYSTEM_INSTRUCTION = (
    "你是一位语言行为分析专家，擅长识别说话者的表达行为方式。\n"
    "在评估时，请重点关注说话者（即文本的发出者）是否在主动表现出该形容词所描述的行为特征，"
    "而非仅仅描述或讨论该特征。\n"
    "例如：\"他只会辱骂诬陷\"中说话者是在描述他人的侮辱行为，而非自己表现出侮辱性；"
    "\"你个废物\"中说话者是在主动表现出侮辱性的行为。\n"
    "请用1到5的数字评估说话者的表现程度，1表示完全没有表现出，5表示强烈表现出。只回答一个数字。"
)


# ★说话者视角 Prompt构建（与 generate_adjective_c_r_vllm_speaker.py 一致）
def build_chat_messages(content, adj, adj_definition=None):
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本的说话者在多大程度上表现出\"{adj}\"的行为特征？回答： ")
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    return messages


def main():
    config = MLPConfig()

    tokenizer, llm_model, qwen3_flag = load_vllm_model(config.models_path, MODEL_NAME, GPU_MEMORY_UTILIZATION)

    # 从形容词词典CSV中查找定义
    adj_df = pd.read_csv(config.adjective_path)
    adj_definition = None
    if "chinese" in adj_df.columns:
        match = adj_df[adj_df["chinese"] == ADJECTIVE]
        if not match.empty and "definition" in adj_df.columns:
            adj_definition = match.iloc[0]["definition"]

    # 构建Chat Template messages
    messages = build_chat_messages(TEXT_CONTENT, ADJECTIVE, adj_definition)

    # verbalizer词表
    verbalizer_words = ["1", "2", "3", "4", "5"]

    # 生成完整prompt文本
    chat_template_kwargs = {"enable_thinking": False} if qwen3_flag else {}
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")
    prompt += prompt_suffix

    print("\n" + "=" * 60)
    print("说话者视角 - 模型推理调试（vLLM版本）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词: {ADJECTIVE}")
    print(f"形容词定义: {adj_definition}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"提示词: {prompt}")
    print(f"提示词token数: {len(tokenizer.encode(prompt))}")

    # vLLM采样配置
    sampling_params_analysis = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # 推理获取首token logprobs
    outputs = llm_model.generate([prompt], sampling_params_analysis, use_tqdm=False)
    output = outputs[0]

    # 提取首token的logprobs分布
    first_token_logprobs = output.outputs[0].logprobs[0]

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
    print(f"\n模型生成序列（贪心解码，max_tokens=10）:")
    sampling_params_gen = SamplingParams(max_tokens=10, temperature=0, logprobs=None)
    outputs_gen = llm_model.generate([prompt], sampling_params_gen, use_tqdm=False)
    generated_text = outputs_gen[0].outputs[0].text
    generated_ids = outputs_gen[0].outputs[0].token_ids
    print(f"生成token序列: {generated_ids}")
    print(f"生成token数量: {len(generated_ids)}")
    print(f"生成文本: {repr(generated_text)}")

    # Verbalizer分析
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

    total_vprob = sum(p for _, _, p in verbalizer_prob_list)
    print(f"\nVerbalizer概率总和: {total_vprob:.6f}")
    print(f"Verbalizer占总概率比例: {total_vprob:.2%}")

    # 分数计算
    print(f"\n{'=' * 60}")
    print("说话者视角概念向量分数计算")
    print(f"{'=' * 60}")

    likert_ids = get_first_token_ids(["1", "2", "3", "4", "5"], tokenizer)
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
