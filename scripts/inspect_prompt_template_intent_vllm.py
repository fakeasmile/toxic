"""意图概念向量提示词模板调试工具（单样本切片分析，vLLM版本）

【定位】
本脚本是 generate_adjective_intent_vllm.py 的"单样本切片"调试工具。
generate_adjective_intent_vllm.py 负责为数据集中所有文本、所有形容词批量生成意图概念向量；
而本脚本只抽取"一个文本 + 一个形容词"进行单步推理，用于在批量生成前快速验证
提示词模板和 Verbalizer 词表的设计是否合理。

【核心功能】
1. 首 token 概率分布 Top-10
   观察模型在第一个输出位置的概率分布。如果 Top-10 中大部分是 verbalizer 词表中的词，
   说明提示词模板成功将模型输出约束到预期方向。
2. 模型实际生成序列（贪心解码，10个token）
   观察模型实际输出的文本是否通顺、是否符合模板要求（如是否直接回答1-3数字）。
3. Verbalizer 概率分析
   统计预定义 verbalizer 词表中所有词的概率总和，评估约束强度。
   - 理想情况下，该总和应占模型首 token 概率质量的 70%~90% 以上。
   - 若过低（如 < 0.5），说明模型大量概率分散到非预期词，提示词模板或 verbalizer 词表需改进。

【与 generate_adjective_intent_vllm.py 的关系】
- 本脚本的提示词构建逻辑、verbalizer 词表、分数计算逻辑与 generate_adjective_intent_vllm.py 完全一致。

【使用方法】
直接修改下方 CONFIG 区域的变量（模型名、文本内容、形容词等），然后运行：
python scripts/inspect_prompt_template_intent_vllm.py
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
TEXT_CONTENT = "什么被害妄想猎巫man"
ADJECTIVE = "嘲讽的"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
# ===================================================================


# 模型加载配置表（与 generate_adjective_intent_vllm.py 保持一致）
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
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置。未知模型将直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer（复用generate_adjective_intent_vllm逻辑）"""
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
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt + language_model_only")

    print(f"Loading vLLM model from {llm_path}")
    print(f" 量化方式: {quantization if quantization else '无量化'}")
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
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")

    return list(dict.fromkeys(token_ids))


# system instruction（与generate_adjective_intent_vllm.py一致）
SYSTEM_INSTRUCTION = (
    "你是一位语用分析专家，擅长识别文本中作者的真实意图。\n"
    "请综合考虑文本的字面意思和隐含语义（比喻、谐音、反讽、文化隐喻等），"
    "判断作者在文本中是否以该形容词所描述的方式表达其态度。\n"
    "用1到3的数字回答，1表示作者仅在讨论、引用、反对或客观陈述相关话题"
    "而未以该方式表达态度，2表示难以明确判断，"
    "3表示作者以该形容词所描述的方式表达其态度，包括以隐含、暗示或反讽等方式。只回答一个数字。"
)


def build_chat_messages(content, adj, adj_definition=None):
    """构建意图判断的Chat Template messages。
    逻辑与 generate_adjective_intent_vllm.py 中的模板构建保持一致。
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"作者是否以\"{adj}\"所描述的方式表达其态度？回答： ")
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

    # verbalizer词表（与generate_adjective_intent_vllm.py一致，3级）
    verbalizer_words = ["1", "2", "3"]

    # 生成完整prompt文本
    chat_template_kwargs = {"enable_thinking": False} if qwen3_flag else {}
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )
    # 追加模型特定的后缀
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")
    prompt += prompt_suffix

    print("\n" + "=" * 60)
    print("意图概念向量 - 模型推理调试（vLLM版本）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词: {ADJECTIVE}")
    print(f"形容词定义: {adj_definition}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"提示词: {prompt}")

    print(f"\n提示词token数: {len(tokenizer.encode(prompt))}")

    # vLLM采样配置：获取logprobs用于分析首token分布
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
    first_token_logprobs = logprobs[0]

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

        # 分数计算（与generate_adjective_intent_vllm.py保持一致，3级）
        print(f"\n{'=' * 60}")
        print("意图概念向量分数计算（3级）")
        print(f"{'=' * 60}")

        intent_ids = get_first_token_ids(verbalizer_words, tokenizer)
        # 3级权重：1=讨论(0.0), 2=模糊(0.5), 3=表达(1.0)
        weights = torch.tensor([0.0, 0.5, 1.0])

        level_probs_list = [probs_dict.get(tid, 0.0) for tid in intent_ids]
        level_probs = torch.tensor(level_probs_list)
        total_level_prob = level_probs.sum() + 1e-8
        score = (weights * level_probs / total_level_prob).sum().item()

        print(f"Level 1 (讨论/引用/反对, 权重0.00): {level_probs_list[0]:.6f}")
        print(f"Level 2 (模糊,           权重0.50): {level_probs_list[1]:.6f}")
        print(f"Level 3 (表达,           权重1.00): {level_probs_list[2]:.6f}")
        print(f"意图概率总和: {total_level_prob.item():.6f}")
        print(f"加权期望意图分数: {score:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
