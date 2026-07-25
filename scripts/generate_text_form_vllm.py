"""生成文本形式特征概念向量（Chat Template + vLLM）

与 generate_adjective_c_r_vllm.py 的核心区别：
1. 系统提示词：聚焦于"文本表达形式（句式、论证结构、修辞）"而非"隐含语义"
2. 形容词词典：text_form_adjectives.csv（10个文本形式特征形容词）
3. 输出命名：concept_{mode}_{model}_text_form.json

核心逻辑（与原脚本完全一致）：
- Verbalizer技术：max_tokens=1, temperature=0, logprobs=20
- Likert加权期望：score = Σ(weight_k × P(k)) / Σ(P(k))
- verbalizer词表：["1","2","3","4","5"]
- 截断阈值：1e-4

使用示例：
python scripts/generate_text_form_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
python scripts/generate_text_form_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json

# AutoDL环境中OMP_NUM_THREADS可能被设为无效值
if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


# =============================================================================
# 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="生成文本形式特征概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集的概念向量，test:生成测试集的概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表（与原脚本完全一致）
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen2.5-14B-Instruct": {
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
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载（与原脚本完全一致）
# =============================================================================
def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
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
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 提示词定义（核心差异：聚焦文本表达形式）
# =============================================================================
SYSTEM_INSTRUCTION = (
    "你是一位语言形式分析专家，擅长识别文本的表达方式和句式结构。\n"
    "在评估时，请关注文本的表达形式（如句式类型、论证结构、修辞手法、语气的直接或间接程度），\n"
    "而非文本的语义内容或情感倾向。\n"
    "判断文本是否采用了该形容词所描述的表达方式。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)


def build_chat_messages(content, adj, adj_definition=None):
    """构建Likert评分的Chat Template messages。"""
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的表达方式？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Verbalizer工具（与原脚本完全一致）
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")
    return list(dict.fromkeys(token_ids))


def extract_likert_score(first_token_logprobs, likert_ids):
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    level_probs = [probs_dict.get(tid, 0.0) for tid in likert_ids]

    weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    level_probs_t = torch.tensor(level_probs)
    total = level_probs_t.sum() + 1e-8
    score = (weights * level_probs_t / total).sum().item()

    return score, level_probs


# =============================================================================
# 核心流程：生成文本形式特征概念向量
# =============================================================================
def generate_text_form_concepts(data_path, output_path, csv_output_path, adjective_path,
                                 tokenizer, llm_model,
                                 is_qwen3=False, prompt_suffix="", threshold=1e-4):
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]

        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            messages = build_chat_messages(content, adj, adj_def)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        raw_probs = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_likert_score(first_token_logprobs, likert_ids)
            concept_vector.append(score)
            raw_probs.append(level_probs)

        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector长度异常：期望{num_adjs}，实际{len(concept_vector)}")

        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "likert_probs": raw_probs,
        }
        results.append(result_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"文本形式特征概念向量(JSON)保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"文本形式特征概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {num_adjs}] (文本数, 形容词数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"

    # 固定使用text_form_adjectives.csv
    adjective_path = config.raw_data_path / "adjective" / "text_form_adjectives.csv"
    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件命名：concept_{mode}_{model}_text_form.json
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_text_form.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_text_form.csv"

    print("\n" + "=" * 60)
    print("文本形式特征概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"形容词词典: text_form_adjectives.csv ({adjective_path})")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")

    generate_text_form_concepts(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
