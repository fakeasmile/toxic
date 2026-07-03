"""生成形容词概念向量——说话者行为视角（Chat Template + vLLM）

与 generate_adjective_c_r_vllm.py 的区别：
  - 原版（Likert视角）：评估"文本内容是否体现了该形容词的特征"（语义相关度）
  - 本版（Speaker视角）：评估"说话者是否在主动表现出该形容词的行为特征"（行为表现度）

核心区别在于prompt措辞：
  原版 System: "判断文本是否体现了该形容词所描述的特征"
  本版 System: "判断说话者是否在主动表现出该形容词所描述的行为特征"

  原版 User: "该文本在多大程度上体现了'{adj}'所描述的特征？"
  本版 User: "该文本的说话者在多大程度上表现出'{adj}'的行为特征？"

其他完全一致：Likert 1-5 verbalizer、权重、vLLM参数、输出格式。

理论依据 (BPDMoE-Hate, ACL 2026)：
  对同一输入从对抗性双视角分别评估，下游分类器学习哪个视角更有判别力。
  说话者视角的直觉：讨论有毒内容的说话者并未"表现出"有毒行为特征，
  而实施有毒行为的说话者确实"表现出"了该特征。

使用示例：
python scripts/generate_adjective_c_r_vllm_speaker.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
python scripts/generate_adjective_c_r_vllm_speaker.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json

# AutoDL环境中OMP_NUM_THREADS可能被设为无效值，导致vLLM报错，需清理
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
        description="生成形容词概念向量——说话者行为视角（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集的概念向量，test:生成测试集的概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--adjective_name', type=str, default=None,
                        help='形容词词典文件名（如toxic_adjectives_v1.csv），默认使用MLP_config.py中的adjective_path')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表（与Likert版完全一致）
# =============================================================================
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
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载（与Likert版完全一致）
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
# 提示词定义（★与Likert版不同★）
# =============================================================================
SYSTEM_INSTRUCTION = (
    "你是一位语言行为分析专家，擅长识别说话者的表达行为方式。\n"
    "在评估时，请重点关注说话者（即文本的发出者）是否在主动表现出该形容词所描述的行为特征，"
    "而非仅仅描述或讨论该特征。\n"
    "例如：\"他只会辱骂诬陷\"中说话者是在描述他人的侮辱行为，而非自己表现出侮辱性；"
    "\"你个废物\"中说话者是在主动表现出侮辱性的行为。\n"
    "请用1到5的数字评估说话者的表现程度，1表示完全没有表现出，5表示强烈表现出。只回答一个数字。"
)


# =============================================================================
# Prompt构建（★与Likert版不同★）
# =============================================================================
def build_chat_messages(content, adj, adj_definition=None):
    """构建说话者行为视角的Chat Template messages。

    与Likert版的区别：
      Likert: "该文本在多大程度上体现了'{adj}'所描述的特征？"
      Speaker: "该文本的说话者在多大程度上表现出'{adj}'的行为特征？"
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本的说话者在多大程度上表现出\"{adj}\"的行为特征？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Verbalizer工具（与Likert版完全一致）
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
# 核心流程（与Likert版完全一致，仅输出文件名后缀不同）
# =============================================================================
def generate_adj_concept(data_path, output_path, csv_output_path, adjective_path,
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
    print(f"说话者视角概念向量(JSON)保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"说话者视角概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(adjectives)}] (文本数, 形容词数)")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"

    if args.adjective_name is not None:
        adjective_path = config.raw_data_path / "adjective" / args.adjective_name
    else:
        adjective_path = config.adjective_path
    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    adj_stem = adjective_path.stem
    adj_version = adj_stem.replace("toxic_adjectives_", "")

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    # ★输出文件名后缀为 _v1_speaker（区别于Likert版的 _v1）
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}_speaker.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}_speaker.csv"

    print("\n" + "=" * 60)
    print("形容词概念向量生成(vLLM) - 说话者行为视角")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"形容词词典: {adjective_path.name} ({adjective_path})")
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

    generate_adj_concept(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
