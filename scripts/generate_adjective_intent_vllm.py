"""生成形容词意图概念向量（Chat Template + vLLM）

核心思路：利用LLM的verbalizer token技术，提取首token概率分布，
量化作者以各形容词所描述方式"表达态度"的意图强度，构建可解释的意图概念向量。

【双信号方案】
本脚本生成"意图信号"，与 generate_adjective_c_r_vllm.py 生成的"相关度信号"拼接，
形成双信号概念向量（354维 = 177维相关度 + 177维意图），用于下游MLP训练。
- 相关度信号：文本与形容词的语义相关程度（1-5级Likert）
- 意图信号：作者是否以该形容词方式表达态度（1-3级：1=讨论/引用/反对，2=模糊，3=表达）

【Verbalizer提取流程】
1. 构建Chat Template prompt → LLM推理（max_tokens=1, temperature=0）
2. 提取首token的Top-20 logprobs → exp()转为概率
3. 从概率分布中提取verbalizer token（"1"~"3"）的概率
4. 加权期望：score = Σ(weight_k × P(k)) / Σ(P(k))，weights=[0.0, 0.5, 1.0]

【配套调试工具】
- inspect_prompt_template_intent_vllm.py：单样本切片（1文本+1形容词），调试提示词和verbalizer
- inspect_verbalizer_coverage_intent_vllm.py：全景扫描（1文本+全部形容词），验证verbalizer覆盖率

使用示例：
python scripts/generate_adjective_intent_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
python scripts/generate_adjective_intent_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
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
        description="生成形容词意图概念向量（vLLM版本，3级意图判断）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集的意图概念向量，test:生成测试集的意图概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--adjective_name', type=str, default=None,
                        help='形容词词典文件名（如toxic_adjectives_v1.csv），默认使用MLP_config.py中的adjective_path')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
# 与 generate_adjective_c_r_vllm.py 保持一致，保证LLM切换对后续流程透明。
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
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置，未知模型直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载
# =============================================================================
def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer。

    Returns: (tokenizer, llm, is_qwen3)
    """
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
    print(f" 量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 提示词定义
# =============================================================================
# system instruction：引导LLM进行语用推理，判断作者是否在"表达"该形容词所描述的态度
# 3级评分：1=讨论/引用/反对，2=模糊，3=表达（包括隐含方式）
SYSTEM_INSTRUCTION = (
    "你是一位语用分析专家，擅长识别文本中作者的真实意图。\n"
    "请综合考虑文本的字面意思和隐含语义（比喻、谐音、反讽、文化隐喻等），"
    "判断作者在文本中是否以该形容词所描述的方式表达其态度。\n"
    "用1到3的数字回答，1表示作者仅在讨论、引用、反对或客观陈述相关话题"
    "而未以该方式表达态度，2表示难以明确判断，"
    "3表示作者以该形容词所描述的方式表达其态度，包括以隐含、暗示或反讽等方式。只回答一个数字。"
)


# =============================================================================
# Prompt构建
# =============================================================================
def build_chat_messages(content, adj, adj_definition=None):
    """构建意图判断的Chat Template messages。

    user_content结构：
        文本内容：{content}
        形容词：{adj}
        定义：{adj_definition}  ← 仅当定义存在时插入
        作者是否以"{adj}"所描述的方式表达其态度？回答：
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"作者是否以\"{adj}\"所描述的方式表达其态度？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Verbalizer工具
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（去重）。

    verbalizer技术要求LLM的首token为预定义的token（如"1"~"3"），
    此函数将verbalizer词表映射为token id列表，用于从logprobs中提取概率。
    """
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")
    return list(dict.fromkeys(token_ids))


def extract_intent_score(first_token_logprobs, intent_ids):
    """从首token的logprobs中提取意图加权期望分数（3级）。

    Args:
        first_token_logprobs: vLLM返回的首token logprobs字典 {token_id: Logprob对象}
        intent_ids: verbalizer token id列表（对应"1"~"3"）

    Returns:
        (score, level_probs): 加权期望分数(0~1), 3级概率列表
    """
    # logprobs → 概率
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    # 提取1~3等级的概率
    level_probs = [probs_dict.get(tid, 0.0) for tid in intent_ids]

    # 加权期望：score = Σ(w_k × p_k) / Σ(p_k + ε)
    # 3级权重：1=讨论(0.0), 2=模糊(0.5), 3=表达(1.0)
    weights = torch.tensor([0.0, 0.5, 1.0])
    level_probs_t = torch.tensor(level_probs)
    total = level_probs_t.sum() + 1e-8
    score = (weights * level_probs_t / total).sum().item()

    return score, level_probs


# =============================================================================
# 核心流程：生成形容词意图概念向量
# =============================================================================
def generate_adj_intent(data_path, output_path, csv_output_path, adjective_path,
                        tokenizer, llm_model,
                        is_qwen3=False, prompt_suffix="", threshold=1e-4):
    """生成形容词意图概念向量。

    对数据集中每条文本，遍历所有形容词，通过verbalizer技术提取意图评分，
    构建意图概念向量（每条文本一个V维向量，V=形容词数量）。

    流程：
    1. 加载形容词词典和数据集
    2. 逐文本处理：构建prompt → vLLM推理 → 提取首token概率 → 意图加权期望
    3. 保存结果（JSON含完整信息，CSV为纯矩阵）
    """
    # --- 准备工作 ---
    # 意图verbalizer token id（3级：1=讨论, 2=模糊, 3=表达）
    intent_tokens = ["1", "2", "3"]
    intent_ids = get_first_token_ids(intent_tokens, tokenizer)

    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # --- 意图评分 ---
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]

        # 为当前文本构建所有形容词的prompt
        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            messages = build_chat_messages(content, adj, adj_def)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        # 批量推理
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        # 从每条推理结果中提取意图分数
        concept_vector = []
        raw_probs = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_intent_score(first_token_logprobs, intent_ids)
            concept_vector.append(score)
            raw_probs.append(level_probs)

        # 防御性校验
        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector长度异常：期望{num_adjs}，实际{len(concept_vector)}")

        # 截断极小值
        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        # 组装结果
        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "intent_probs": raw_probs,
        }
        results.append(result_item)

    # --- 保存结果 ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"意图概念向量(JSON)保存到: {output_path}")

    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"意图概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(adjectives)}] (文本数, 形容词数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")


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

    # 输出文件后缀加 _intent
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}_intent.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}_intent.csv"

    print("\n" + "=" * 60)
    print("形容词意图概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"形容词词典: {adjective_path.name} ({adjective_path})")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print(f"评分级别: 3级（1=讨论/引用/反对, 2=模糊, 3=表达）")
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

    generate_adj_intent(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
