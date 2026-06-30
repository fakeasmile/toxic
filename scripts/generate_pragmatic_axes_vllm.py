"""生成语用轴概念向量（Chat Template + vLLM）

核心思路：利用LLM的verbalizer token技术，提取首token概率分布，
从8个设计正交的语用维度量化文本的表达方式，构建可解释的概念向量。

与形容词相关度概念向量的本质区别：
- 形容词评分问"文本与[形容词]多相关？"→ 测量WHAT（是什么毒性），维度间高度相关r=0.77
- 语用轴评分问"文本在[维度]上处于什么水平？"→ 测量HOW（如何表达），维度间设计正交
- 8个语用轴：明确性、立场认同、反讽性、行动煽动、目标针对性、身份指向、情绪强度、语境依赖

【Verbalizer提取流程】
1. 构建Chat Template prompt → LLM推理（max_tokens=1, temperature=0）
2. 提取首token的Top-20 logprobs → exp()转为概率
3. 从概率分布中提取verbalizer token（"1"~"3"）的概率
4. 加权期望：score = Σ(weight_k × P(k)) / Σ(P(k))，weights=[0.0, 0.5, 1.0]

【3级verbalizer + 锚点对比设计】
- 3级verbalizer（1/2/3）替代5级（1~5），减少中间分hedging倾向
- 锚点对比（相比普通对话文本）提供明确参照基准，强制LLM给出明确判断
- 解决5级verbalizer下6/8轴argmax=3比例超过67%的hedging问题

使用示例：
python scripts/generate_pragmatic_axes_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
python scripts/generate_pragmatic_axes_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
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
        description="生成语用轴概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集的概念向量，test:生成测试集的概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--axes_name', type=str, default=None,
                        help='语用轴定义文件名（如pragmatic_axes.csv），默认使用data/raw/adjective/pragmatic_axes.csv')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
# 所有模型相关的加载参数均集中在此配置表中，保证LLM切换对后续流程透明。
# 新增模型只需在本字典中增加条目，通常无需修改核心推理逻辑。
# prompt_suffix：部分模型在chat template后需要追加后缀才能让首token为verbalizer token
#   - GLM-4：首token为\n，追加\n使其直接输出数字
#   - Qwen：首token带空格，已在提示词末尾加空格处理，suffix为空
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",       # FP8在线量化，加速推理
        "is_qwen3": True,            # 需禁用thinking模式
        "is_multimodal": True,       # 需跳过视觉编码器
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

    模型差异（量化、多模态、Qwen3+）均从MODEL_LOADING_CONFIG读取。
    Returns: (tokenizer, llm, is_qwen3)
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    # 加载tokenizer
    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建vLLM加载参数
    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,    # 启用前缀缓存，加速system指令复用
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    # 多模态模型：纯文本推理时跳过视觉编码器，释放显存
    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt + language_model_only")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 提示词定义
# =============================================================================
# system instruction：语用分析，3级verbalizer + 锚点对比
# 3级verbalizer减少中间分hedging倾向；锚点对比（普通对话文本）提供明确参照基准
SYSTEM_INSTRUCTION = (
    "你是一位语用分析专家，擅长从多个正交维度分析文本的表达方式。\n"
    "在评估时，请综合考虑文本的字面意思和可能的隐含语义（如比喻、谐音、反讽、文化隐喻等）。\n"
    "请以普通对话文本作为参照基准，用1到3的数字评估：\n"
    "1=低于普通文本（基本不涉及此维度）\n"
    "2=与普通文本相当（有一定涉及）\n"
    "3=高于普通文本（强烈体现此维度）\n"
    "只回答一个数字。"
)


# =============================================================================
# Prompt构建
# =============================================================================
def build_chat_messages(content, axis_chinese, definition=None, scale_description=None):
    """构建语用轴评分的Chat Template messages。

    采用3级verbalizer + 锚点对比设计：
    - 3级verbalizer（1/2/3）减少中间分hedging倾向
    - 锚点对比（相比普通对话文本）提供明确参照基准，强制LLM给出明确判断

    user_content结构：
        文本内容：{content}
        分析维度：{axis_chinese}
        定义：{definition}              ← 仅当定义存在时插入
        评分标准：{scale_description}    ← 仅当评分标准存在时插入
        相比普通对话文本，该文本在"{axis_chinese}"维度上的程度是？回答：
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"分析维度：{axis_chinese}")
    if definition:
        user_lines.append(f"定义：{definition}")
    if scale_description:
        user_lines.append(f"评分标准：{scale_description}")
    user_lines.append(f"相比普通对话文本，该文本在\"{axis_chinese}\"维度上的程度是？回答： ")
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

    verbalizer技术要求LLM的首token为预定义的token（如"1"~"5"），
    此函数将verbalizer词表映射为token id列表，用于从logprobs中提取概率。
    """
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")
    return list(dict.fromkeys(token_ids))  # 去重保序


def extract_pragmatic_score(first_token_logprobs, pragmatic_ids):
    """从首token的logprobs中提取语用轴加权期望分数。

    3级verbalizer设计（1/2/3），等距权重[0.0, 0.5, 1.0]。
    相比5级verbalizer，3级减少中间分hedging倾向，强制LLM给出更明确的判断。

    Args:
        first_token_logprobs: vLLM返回的首token logprobs字典 {token_id: Logprob对象}
        pragmatic_ids: verbalizer token id列表（对应"1"~"3"）

    Returns:
        (score, level_probs): 加权期望分数(0~1), 3级概率列表
    """
    # logprobs → 概率
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    # 提取1~3等级的概率
    level_probs = [probs_dict.get(tid, 0.0) for tid in pragmatic_ids]

    # 加权期望：score = Σ(w_k × p_k) / Σ(p_k + ε)
    weights = torch.tensor([0.0, 0.5, 1.0])
    level_probs_t = torch.tensor(level_probs)
    total = level_probs_t.sum() + 1e-8  # 防零除
    score = (weights * level_probs_t / total).sum().item()

    return score, level_probs


# =============================================================================
# 核心流程：生成语用轴概念向量
# =============================================================================
def generate_pragmatic_concept(data_path, output_path, csv_output_path, axes_path,
                               tokenizer, llm_model,
                               is_qwen3=False, prompt_suffix="", threshold=1e-4):
    """生成语用轴概念向量。

    对数据集中每条文本，遍历所有语用轴，通过verbalizer技术提取Likert评分，
    构建概念向量（每条文本一个V维向量，V=语用轴数量，默认8维）。

    流程：
    1. 加载语用轴定义和数据集
    2. 逐文本处理：构建prompt → vLLM推理 → 提取首token概率 → Likert加权期望
    3. 保存结果（JSON含完整信息，CSV为纯矩阵）
    """
    # --- 准备工作 ---
    # 语用轴 verbalizer token id（3级：1/2/3，减少中间分hedging）
    pragmatic_tokens = ["1", "2", "3"]
    pragmatic_ids = get_first_token_ids(pragmatic_tokens, tokenizer)

    # 加载语用轴定义
    axes_df = pd.read_csv(axes_path)
    axes_chinese = axes_df["axis_chinese"].tolist()
    axes_definitions = axes_df["definition"].tolist() if "definition" in axes_df.columns else [None] * len(axes_chinese)
    axes_scales = axes_df["scale_description"].tolist() if "scale_description" in axes_df.columns else [None] * len(axes_chinese)
    num_axes = len(axes_chinese)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # --- 语用轴评分 ---
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []  # [N, V] 矩阵，用于CSV输出

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]

        # 为当前文本构建所有语用轴的prompt
        prompts = []
        for axis, axis_def, axis_scale in zip(axes_chinese, axes_definitions, axes_scales):
            messages = build_chat_messages(content, axis, axis_def, axis_scale)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        # 批量推理：一次性送入当前文本的所有prompt（vLLM自动调度）
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        # 从每条推理结果中提取语用轴分数
        concept_vector = []
        raw_probs = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_pragmatic_score(first_token_logprobs, pragmatic_ids)
            concept_vector.append(score)
            raw_probs.append(level_probs)

        # 防御性校验
        if len(concept_vector) != num_axes:
            raise RuntimeError(f"concept_vector长度异常：期望{num_axes}，实际{len(concept_vector)}")

        # 截断极小值（低于阈值的分数设为0，避免浮点噪声）
        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        # 组装结果
        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "pragmatic_probs": raw_probs,
        }
        results.append(result_item)

    # --- 保存结果 ---
    # JSON：含完整信息（content, toxic, concept, pragmatic_probs）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"语用轴概念向量(JSON)保存到: {output_path}")

    # CSV：纯矩阵 [N, V]，横轴为文本，纵轴为语用轴
    df = pd.DataFrame(concept_matrix, columns=axes_chinese)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"语用轴概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(axes_chinese)}] (文本数, 语用轴数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    # 构建路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"

    # 语用轴定义路径：命令行指定 > 默认pragmatic_axes.csv
    if args.axes_name is not None:
        axes_path = config.raw_data_path / "adjective" / args.axes_name
    else:
        axes_path = config.raw_data_path / "adjective" / "pragmatic_axes.csv"
    if not axes_path.exists():
        raise FileNotFoundError(f"语用轴定义文件不存在: {axes_path}")

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_pragmatic.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_pragmatic.csv"

    # 打印配置
    print("\n" + "=" * 60)
    print("语用轴概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"语用轴定义: {axes_path.name} ({axes_path})")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    # 加载模型
    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")

    # 获取模型特定配置
    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")

    # 执行概念向量生成
    generate_pragmatic_concept(
        data_path, output_path, csv_output_path, axes_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
