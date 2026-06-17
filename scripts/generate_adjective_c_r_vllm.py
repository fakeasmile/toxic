"""生成形容词概念向量（Chat Template + vLLM）

核心思路：利用LLM的verbalizer token技术，提取首token概率分布，
量化文本与各形容词之间的相关程度，构建可解释的概念向量。

【两种推理模式】
1. 单阶段模式（默认）：直接让LLM评估文本与形容词的相关程度
2. 两阶段模式（--use_two_stage）：
   - Stage 1：让LLM分析文本的隐含语义（谐音暗语、文化隐喻、反讽等），结果缓存
   - Stage 2：将隐含语义分析注入prompt，再进行Likert评分

【Verbalizer提取流程】
1. 构建Chat Template prompt → LLM推理（max_tokens=1, temperature=0）
2. 提取首token的Top-20 logprobs → exp()转为概率
3. 从概率分布中提取verbalizer token（"1"~"5"）的概率
4. 加权期望：score = Σ(weight_k × P(k)) / Σ(P(k))，weights=[0, 0.25, 0.5, 0.75, 1.0]

【配套调试工具】
- inspect_prompt_template_vllm.py：单样本切片（1文本+1形容词），调试提示词和verbalizer
- inspect_verbalizer_coverage_vllm.py：全景扫描（1文本+全部形容词），验证verbalizer覆盖率

使用示例：
# 单阶段模式
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct

# 两阶段模式
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct --use_two_stage
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
        description="生成形容词概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集的概念向量，test:生成测试集的概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='采样温度（默认2.0），用于控制概率分布的分散程度')
    parser.add_argument('--use_two_stage', action='store_true',
                        help='启用两阶段推理：Stage1生成隐含语义分析，Stage2基于分析进行Likert评分')
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
# 单阶段模式的system instruction：提示LLM考虑隐含语义
SINGLE_STAGE_INSTRUCTION = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "在评估时，请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
    "判断文本是否体现了该形容词所描述的特征。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)

# 两阶段模式Stage 2的system instruction：提示LLM综合考虑隐含语义分析
STAGE2_INSTRUCTION = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "在评估时，请综合考虑文本的字面意思和隐含语义分析，\n"
    "判断文本是否体现了该形容词所描述的特征。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)

# 两阶段模式Stage 1的提示词：让LLM分析文本的隐含语义
STAGE1_SYSTEM = "你是一位语言分析专家，擅长识别中文文本中的隐含语义。"
STAGE1_USER_TEMPLATE = (
    "分析以下文本是否包含隐含语义，从以下角度简要分析：\n"
    "1. 谐音暗语（如\"鲨bee\"=\"傻逼\"，\"默\"=\"黑\"）\n"
    "2. 文化隐喻或间接攻击\n"
    "3. 反讽或阴阳怪气\n"
    "如果包含隐含语义，简要说明；如果不包含，回答\"无\"。\n"
    "文本内容：{content}\n"
    "分析："
)


# =============================================================================
# Prompt构建
# =============================================================================
def build_chat_messages(content, adj, adj_definition=None, implicit_analysis=None, use_two_stage=False):
    """构建Likert评分的Chat Template messages。

    根据use_two_stage自动选择instruction：
    - False（单阶段）：instruction提示考虑隐含意思
    - True（两阶段）：instruction提示综合考虑隐含语义分析

    user_content结构：
        文本内容：{content}
        隐含语义：{analysis}    ← 仅两阶段模式且有分析结果时插入
        形容词：{adj}
        定义：{adj_definition}  ← 仅当定义存在时插入
        该文本在多大程度上体现了"{adj}"所描述的特征？回答：
    """
    instruction = STAGE2_INSTRUCTION if use_two_stage else SINGLE_STAGE_INSTRUCTION

    user_lines = [f"文本内容：{content}"]
    if implicit_analysis:
        user_lines.append(f"隐含语义：{implicit_analysis}")
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": instruction},
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


def extract_likert_score(first_token_logprobs, likert_ids):
    """从首token的logprobs中提取Likert加权期望分数。

    Args:
        first_token_logprobs: vLLM返回的首token logprobs字典 {token_id: Logprob对象}
        likert_ids: verbalizer token id列表（对应"1"~"5"）

    Returns:
        (score, level_probs): 加权期望分数(0~1), 5级概率列表
    """
    # logprobs → 概率
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    # 提取1~5等级的概率
    level_probs = [probs_dict.get(tid, 0.0) for tid in likert_ids]

    # 加权期望：score = Σ(w_k × p_k) / Σ(p_k + ε)
    weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    level_probs_t = torch.tensor(level_probs)
    total = level_probs_t.sum() + 1e-8  # 防零除
    score = (weights * level_probs_t / total).sum().item()

    return score, level_probs.tolist()


# =============================================================================
# Stage 1：隐含语义分析
# =============================================================================
def generate_stage1_analysis(data_set, tokenizer, llm_model, is_qwen3=False, prompt_suffix="", cache_path=None):
    """Stage 1：为每条文本生成隐含语义分析，结果缓存到文件。

    每条文本只运行1次Stage 1，其分析结果被该文本的所有177个形容词共享。
    缓存机制保证可复现性：同一文本的分析内容在多次运行间完全一致。

    Args:
        data_set: 数据集列表，每条含"content"字段
        tokenizer: tokenizer
        llm_model: vLLM模型
        is_qwen3: 是否为Qwen3+模型（需禁用thinking）
        prompt_suffix: 模型特定的prompt后缀
        cache_path: 缓存文件路径，已存在则直接读取

    Returns:
        list[str]: 每条文本的隐含语义分析文本
    """
    # 缓存已存在则直接读取
    if cache_path and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"Stage 1缓存已存在，直接读取: {cache_path} ({len(cached)}条)")
        return cached

    # 构建所有文本的Stage 1 prompt
    stage1_prompts = []
    for sample in data_set:
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": STAGE1_USER_TEMPLATE.format(content=sample["content"])},
        ]
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        stage1_prompts.append(prompt_text)

    # 批量推理：max_tokens=80控制分析长度，temperature=0保证确定性
    stage1_params = SamplingParams(max_tokens=80, temperature=0)
    print(f"Stage 1：为 {len(data_set)} 条文本生成隐含语义分析...")
    outputs = llm_model.generate(stage1_prompts, stage1_params, use_tqdm=True)

    analyses = [output.outputs[0].text.strip() for output in outputs]

    # 缓存到文件
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)
        print(f"Stage 1分析已缓存: {cache_path}")

    return analyses


# =============================================================================
# 核心流程：生成形容词概念向量
# =============================================================================
def generate_adj_concept(data_path, output_path, csv_output_path, adjective_path,
                         temperature, tokenizer, llm_model,
                         is_qwen3=False, prompt_suffix="", threshold=1e-4,
                         use_two_stage=False, stage1_cache_path=None):
    """生成形容词概念向量。

    对数据集中每条文本，遍历所有形容词，通过verbalizer技术提取Likert评分，
    构建概念向量（每条文本一个V维向量，V=形容词数量）。

    流程：
    1. 加载形容词词典和数据集
    2. [两阶段] Stage 1：为每条文本生成隐含语义分析
    3. 逐文本处理：构建prompt → vLLM推理 → 提取首token概率 → Likert加权期望
    4. 保存结果（JSON含完整信息，CSV为纯矩阵）
    """
    # --- 准备工作 ---
    # Likert verbalizer token id
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # --- Stage 1：隐含语义分析（仅两阶段模式） ---
    implicit_analyses = None
    if use_two_stage:
        implicit_analyses = generate_stage1_analysis(
            data_set, tokenizer, llm_model,
            is_qwen3=is_qwen3, prompt_suffix=prompt_suffix,
            cache_path=stage1_cache_path
        )

    # --- Stage 2：Likert评分 ---
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []  # [N, V] 矩阵，用于CSV输出

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]
        analysis = implicit_analyses[sample_idx] if implicit_analyses else None

        # 为当前文本构建所有形容词的prompt
        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            messages = build_chat_messages(
                content, adj, adj_def,
                implicit_analysis=analysis, use_two_stage=use_two_stage
            )
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        # 批量推理：一次性送入当前文本的所有prompt（vLLM自动调度）
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        # 从每条推理结果中提取Likert分数
        concept_vector = []
        raw_probs = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_likert_score(first_token_logprobs, likert_ids)
            concept_vector.append(score)
            raw_probs.append(level_probs)

        # 防御性校验
        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector长度异常：期望{num_adjs}，实际{len(concept_vector)}")

        # 截断极小值（低于阈值的分数设为0，避免浮点噪声）
        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        # 组装结果
        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "likert_probs": raw_probs,
        }
        if use_two_stage and analysis:
            result_item["implicit_analysis"] = analysis
        results.append(result_item)

    # --- 保存结果 ---
    # JSON：含完整信息（content, toxic, concept, likert_probs, implicit_analysis）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"形容词概念向量(JSON)保存到: {output_path}")

    # CSV：纯矩阵 [N, V]，横轴为文本，纵轴为形容词
    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"形容词概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(adjectives)}] (文本数, 形容词数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    # 构建路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    # 两阶段模式下输出文件名加_two_stage后缀
    suffix = "_two_stage" if args.use_two_stage else ""
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}{suffix}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}{suffix}.csv"

    # Stage 1缓存路径
    stage1_cache_path = None
    if args.use_two_stage:
        stage1_cache_path = concept_dir / f"implicit_analysis_{args.mode}_{args.model_name}.json"

    # 打印配置
    print("\n" + "=" * 60)
    print("形容词概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"采样温度: {args.temperature}")
    print(f"两阶段模式: {'是' if args.use_two_stage else '否'}")
    if args.use_two_stage:
        print(f"Stage 1缓存路径: {stage1_cache_path}")
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
    generate_adj_concept(
        data_path, output_path, csv_output_path, config.adjective_path,
        args.temperature, tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4,
        use_two_stage=args.use_two_stage, stage1_cache_path=stage1_cache_path,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
