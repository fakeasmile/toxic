"""诊断：Context-Aware Verbalizer 信息增量验证

核心问题：新提示词方案是否为概念向量带来**额外信息**？

验证策略（200样本，纯统计分析，无需训练分类器）：

实验1：新旧概念向量的信息增量
  - 逐维度计算新旧概念分数的Pearson r
  - r<0.8的维度比例：信息增量大的维度
  - r>0.95的维度比例：信息重叠的维度
  - 信息增量是否集中在FP关键概念上

实验2：FP区分力变化
  - 旧提示词的FP样本与Toxic样本在概念空间余弦相似度=0.8546
  - 新提示词是否降低这个相似度

实验3：Cohen's d变化
  - 新提示词是否提升概念的toxic/nontoxic区分力

实验4：概念交互验证
  - 新提示词下，同一文本不同概念间的相关性结构是否更强
  - 验证"概念交互"假说：新提示词让LLM在评分时考虑概念间关系

实验5：语用区分度
  - 新提示词下，FP样本(讨论toxic)与TP样本(实施toxic)的概念分数差异
  - 重点关注高区分力概念上的FP修正效果

三种条件：
1. Baseline：当前verbalizer（无Stage 1）
2. +Pragmatic Context：Stage 1语用分析 + Stage 2语境化评分
3. +Simple Instruction：原有prompt中加入"注意区分讨论和实施"（对照组）

输出保存到: experiments/context_aware_verbalizer_diagnosis.txt
"""

import json
import math
import os
import sys
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# ==== 路径 ====
PROCESSED_PATH = project_root / "data" / "processed" / "TOXICN"
GLM_PATH = PROCESSED_PATH / "glm-4-9b-chat"
ADJ_PATH = project_root / "data" / "raw" / "adjective" / "toxic_adjectives_v1.csv"
OUTPUT_PATH = project_root / "experiments" / "context_aware_verbalizer_diagnosis.txt"

# ==== 模型配置 ====
MODEL_NAME = "glm-4-9b-chat"
MODEL_PATH = project_root / "models"
N_SAMPLES = 200
SEED = 42


# =============================================================================
# 模型加载（与generate_adjective_c_r_vllm.py一致）
# =============================================================================
def load_model():
    llm_path = MODEL_PATH / MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(str(llm_path), trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    return tokenizer, llm


# =============================================================================
# Verbalizer工具（与generate_adjective_c_r_vllm.py一致）
# =============================================================================
LIKERT_TOKENS = ["1", "2", "3", "4", "5"]
PROMPT_SUFFIX = "\n"

SYSTEM_INSTRUCTION = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "在评估时，请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
    "判断文本是否体现了该形容词所描述的特征。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)

SYSTEM_INSTRUCTION_CONTEXT = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义和语用行为。\n"
    "在评估时，请综合考虑文本的字面意思、可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
    "以及文本的语用行为（说话者是在实施有害行为还是仅在讨论/引用/报道），\n"
    "判断文本是否体现了该形容词所描述的特征。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)

SYSTEM_INSTRUCTION_SIMPLE = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "在评估时，请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
    "判断文本是否体现了该形容词所描述的特征。\n"
    "注意：如果文本只是在讨论或引用有害内容，而非实施有害行为，应给出较低的评分。\n"
    "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。"
)


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
# Stage 1: 语用分析
# =============================================================================
PRAGMATIC_SYSTEM = "你是一位语用分析专家，请简要分析文本的语用特征。"

PRAGMATIC_USER_TEMPLATE = (
    "请分析以下文本的语用特征：\n\n"
    "文本：{content}\n\n"
    "从以下角度简要分析：\n"
    "1. 说话者是在实施有害行为（直接侮辱、威胁、歧视等），还是在讨论/引用/报道有害内容？\n"
    "2. 说话者的语用行为是什么（陈述事实、表达观点、发出指令、嘲讽等）？\n"
    "3. 是否有明确的被攻击对象？\n\n"
    "请简要回答，不超过80字。"
)


def run_stage1(tokenizer, llm, contents):
    sampling_params = SamplingParams(max_tokens=150, temperature=0)

    prompts = []
    for content in contents:
        messages = [
            {"role": "system", "content": PRAGMATIC_SYSTEM},
            {"role": "user", "content": PRAGMATIC_USER_TEMPLATE.format(content=content)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_text += PROMPT_SUFFIX
        prompts.append(prompt_text)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    analyses = [output.outputs[0].text.strip() for output in outputs]
    return analyses


# =============================================================================
# 三种条件的概念向量生成（仅200样本测试集）
# =============================================================================
def build_baseline_messages(content, adj, adj_definition):
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def build_context_messages(content, adj, adj_definition, pragmatic_analysis):
    user_lines = [f"语用分析：{pragmatic_analysis}"]
    user_lines.append(f"文本内容：{content}")
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION_CONTEXT},
        {"role": "user", "content": user_content},
    ]


def build_simple_messages(content, adj, adj_definition):
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： ")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION_SIMPLE},
        {"role": "user", "content": user_content},
    ]


def generate_concept_vectors(tokenizer, llm, contents, adjectives, adj_definitions,
                              likert_ids, mode="baseline", pragmatic_analyses=None):
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    all_concepts = []

    for idx, content in enumerate(tqdm(contents, desc=f"  Generating ({mode})")):
        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            if mode == "baseline":
                messages = build_baseline_messages(content, adj, adj_def)
            elif mode == "context":
                messages = build_context_messages(content, adj, adj_def, pragmatic_analyses[idx])
            elif mode == "simple":
                messages = build_simple_messages(content, adj, adj_def)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_text += PROMPT_SUFFIX
            prompts.append(prompt_text)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        for sample_info in outputs:
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, _ = extract_likert_score(first_token_logprobs, likert_ids)
            concept_vector.append(score if abs(score) >= 1e-4 else 0.0)

        all_concepts.append(concept_vector)

    return np.array(all_concepts)


# =============================================================================
# 统计分析
# =============================================================================
def compute_cohens_d(concepts, labels):
    """逐概念计算Cohen's d"""
    toxic_mask = labels == 1
    nontoxic_mask = labels == 0
    d_values = []
    for i in range(concepts.shape[1]):
        t = concepts[toxic_mask, i]
        nt = concepts[nontoxic_mask, i]
        d = (t.mean() - nt.mean()) / np.sqrt((t.var() + nt.var()) / 2 + 1e-8)
        d_values.append(d)
    return np.array(d_values)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def experiment1_information_increment(baseline, context, simple, labels, adjectives, lines):
    """实验1：新旧概念向量的信息增量"""
    log = lambda s="": (print(s), lines.append(s))

    log("\n" + "=" * 60)
    log("实验1：新旧概念向量的信息增量")
    log("=" * 60)

    for new_name, new_concepts in [("Context-Aware", context), ("Simple-Instruction", simple)]:
        log(f"\n  --- Baseline vs {new_name} ---")

        # 逐维度Pearson r
        per_dim_r = []
        for i in range(baseline.shape[1]):
            r, _ = pearsonr(baseline[:, i], new_concepts[:, i])
            per_dim_r.append(r)
        per_dim_r = np.array(per_dim_r)

        # 全局Pearson r（展平）
        r_global, _ = pearsonr(baseline.flatten(), new_concepts.flatten())

        log(f"    全局Pearson r: {r_global:.4f}")
        log(f"    逐维度r统计: 均值={per_dim_r.mean():.4f}, 中位数={np.median(per_dim_r):.4f}")
        log(f"    r>0.95维度数(高度重叠): {(per_dim_r > 0.95).sum()}/177")
        log(f"    r>0.90维度数: {(per_dim_r > 0.90).sum()}/177")
        log(f"    0.8<r<0.90维度数(部分增量): {((per_dim_r > 0.8) & (per_dim_r <= 0.9)).sum()}/177")
        log(f"    r<0.80维度数(显著增量): {(per_dim_r < 0.80).sum()}/177")
        log(f"    r<0.50维度数(重大增量): {(per_dim_r < 0.50).sum()}/177")

        # 信息增量集中在哪些概念上
        low_r_mask = per_dim_r < 0.80
        if low_r_mask.sum() > 0:
            log(f"\n    r<0.80的概念（信息增量最大的维度）:")
            low_r_idx = np.where(low_r_mask)[0]
            sorted_idx = low_r_idx[np.argsort(per_dim_r[low_r_idx])]
            for idx in sorted_idx[:20]:
                adj = adjectives[idx] if idx < len(adjectives) else f"概念{idx}"
                log(f"      {adj:15s} r={per_dim_r[idx]:.4f}")

        # 按标签分析：信息增量在Toxic/NonToxic上是否有差异
        toxic_mask = labels == 1
        nontoxic_mask = labels == 0

        r_toxic = []
        r_nontoxic = []
        for i in range(baseline.shape[1]):
            rt, _ = pearsonr(baseline[toxic_mask, i], new_concepts[toxic_mask, i])
            rn, _ = pearsonr(baseline[nontoxic_mask, i], new_concepts[nontoxic_mask, i])
            r_toxic.append(rt)
            r_nontoxic.append(rn)

        log(f"\n    按标签的信息增量:")
        log(f"      Toxic样本平均r: {np.mean(r_toxic):.4f}")
        log(f"      NonToxic样本平均r: {np.mean(r_nontoxic):.4f}")
        log(f"      增量差异(NonToxic-Toxic): {np.mean(r_nontoxic) - np.mean(r_toxic):.4f}")
        if np.mean(r_nontoxic) < np.mean(r_toxic):
            log(f"      → 新提示词对NonToxic样本产生更大信息增量（有利于FP区分）")
        else:
            log(f"      → 新提示词对Toxic样本产生更大信息增量")

    return per_dim_r


def experiment2_fp_discrimination(baseline, context, simple, labels, lines):
    """实验2：FP区分力变化"""
    log = lambda s="": (print(s), lines.append(s))

    log("\n" + "=" * 60)
    log("实验2：FP区分力变化")
    log("=" * 60)

    from sklearn.linear_model import LogisticRegression

    toxic_mask = labels == 1
    nontoxic_mask = labels == 0

    for name, concepts in [("Baseline", baseline), ("Context-Aware", context), ("Simple", simple)]:
        # 用5-fold CV方式，简单用LR预测来识别FP
        lr = LogisticRegression(C=10, max_iter=2000, random_state=42)
        lr.fit(concepts, labels)
        pred = lr.predict(concepts)

        fp_mask = (pred == 1) & (labels == 0)
        tp_mask = (pred == 1) & (labels == 1)
        tn_mask = (pred == 0) & (labels == 0)
        fn_mask = (pred == 0) & (labels == 1)

        # FP vs Toxic在概念空间的余弦相似度
        if fp_mask.sum() > 0 and tp_mask.sum() > 0:
            fp_mean = concepts[fp_mask].mean(axis=0)
            tp_mean = concepts[tp_mask].mean(axis=0)
            cos_fp_tp = cosine_similarity(fp_mean, tp_mean)

            # NonToxic正确 vs Toxic正确
            tn_mean = concepts[tn_mask].mean(axis=0)
            cos_tn_tp = cosine_similarity(tn_mean, tp_mean)
        else:
            cos_fp_tp = float('nan')
            cos_tn_tp = float('nan')

        # FP与NonToxic正确的余弦相似度
        if fp_mask.sum() > 0 and tn_mask.sum() > 0:
            fp_mean = concepts[fp_mask].mean(axis=0)
            tn_mean = concepts[tn_mask].mean(axis=0)
            cos_fp_tn = cosine_similarity(fp_mean, tn_mean)
        else:
            cos_fp_tn = float('nan')

        log(f"\n  {name}:")
        log(f"    FP样本数: {fp_mask.sum()}")
        log(f"    FP vs Toxic余弦相似度: {cos_fp_tp:.4f}")
        log(f"    正确NonToxic vs Toxic余弦相似度: {cos_tn_tp:.4f}")
        log(f"    FP vs 正确NonToxic余弦相似度: {cos_fp_tn:.4f}")

        # 关键指标：FP更接近哪一边
        if not np.isnan(cos_fp_tp) and not np.isnan(cos_fp_tn):
            if cos_fp_tn > cos_fp_tp:
                log(f"    → FP更接近正确NonToxic（新提示词有助于将FP拉回正确侧）")
            else:
                log(f"    → FP仍更接近Toxic（新提示词未改善FP区分）")


def experiment3_cohens_d_change(baseline, context, simple, labels, adjectives, lines):
    """实验3：Cohen's d变化"""
    log = lambda s="": (print(s), lines.append(s))

    log("\n" + "=" * 60)
    log("实验3：Cohen's d变化")
    log("=" * 60)

    d_base = compute_cohens_d(baseline, labels)
    d_ctx = compute_cohens_d(context, labels)
    d_simple = compute_cohens_d(simple, labels)

    log(f"\n  {'指标':<25s} {'Baseline':>10s} {'Context':>10s} {'Simple':>10s}")
    log(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    log(f"  {'平均|d|':<25s} {np.abs(d_base).mean():>10.4f} {np.abs(d_ctx).mean():>10.4f} {np.abs(d_simple).mean():>10.4f}")
    log(f"  {'|d|>1.0概念数':<25s} {(np.abs(d_base)>1.0).sum():>10d} {(np.abs(d_ctx)>1.0).sum():>10d} {(np.abs(d_simple)>1.0).sum():>10d}")
    log(f"  {'|d|>0.5概念数':<25s} {(np.abs(d_base)>0.5).sum():>10d} {(np.abs(d_ctx)>0.5).sum():>10d} {(np.abs(d_simple)>0.5).sum():>10d}")

    # 逐概念d变化
    d_change_ctx = np.abs(d_ctx) - np.abs(d_base)
    d_change_simple = np.abs(d_simple) - np.abs(d_base)

    log(f"\n  |d|变化统计:")
    log(f"    Context: 提升{(d_change_ctx > 0).sum()}/177, 下降{(d_change_ctx < 0).sum()}/177, 平均变化{d_change_ctx.mean():+.4f}")
    log(f"    Simple: 提升{(d_change_simple > 0).sum()}/177, 下降{(d_change_simple < 0).sum()}/177, 平均变化{d_change_simple.mean():+.4f}")

    # d提升最大的概念
    log(f"\n  Context |d|提升Top-10:")
    top10_idx = np.argsort(d_change_ctx)[-10:][::-1]
    for rank, idx in enumerate(top10_idx):
        adj = adjectives[idx] if idx < len(adjectives) else f"概念{idx}"
        log(f"    {rank+1}. {adj:15s} baseline_d={d_base[idx]:+.2f} context_d={d_ctx[idx]:+.2f} Δ={d_change_ctx[idx]:+.2f}")


def experiment4_concept_interaction(baseline, context, simple, labels, lines):
    """实验4：概念交互验证"""
    log = lambda s="": (print(s), lines.append(s))

    log("\n" + "=" * 60)
    log("实验4：概念交互验证")
    log("=" * 60)

    # 假说：新提示词让LLM在评分时考虑概念间关系
    # 验证：新提示词下，概念间相关矩阵的结构是否更强

    for name, concepts in [("Baseline", baseline), ("Context-Aware", context), ("Simple", simple)]:
        # 计算概念间相关矩阵
        corr_matrix = np.corrcoef(concepts.T)  # [177, 177]

        # 去掉对角线
        n = corr_matrix.shape[0]
        mask_offdiag = ~np.eye(n, dtype=bool)
        offdiag = corr_matrix[mask_offdiag]

        # 绝对值统计
        abs_mean = np.abs(offdiag).mean()
        abs_std = np.abs(offdiag).std()
        high_corr = (np.abs(offdiag) > 0.5).mean()
        very_high_corr = (np.abs(offdiag) > 0.8).mean()

        log(f"\n  {name} 概念间相关统计:")
        log(f"    概念间|corr|均值: {abs_mean:.4f}")
        log(f"    概念间|corr|标准差: {abs_std:.4f}")
        log(f"    |corr|>0.5比例: {high_corr:.4f}")
        log(f"    |corr|>0.8比例: {very_high_corr:.4f}")

    # 关键对比：Context vs Baseline的概念间相关性变化
    corr_base = np.corrcoef(baseline.T)
    corr_ctx = np.corrcoef(context.T)

    mask_offdiag = ~np.eye(177, dtype=bool)
    diff = np.abs(corr_ctx[mask_offdiag]) - np.abs(corr_base[mask_offdiag])

    log(f"\n  Context vs Baseline 概念间相关性变化:")
    log(f"    |corr|平均变化: {diff.mean():+.4f}")
    log(f"    增强比例: {(diff > 0).mean():.4f}")
    log(f"    减弱比例: {(diff < 0).mean():.4f}")

    if diff.mean() > 0:
        log(f"    → Context-Aware增强了概念间交互（概念更相互依赖）")
    else:
        log(f"    → Context-Aware未增强概念间交互")


def experiment5_pragmatic_separation(baseline, context, simple, labels, adjectives, lines):
    """实验5：语用区分度——FP样本(讨论toxic)在新提示词下是否被更好区分"""
    log = lambda s="": (print(s), lines.append(s))

    log("\n" + "=" * 60)
    log("实验5：语用区分度")
    log("=" * 60)

    from sklearn.linear_model import LogisticRegression

    # 用baseline的LR识别FP/FN
    lr = LogisticRegression(C=10, max_iter=2000, random_state=42)
    lr.fit(baseline, labels)
    pred_base = lr.predict(baseline)

    fp_mask = (pred_base == 1) & (labels == 0)
    tp_mask = (pred_base == 1) & (labels == 1)

    if fp_mask.sum() == 0:
        log("  FP样本数为0，跳过")
        return

    log(f"  Baseline FP样本数: {fp_mask.sum()}")
    log(f"  Baseline TP样本数: {tp_mask.sum()}")

    # 分析FP样本在三种条件下的概念分数变化
    toxic_mask = labels == 1
    nontoxic_mask = labels == 0

    for name, concepts in [("Baseline", baseline), ("Context-Aware", context), ("Simple", simple)]:
        # FP与TP的余弦相似度
        fp_mean = concepts[fp_mask].mean(axis=0)
        tp_mean = concepts[tp_mask].mean(axis=0)
        nontoxic_mean = concepts[nontoxic_mask].mean(axis=0)

        cos_fp_tp = cosine_similarity(fp_mean, tp_mean)
        cos_fp_nt = cosine_similarity(fp_mean, nontoxic_mean)

        log(f"\n  {name} (基于Baseline FP样本):")
        log(f"    FP vs Toxic余弦相似度: {cos_fp_tp:.4f}")
        log(f"    FP vs NonToxic余弦相似度: {cos_fp_nt:.4f}")
        log(f"    FP偏向: {'NonToxic侧' if cos_fp_nt > cos_fp_tp else 'Toxic侧'}")

    # 逐概念分析：FP样本在哪些概念上被修正
    log(f"\n  FP样本逐概念修正分析 (Context - Baseline):")
    diff_fp = context[fp_mask] - baseline[fp_mask]
    diff_tp = context[tp_mask] - baseline[tp_mask]

    d_base = compute_cohens_d(baseline, labels)
    high_d_idx = np.where(np.abs(d_base) > 0.8)[0]

    if len(high_d_idx) > 0:
        log(f"    高区分力概念(|d|>0.8, {len(high_d_idx)}个):")
        log(f"      FP平均变化: {diff_fp[:, high_d_idx].mean():.6f}")
        log(f"      TP平均变化: {diff_tp[:, high_d_idx].mean():.6f}")
        log(f"      修正效果(TP-FP): {diff_tp[:, high_d_idx].mean() - diff_fp[:, high_d_idx].mean():+.6f}")

        # 修正方向是否正确：FP应降分，TP应升分
        fp_decrease = (diff_fp[:, high_d_idx] < 0).mean()
        tp_increase = (diff_tp[:, high_d_idx] > 0).mean()
        log(f"      FP降分比例: {fp_decrease:.4f}")
        log(f"      TP升分比例: {tp_increase:.4f}")

        if fp_decrease > 0.5 and tp_increase > 0.5:
            log(f"      ✅ 新提示词同时降低FP分数和升高TP分数 → 有效修正")
        elif fp_decrease > 0.5:
            log(f"      ⚠️ 新提示词降低FP分数但未升高TP分数 → 部分修正")
        elif tp_increase > 0.5:
            log(f"      ⚠️ 新提示词升高TP分数但未降低FP分数 → 单向效果")
        else:
            log(f"      ❌ 新提示词未有效修正FP/TP分数")


def show_sample_analyses(contents, labels, pragmatic_analyses, n=10, lines=None):
    """展示Stage 1语用分析样例"""
    log = lambda s="": (print(s), lines.append(s))

    log(f"\n  Stage 1语用分析样例 (前{n}条):")
    for i in range(min(n, len(contents))):
        label_str = "有毒" if labels[i] == 1 else "无毒"
        log(f"\n  [{i+1}] [{label_str}] {contents[i][:80]}...")
        log(f"      语用分析: {pragmatic_analyses[i]}")


# =============================================================================
# 主流程
# =============================================================================
def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 60)
    log("Context-Aware Verbalizer 信息增量验证")
    log("=" * 60)
    log(f"模型: {MODEL_NAME}")
    log(f"样本数: {N_SAMPLES}（仅测试集，纯统计分析）")
    log(f"种子: {SEED}")

    # 加载数据
    log("\n>>> 加载数据...")
    test_path = GLM_PATH / "concept_test_glm-4-9b-chat_v1.json"

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 加载形容词
    adj_df = pd.read_csv(ADJ_PATH)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    log(f"  形容词数: {len(adjectives)}")

    # 采样200条（分层采样，保持有毒比例）
    np.random.seed(SEED)
    n_total = len(test_data)
    toxic_idx = [i for i, d in enumerate(test_data) if d["toxic"] == 1]
    nontoxic_idx = [i for i, d in enumerate(test_data) if d["toxic"] == 0]

    n_toxic = int(N_SAMPLES * len(toxic_idx) / n_total)
    n_nontoxic = N_SAMPLES - n_toxic

    sampled_toxic = np.random.choice(toxic_idx, n_toxic, replace=False)
    sampled_nontoxic = np.random.choice(nontoxic_idx, n_nontoxic, replace=False)
    sampled_idx = np.concatenate([sampled_toxic, sampled_nontoxic])
    np.random.shuffle(sampled_idx)

    sampled_test = [test_data[i] for i in sampled_idx]
    test_contents = [d["content"] for d in sampled_test]
    test_labels = np.array([d["toxic"] for d in sampled_test])

    log(f"  采样测试集: {len(sampled_test)}, 有毒比例: {test_labels.mean():.3f}")

    # 加载模型
    log("\n>>> 加载模型...")
    tokenizer, llm = load_model()
    likert_ids = get_first_token_ids(LIKERT_TOKENS, tokenizer)
    log(f"  Verbalizer token IDs: {likert_ids}")

    # ==== Stage 1: 语用分析 ====
    log("\n" + "=" * 60)
    log("Stage 1: 语用分析（200样本）")
    log("=" * 60)

    pragmatic_analyses = run_stage1(tokenizer, llm, test_contents)
    show_sample_analyses(test_contents, test_labels, pragmatic_analyses, n=10, lines=lines)

    # ==== 条件1: Baseline ====
    log("\n" + "=" * 60)
    log("条件1: Baseline（当前verbalizer）")
    log("=" * 60)

    baseline = generate_concept_vectors(
        tokenizer, llm, test_contents, adjectives, adj_definitions,
        likert_ids, mode="baseline"
    )
    log(f"  Baseline概念向量: {baseline.shape}")
    d_base = compute_cohens_d(baseline, test_labels)
    log(f"  平均|d|: {np.abs(d_base).mean():.4f}, |d|>1.0: {(np.abs(d_base)>1.0).sum()}")

    # ==== 条件2: +Pragmatic Context ====
    log("\n" + "=" * 60)
    log("条件2: +Pragmatic Context")
    log("=" * 60)

    context = generate_concept_vectors(
        tokenizer, llm, test_contents, adjectives, adj_definitions,
        likert_ids, mode="context", pragmatic_analyses=pragmatic_analyses
    )
    log(f"  Context概念向量: {context.shape}")
    d_ctx = compute_cohens_d(context, test_labels)
    log(f"  平均|d|: {np.abs(d_ctx).mean():.4f}, |d|>1.0: {(np.abs(d_ctx)>1.0).sum()}")

    # ==== 条件3: +Simple Instruction ====
    log("\n" + "=" * 60)
    log("条件3: +Simple Instruction")
    log("=" * 60)

    simple = generate_concept_vectors(
        tokenizer, llm, test_contents, adjectives, adj_definitions,
        likert_ids, mode="simple"
    )
    log(f"  Simple概念向量: {simple.shape}")
    d_simple = compute_cohens_d(simple, test_labels)
    log(f"  平均|d|: {np.abs(d_simple).mean():.4f}, |d|>1.0: {(np.abs(d_simple)>1.0).sum()}")

    # ==== 五大实验 ====
    experiment1_information_increment(baseline, context, simple, test_labels, adjectives, lines)
    experiment2_fp_discrimination(baseline, context, simple, test_labels, lines)
    experiment3_cohens_d_change(baseline, context, simple, test_labels, adjectives, lines)
    experiment4_concept_interaction(baseline, context, simple, test_labels, lines)
    experiment5_pragmatic_separation(baseline, context, simple, test_labels, adjectives, lines)

    # ==== 最终判断 ====
    log("\n" + "=" * 60)
    log("最终判断")
    log("=" * 60)

    # 信息增量判断
    r_ctx_base, _ = pearsonr(baseline.flatten(), context.flatten())
    r_simple_base, _ = pearsonr(baseline.flatten(), simple.flatten())

    log(f"\n  信息增量指标:")
    log(f"    Context vs Baseline 全局r: {r_ctx_base:.4f}")
    log(f"    Simple vs Baseline 全局r: {r_simple_base:.4f}")

    # 逐维度r<0.8的数量
    low_r_ctx = 0
    low_r_simple = 0
    for i in range(177):
        r_c, _ = pearsonr(baseline[:, i], context[:, i])
        r_s, _ = pearsonr(baseline[:, i], simple[:, i])
        if r_c < 0.8:
            low_r_ctx += 1
        if r_s < 0.8:
            low_r_simple += 1

    log(f"    Context r<0.80维度数: {low_r_ctx}/177")
    log(f"    Simple r<0.80维度数: {low_r_simple}/177")

    # Cohen's d变化
    d_change_ctx = np.abs(d_ctx).mean() - np.abs(d_base).mean()
    d_change_simple = np.abs(d_simple).mean() - np.abs(d_base).mean()

    log(f"\n  Cohen's d变化:")
    log(f"    Context |d|均值变化: {d_change_ctx:+.4f}")
    log(f"    Simple |d|均值变化: {d_change_simple:+.4f}")

    # 综合判断
    log(f"\n  综合判断:")

    info_gain = low_r_ctx > 10  # 至少10个维度有显著信息增量
    d_improve = d_change_ctx > 0.01  # Cohen's d有实质提升

    if info_gain and d_improve:
        log(f"  ✅ Context-Aware Verbalizer有信息增量且提升区分力，值得全量实验")
    elif info_gain and not d_improve:
        log(f"  ⚠️ Context-Aware有信息增量但未提升区分力，需分析增量来源")
    elif not info_gain and d_improve:
        log(f"  ⚠️ Context-Aware提升区分力但无额外信息，可能是校准效果")
    else:
        log(f"  ❌ Context-Aware无信息增量且未提升区分力，方向无效")

    if d_change_ctx > d_change_simple + 0.005:
        log(f"  ✅ 两阶段(Context)显著优于简单指令(Simple)，语用分析有价值")
    elif d_change_simple > d_change_ctx + 0.005:
        log(f"  ⚠️ 简单指令优于两阶段，语用分析Stage 1可能引入噪声")

    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n报告已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
