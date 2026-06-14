"""LLM Zero-Shot分类基线（参考SCBM论文Table 4 Category IV）

直接使用LLM进行有害言论二分类，不经过形容词概念瓶颈。
通过verbalizer技术提取"是"/"否"首token概率作为分类依据。

【对比目的】
SCBM论文Table 4中将GPT-3.5/GPT-4o的zero-shot分类作为Category IV基线，
与SCBM（Category V）对比，证明概念瓶颈方法的优势。
本脚本复现这一对比：LLM直接分类 vs LLM概念评分+MLP。

【执行流程】
1. 加载vLLM模型和tokenizer
2. 构建zero-shot分类提示词（与概念评分提示词风格一致）
3. 提取"是"/"否"首token概率，P("是")>0.5判定为toxic
4. 计算macro-F1等指标，保存结果

使用示例：
python scripts/llm_zero_shot_classify.py --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct
python scripts/llm_zero_shot_classify.py --dataset_name TOXICN --model_name Qwen3.5-9B
python scripts/llm_zero_shot_classify.py --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.metrics import precision_recall_fscore_support, classification_report

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# 复用 generate_adjective_c_r_vllm.py 的模型配置表和加载逻辑
from generate_adjective_c_r_vllm import (
    MODEL_LOADING_CONFIG,
    get_model_loading_config,
    load_vllm_model,
    get_first_token_ids,
)


def build_zero_shot_messages(content):
    """
    构建Zero-Shot分类Chat Template。
    System指令前两句与概念评分提示词保持一致，仅修改任务描述和输出格式。
    """
    instruction = ("你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
                   "在评估时，请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
                   "判断文本是否为有害言论。\n"
                   "直接回答\"是\"或\"否\"。")
    user_content = f"文本内容：{content}\n该文本是否为有害言论？回答： "

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def zero_shot_classify(data_path, output_dir, tokenizer, llm_model, is_qwen3=False, prompt_suffix="", model_name=""):
    """使用LLM进行zero-shot二分类"""

    # 定义verbalizer token："是"/"否"
    yes_tokens = ["是", "是的", "是的。", "Yes", "yes"]
    no_tokens = ["否", "不是", "不是的", "No", "no", "否。"]

    yes_ids = get_first_token_ids(yes_tokens, tokenizer)
    no_ids = get_first_token_ids(no_tokens, tokenizer)

    print(f"Verbalizer token IDs:")
    print(f"  是: {yes_ids} -> {[tokenizer.decode([tid]) for tid in yes_ids]}")
    print(f"  否: {no_ids} -> {[tokenizer.decode([tid]) for tid in no_ids]}")

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    print(f"数据集大小: {len(data_set)}")

    # vLLM采样配置
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20
    )

    # 构建所有prompt
    prompts = []
    for sample in data_set:
        messages = build_zero_shot_messages(sample["content"])
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    # 批量推理
    print("开始批量推理...")
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=True)

    # 提取结果
    results = []
    pred_labels = []
    true_labels = []

    for idx, (sample, output) in enumerate(zip(data_set, outputs)):
        logprobs = output.outputs[0].logprobs
        last_token_logprobs = logprobs[0]

        # 将logprobs转为概率字典
        probs_dict = {}
        for tid, logprob_obj in last_token_logprobs.items():
            probs_dict[tid] = math.exp(logprob_obj.logprob)

        # 提取"是"和"否"的概率
        p_yes = sum(probs_dict.get(tid, 0.0) for tid in yes_ids)
        p_no = sum(probs_dict.get(tid, 0.0) for tid in no_ids)

        # 归一化
        total = p_yes + p_no
        if total > 0:
            p_yes_norm = p_yes / total
            p_no_norm = p_no / total
        else:
            p_yes_norm = 0.5
            p_no_norm = 0.5

        pred_label = 1 if p_yes_norm > 0.5 else 0
        true_label = sample["toxic"]

        pred_labels.append(pred_label)
        true_labels.append(true_label)

        results.append({
            "index": idx,
            "content": sample["content"],
            "true_label": true_label,
            "true_label_name": "Toxic" if true_label == 1 else "Non-Toxic",
            "pred_label": pred_label,
            "pred_label_name": "Toxic" if pred_label == 1 else "Non-Toxic",
            "correct": pred_label == true_label,
            "p_yes": p_yes,
            "p_no": p_no,
            "p_yes_norm": p_yes_norm,
            "p_no_norm": p_no_norm,
            "verbalizer_coverage": p_yes + p_no,
        })

    # 计算指标
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted"
    )

    # verbalizer覆盖率统计
    coverages = [r["verbalizer_coverage"] for r in results]
    avg_coverage = np.mean(coverages)
    miss_rate = sum(1 for c in coverages if c < 0.01) / len(coverages)

    # 打印结果
    report_str = classification_report(true_labels, pred_labels, target_names=["Non-Toxic", "Toxic"])
    print("\n" + "=" * 60)
    print(f"LLM Zero-Shot分类结果 ({args.model_name})")
    print("=" * 60)
    print(f"精确率 (Precision - Macro): {precision_macro:.4f}")
    print(f"召回率 (Recall - Macro):    {recall_macro:.4f}")
    print(f"F1 分数 (F1 Score - Macro): {f1_macro:.4f}")
    print(f"Verbalizer覆盖率: {avg_coverage:.4f} (遗漏率: {miss_rate*100:.2f}%)")
    print("------------------------------")
    print("详细分类报告:")
    print(report_str)
    print("=" * 60)

    # 保存结果（与experiments/test_results格式一致）
    # 1. metrics.json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "verbalizer_coverage": float(avg_coverage),
            "verbalizer_miss_rate": float(miss_rate),
        }, f, ensure_ascii=False, indent=4)
    print(f"指标保存到: {metrics_path}")

    # 2. classification_report.txt
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"LLM Zero-Shot分类结果 ({args.model_name})\n")
        f.write("=" * 30 + "\n")
        f.write(f"精确率 (Precision - Macro): {precision_macro:.4f}\n")
        f.write(f"召回率 (Recall - Macro):    {recall_macro:.4f}\n")
        f.write(f"F1 分数 (F1 Score - Macro): {f1_macro:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("详细分类报告:\n")
        f.write(report_str + "\n")
        f.write("=" * 30 + "\n")
    print(f"分类报告保存到: {report_path}")

    # 3. predictions.json（与experiments格式一致）
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"预测结果保存到: {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Zero-Shot分类基线")
    parser.add_argument("--dataset_name", type=str, default="TOXICN",
                        choices=["TOXICN", "COLD"], help="数据集名称")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct",
                        help="LLM模型名称")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["train", "test"], help="数据集划分")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU显存占用比例")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    output_dir = config.processed_path / args.dataset_name / args.model_name / f"zero_shot_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("LLM Zero-Shot分类基线 - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"数据集划分: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"输出目录: {output_dir}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(config.models_path, args.model_name, args.gpu_memory_utilization)
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")
    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")

    zero_shot_classify(data_path, output_dir, tokenizer, llm_model, is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, model_name=args.model_name)
