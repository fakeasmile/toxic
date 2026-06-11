"""语用推理切片调试工具

单样本快速验证提示词效果和LLM输出质量。
类似于 inspect_prompt_template_vllm.py 的定位。

使用示例：
python scripts/inspect_pragmatic_reasoning.py --model_name Qwen2.5-7B-Instruct-AWQ --sample_idx 0 --dataset_name TOXICN
"""

import argparse
import json
import sys
from pathlib import Path
import os

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.PCCG_config import PCCGConfig
from scripts.generate_adjective_c_r_vllm import (
    MODEL_LOADING_CONFIG, get_model_loading_config, load_vllm_model
)
from scripts.generate_pragmatic_reasoning import (
    DIMENSION_CONFIG, DIMENSION_NAMES, build_reasoning_prompt, parse_reasoning_output
)


def parse_args():
    parser = argparse.ArgumentParser(description="语用推理切片调试工具")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='TOXICN')
    parser.add_argument('--sample_idx', type=int, default=None, help='查看第几条样本（不指定则随机抽样）')
    parser.add_argument('--num_samples', type=int, default=5, help='随机抽样数量')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max_tokens', type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    config = PCCGConfig()

    # 加载数据
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # 选择样本：指定索引 or 随机抽样（保证有毒/无毒各半）
    import random
    random.seed(42)

    if args.sample_idx is not None:
        if args.sample_idx >= len(data_set):
            print(f"错误: sample_idx={args.sample_idx} 超出范围 (0~{len(data_set)-1})")
            sys.exit(1)
        samples = [(args.sample_idx, data_set[args.sample_idx])]
    else:
        toxic_indices = [i for i, s in enumerate(data_set) if s["toxic"] == 1]
        nontoxic_indices = [i for i, s in enumerate(data_set) if s["toxic"] == 0]
        n_half = args.num_samples // 2
        n_half = min(n_half, len(toxic_indices), len(nontoxic_indices))
        selected = random.sample(toxic_indices, n_half) + random.sample(nontoxic_indices, n_half)
        samples = [(i, data_set[i]) for i in selected]

    # 加载模型（一次性）
    tokenizer, llm_model, is_qwen3 = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
    )
    chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}

    # 批量构建prompt
    prompts = []
    for idx, sample in samples:
        messages = build_reasoning_prompt(sample["content"])
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompts.append(prompt_text)

    # 批量推理
    outputs = llm_model.generate(prompts, sampling_params)

    # 统计
    total_dims = 0
    hit_dims = 0
    sample_results = []

    for (idx, sample), prompt_text, output in zip(samples, prompts, outputs):
        content = sample["content"]
        toxic = sample["toxic"]
        generated_text = output.outputs[0].text.strip()
        generated_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(tokenizer.encode(prompt_text))

        reasoning = parse_reasoning_output(generated_text)
        hit = sum(1 for r in reasoning.values() if r["concept"] != "解析失败")
        total = len(reasoning)
        total_dims += total
        hit_dims += hit

        sample_results.append({
            "idx": idx, "content": content, "toxic": toxic,
            "reasoning": reasoning, "hit": hit, "total": total,
            "prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
            "raw_output": generated_text,
        })

    # 打印结果
    print("\n" + "=" * 60)
    print(f"语用推理切片验证 ({len(samples)} 条样本)")
    print("=" * 60)

    for sr in sample_results:
        label = "有毒" if sr["toxic"] else "无毒"
        status = f"{sr['hit']}/{sr['total']}命中"
        print(f"\n--- 样本 #{sr['idx']} [{label}] {status} ---")
        print(f"文本: {sr['content'][:80]}")
        for dim_name, dim_cfg in DIMENSION_CONFIG.items():
            r = sr["reasoning"][dim_name]
            mark = "✓" if r["concept"] != "解析失败" else "✗"
            print(f"  {dim_cfg['label']}: {r['concept']} {mark}")
            if r["concept"] == "解析失败":
                print(f"    理由: {r['reason'][:60]}")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    print(f"样本数: {len(samples)}")
    print(f"维度命中率: {hit_dims}/{total_dims} = {hit_dims/total_dims*100:.1f}%")
    full_hit = sum(1 for sr in sample_results if sr["hit"] == sr["total"])
    print(f"完全命中(7/7): {full_hit}/{len(samples)} = {full_hit/len(samples)*100:.1f}%")

    # 各维度命中率
    print("\n各维度命中率:")
    for dim_name, dim_cfg in DIMENSION_CONFIG.items():
        dim_hit = sum(1 for sr in sample_results if sr["reasoning"][dim_name]["concept"] != "解析失败")
        print(f"  {dim_cfg['label']}: {dim_hit}/{len(samples)} = {dim_hit/len(samples)*100:.1f}%")

    # token统计
    avg_prompt = sum(sr["prompt_tokens"] for sr in sample_results) / len(sample_results)
    avg_output = sum(sr["generated_tokens"] for sr in sample_results) / len(sample_results)
    print(f"\n平均prompt tokens: {avg_prompt:.0f}")
    print(f"平均output tokens: {avg_output:.0f}")

    # 全量估算
    total_samples = len(data_set)
    estimated_minutes = total_samples * (avg_prompt + avg_output) / (2000 * 8)
    print(f"全量({total_samples}条)预估时间: ~{estimated_minutes:.0f} 分钟")

    # 首次使用时显示提示词
    if args.sample_idx is not None:
        messages = build_reasoning_prompt(samples[0][1]["content"])
        print(f"\n--- 提示词内容 ---")
        print(messages[0]["content"])


if __name__ == '__main__':
    main()
