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
    parser.add_argument('--sample_idx', type=int, default=0, help='查看第几条样本')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max_tokens', type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    config = PCCGConfig()

    # 加载数据
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    if args.sample_idx >= len(data_set):
        print(f"错误: sample_idx={args.sample_idx} 超出范围 (0~{len(data_set)-1})")
        sys.exit(1)

    sample = data_set[args.sample_idx]
    content = sample["content"]
    toxic = sample["toxic"]

    print("=" * 60)
    print(f"样本 #{args.sample_idx}")
    print(f"标签: {'有毒' if toxic else '无毒'}")
    print(f"文本: {content}")
    print("=" * 60)

    # 构建提示词
    messages = build_reasoning_prompt(content)

    print("\n--- System消息 ---")
    print(messages[0]["content"])
    print(f"\n--- User消息 ---")
    print(messages[1]["content"])

    # 加载模型
    tokenizer, llm_model, is_qwen3 = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )

    chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )

    # 统计token数
    prompt_tokens = len(tokenizer.encode(prompt_text))
    print(f"\n--- 提示词统计 ---")
    print(f"提示词token数: {prompt_tokens}")

    # 推理
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
    )

    outputs = llm_model.generate([prompt_text], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    generated_tokens = len(outputs[0].outputs[0].token_ids)

    print(f"\n--- LLM原始输出 ({generated_tokens} tokens) ---")
    print(generated_text)

    # 解析
    reasoning = parse_reasoning_output(generated_text)

    print(f"\n--- 解析结果 ---")
    parse_ok = all(r["concept"] != "解析失败" for r in reasoning.values())
    print(f"解析状态: {'成功' if parse_ok else '失败'}")

    for dim_name, dim_cfg in DIMENSION_CONFIG.items():
        result = reasoning[dim_name]
        concept = result["concept"]
        reason = result["reason"]
        valid = "✓" if concept in dim_cfg["concepts"] else "✗(不在候选列表)"
        print(f"  {dim_cfg['label']}: {concept} {valid}")
        print(f"    理由: {reason}")

    # 额外：估算全量生成时间
    total_samples = len(data_set)
    # 假设vLLM批量推理的吞吐量约为单条的5-10倍
    estimated_minutes = total_samples * (prompt_tokens + generated_tokens) / (2000 * 8)  # 粗估
    print(f"\n--- 全量估算 ---")
    print(f"数据集总样本数: {total_samples}")
    print(f"单条prompt+output约: {prompt_tokens + generated_tokens} tokens")
    print(f"粗估全量时间: ~{estimated_minutes:.0f} 分钟 (vLLM批量推理)")


if __name__ == '__main__':
    main()
