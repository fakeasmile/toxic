"""生成推理模式概念向量（Chat Template, vLLM版本）

【核心创新】
将概念从"形容词"（描述文本是什么样）替换为"推理模式"（描述文本如何传达毒性）。
- 形容词概念：问"这段文本有多aggressive？"→ 对隐式毒性无效
- 推理模式概念：问"这段文本是否使用了反讽暗示？"→ 直接捕获毒性推理机制

【执行流程】
1. 加载vLLM模型和tokenizer
2. 定义verbalizer token词表（Likert等级）和系统指令
3. 遍历数据集中的每条文本：
   a. 对该文本，为所有推理模式一次性构建全部Chat Template prompt
   b. vLLM自动调度批量推理，内部处理padding和KV Cache复用
   c. 从推理结果中提取首token的logprobs分布（Top-20，exp转换为概率）
   d. 从概率分布中提取verbalizer token的概率
   e. 归一化计算score（likert: 加权期望），作为该推理模式在文本中的使用程度
   f. 收集所有推理模式的score组成概念向量
4. 保存结果JSON

使用示例：
python scripts/generate_reasoning_concept_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ
python scripts/generate_reasoning_concept_vllm.py --mode test --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ
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

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成推理模式概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        default='test',
        help='train:生成训练集的推理模式概念向量，test:生成测试集的推理模式概念向量'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='数据集名称(TOXICN/COLD)'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='LLM模型名称'
    )

    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.85,
        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='采样温度（默认2.0），用于控制概率分布的分散程度'
    )

    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct-AWQ": {
        "quantization": "awq",
        "is_qwen3": False,
        "is_multimodal": False,
    },
    "Qwen2.5-7B-Instruct-GPTQ-Int8": {
        "quantization": "gptq",
        "is_qwen3": False,
        "is_multimodal": False,
    },
    "Qwen2.5-3B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
    },
    "Qwen3.5-9B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": True,
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
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
    """加载vLLM模型和tokenizer"""
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
        max_num_seqs=256,
        max_num_batched_tokens=2048,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt限制多模态输入")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
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
        raise ValueError("get_first_token_ids ERROR")

    return list(dict.fromkeys(token_ids))


def build_chat_messages(instruction, content, pattern_name, pattern_definition):
    """
    构建推理模式Likert Chat Template

    核心变化：从"评估文本与形容词的相关程度"变为"评估文本使用该表达策略的程度"
    """
    user_content = f"文本内容：{content}\n表达策略：{pattern_name}\n策略说明：{pattern_definition}\n回答： "

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_reasoning_concept(data_path, output_path, csv_output_path, pattern_path,
                                temperature, tokenizer, llm_model, is_qwen3=False, threshold=1e-4):
    """生成推理模式概念向量。

    与形容词概念向量的核心区别：
    1. 概念类型：推理模式（表达策略）而非形容词（描述属性）
    2. 评估维度：文本"使用"该策略的程度，而非与形容词的"相关"程度
    3. 提示词：包含策略名称和详细定义，帮助LLM理解推理机制
    """
    # 定义Likert verbalizer token和推理模式专用提示词
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    instruction = ("你是一位语言分析专家。请评估以下文本在多大程度上使用了所描述的表达策略。\n"
                   "评估等级：\n"
                   "1 = 完全未使用\n"
                   "2 = 很少使用\n"
                   "3 = 有所使用\n"
                   "4 = 较多使用\n"
                   "5 = 大量使用\n"
                   "直接回答数字。")

    # 加载推理模式词典
    patterns_df = pd.read_csv(pattern_path)
    pattern_names = patterns_df["name"].tolist()
    pattern_definitions = patterns_df["definition"].tolist()

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # vLLM采样配置
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20
    )

    results = []
    concept_matrix = []

    # 批量推理
    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]
        concept_vector = []
        raw_probs = []

        # 构建一条文本+所有推理模式的Chat Template
        prompts = []
        for pattern_name, pattern_def in zip(pattern_names, pattern_definitions):
            messages = build_chat_messages(instruction, content, pattern_name, pattern_def)

            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs
            )
            prompts.append(prompt_text)

        # 批量推理
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        for sample_info in outputs:
            logprobs = sample_info.outputs[0].logprobs
            last_token_logprobs = logprobs[0]

            probs_dict = {}
            for token_id in last_token_logprobs:
                logprob_obj = last_token_logprobs[token_id]
                probs_dict[token_id] = math.exp(logprob_obj.logprob)

            # 手动应用temperature
            if temperature > 0:
                logits = {tid: math.log(p + 1e-10) for tid, p in probs_dict.items()}
                adjusted_logits = {tid: l / temperature for tid, l in logits.items()}
                max_logit = max(adjusted_logits.values())
                exp_sum = sum(math.exp(l - max_logit) for l in adjusted_logits.values())
                probs_dict = {tid: math.exp(l - max_logit) / exp_sum for tid, l in adjusted_logits.items()}

            # 提取1-5等级的概率
            level_probs = []
            for tid in likert_ids:
                level_probs.append(probs_dict.get(tid, 0.0))

            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            level_probs = torch.tensor(level_probs)
            total_level_prob = level_probs.sum() + 1e-8
            score = (weights * level_probs / total_level_prob).sum().item()
            raw_probs.append(level_probs.tolist())

            concept_vector.append(score)

        # 防御性校验
        if len(concept_vector) != len(pattern_names):
            raise RuntimeError(
                f"concept_vector 长度异常：期望 {len(pattern_names)}，实际 {len(concept_vector)}"
            )

        # 截断极小值
        truncated_vector = []
        for s in concept_vector:
            if abs(s) >= threshold:
                truncated_vector.append(s)
            else:
                truncated_vector.append(0.0)
        concept_matrix.append(truncated_vector)

        results.append({"content": content, "toxic": sample["toxic"], "concept": truncated_vector})

    # 保存JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"推理模式概念向量(JSON)保存到: {output_path}")

    # 保存CSV矩阵文件
    df = pd.DataFrame(concept_matrix, columns=pattern_names)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"推理模式概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(pattern_names)}] (文本数, 推理模式数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")


def main():
    args = parse_args()

    config = MLPConfig()

    # 动态生成路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    # 推理模式概念向量输出到独立子目录 reasoning_patterns
    concept_dir = config.processed_path / args.dataset_name / args.model_name / "reasoning_patterns"
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}.csv"

    # 推理模式词典路径
    pattern_path = config.raw_data_path / "adjective" / "reasoning_patterns.csv"

    print("\n" + "=" * 60)
    print("推理模式概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"采样温度: {args.temperature}")
    print(f"数据集路径: {data_path}")
    print(f"推理模式词典: {pattern_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")

    generate_reasoning_concept(
        data_path, output_path, csv_output_path, pattern_path,
        args.temperature, tokenizer, llm_model,
        is_qwen3=qwen3_flag, threshold=1e-4
    )

    print("生成完成")


if __name__ == '__main__':
    main()
