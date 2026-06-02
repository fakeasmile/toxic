"""生成形容词概念向量（Chat Template,vLLM版本）

【执行流程】
1. 加载vLLM模型和tokenizer
2. 定义verbalizer token词表（Likert等级）和系统指令
3. 遍历数据集中的每条文本：
   a. 对该文本，为所有形容词一次性构建全部Chat Template prompt（无需手动分batch）
   b. vLLM自动调度批量推理，内部处理padding和KV Cache复用
   c. 从推理结果中提取首token的logprobs分布（Top-20，exp转换为概率）
   d. 从概率分布中提取verbalizer token的概率
   e. 归一化计算score（likert: 加权期望），作为该形容词与文本的相关程度
   f. 收集所有形容词的score组成概念向量
4. 保存结果JSON

【配套调试工具】
- inspect_prompt_template_vllm.py：单样本切片（1文本+1形容词），调试提示词和verbalizer
- inspect_verbalizer_coverage_vllm.py：全景扫描（1文本+全部形容词），验证verbalizer覆盖率，即LLM的首token是否将概率质量分配给verbalizer token词表

使用示例：
# Qwen2.5-7B-Instruct-AWQ（AWQ 4-bit预量化权重，量化方式自动检测，无需指定--quantization）
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ
# Qwen3.5-9B（多模态模型，仅使用文本推理；使用FP8动态量化将显存从~18GB降至~9GB；
#   自动：1)禁用thinking 2)跳过视觉编码器节省显存）
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen3.5-9B --quantization fp8
"""

import argparse
import math
import os
import sys
from pathlib import Path
import json

# 修复AutoDL等环境中OMP_NUM_THREADS设置无效导致vLLM初始化失败的问题
# 错误现象：libgomp: Invalid value for environment variable OMP_NUM_THREADS
#           RuntimeError: set_num_threads expects a positive integer
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
        description="生成形容词概念向量（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        default='test',
        help='train:生成训练集的形容词概念向量，test:生成测试集的形容词概念向量'
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
        '--template',
        type=str,
        default='likert',
        help='提示词模板类型（已废弃，保留仅兼容旧实验）'
    )

    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.85,
        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85'
    )

    parser.add_argument(
        '--quantization',
        type=str,
        default=None,
        choices=[None, 'awq', 'fp8', 'gptq'],
        help='量化方法：awq/fp8/gptq，None表示不使用量化（默认）。'
             'awq/gptq：仅适用于预量化权重（如Qwen2.5-7B-Instruct-AWQ），自动从config.json检测，无需手动指定；'
             'fp8：适用于全量权重（如Qwen3.5-9B）的动态量化，无需校准数据，显存减半。'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='采样温度（默认2.0），用于控制概率分布的分散程度'
    )

    return parser.parse_args()


def is_qwen3_plus(model_name: str) -> bool:
    """检测是否为Qwen3+系列模型（包括Qwen3、Qwen3.5等）"""
    return model_name.startswith("Qwen3")


def is_multimodal_model(model_name: str) -> bool:
    """检测是否为多模态模型（如Qwen3.5-9B，含Vision Encoder）
    
    Qwen3.5系列是原生多模态模型，纯文本推理时需设置limit_mm_per_prompt跳过视觉编码器以节省显存。
    Qwen2.5系列是纯文本模型，不需要此参数。
    """
    return model_name.startswith("Qwen3.5")


def detect_quantization_from_config(llm_path: Path) -> str | None:
    """从模型config.json中自动检测量化配置
    
    对于预量化权重（如Qwen2.5-7B-Instruct-AWQ），模型目录中的config.json
    已包含quantization_config字段，此时不需要用户手动指定--quantization参数。
    """
    config_path = llm_path / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        if "quantization_config" in model_config:
            return model_config["quantization_config"].get("quant_method", None)
    return None


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85, quantization: str = None):
    """加载vLLM模型和tokenizer
    
    自动适配不同模型系列：
    - Qwen2.5-7B-Instruct-AWQ：纯文本模型，AWQ 4-bit预量化权重，量化方式自动检测
    - Qwen3.5-9B：多模态模型（仅使用文本推理），当前无官方预量化版本，
      通过--quantization fp8进行FP8动态量化（无需校准数据，显存从~18GB降至~9GB）；
      纯文本推理时跳过视觉编码器以节省显存
    
    关键差异处理：
    1. 量化检测：预量化权重（AWQ/GPTQ）自动从config.json读取quantization_config，无需手动指定；
       全量权重（如Qwen3.5-9B）通过--quantization fp8进行FP8动态量化
    2. 多模态模型：Qwen3.5系列设置limit_mm_per_prompt跳过视觉编码器
    3. 数据类型：Qwen3.5-9B原版权重为bfloat16，在不支持bf16的GPU(如3080Ti)上需使用float16
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    # 量化方式检测：优先使用用户指定的量化方式，否则自动从config.json检测
    effective_quantization = quantization
    if effective_quantization is None:
        auto_detected = detect_quantization_from_config(llm_path)
        if auto_detected:
            effective_quantization = auto_detected
            print(f"检测到模型自带量化配置: {auto_detected}")

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建vLLM加载参数
    # 注意：quantization=None 与不传quantization参数的行为不同，
    # 显式传入None可能被vLLM理解为"不使用量化"，导致预量化权重被当作FP16加载而OOM。
    # 因此只在effective_quantization有值时才传入quantization参数，否则让vLLM自动检测。
    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )
    if effective_quantization is not None:
        llm_kwargs["quantization"] = effective_quantization

    # Qwen3.5系列是原生多模态模型，纯文本推理时设置limit_mm_per_prompt限制多模态输入。
    # 注意：只保留limit_mm_per_prompt，不修改max_num_seqs/max_model_len/enforce_eager。
    # 之前设置的enforce_eager=True会禁用CUDA Graph（vLLM核心加速机制），
    # max_num_seqs=64会强制将236个prompt拆分为4批调度，两者叠加导致推理速度慢6倍。
    # vLLM的PagedAttention采用按需分配KV Cache，增大max_num_seqs不会一次性占满显存。
    if is_multimodal_model(model_name):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt限制多模态输入")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {effective_quantization if effective_quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（目前已保证每个词只会对应一个token id）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return list(dict.fromkeys(token_ids))

def build_chat_messages(instruction, content, adj):
    """
    构建Likert Chat Template
    """
    user_content = f"文本内容：{content}\n形容词：{adj}\n回答： "

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages

def generate_adj_concept(data_path, output_path, csv_output_path, adjective_path, temperature, tokenizer, llm_model, is_qwen3=False, threshold=1e-4):
    # 定义Likert verbalizer token（首token id集合）和提示词指令
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    instruction = ("你是一位语言分析专家。请评估以下文本与形容词的相关程度。\n"
                   "评估等级：\n"
                   "1 = 完全不相关\n"
                   "2 = 不太相关\n"
                   "3 = 有点相关\n"
                   "4 = 比较相关\n"
                   "5 = 非常相关\n"
                   "直接回答数字。")

    # 加载形容词词典
    adjectives = pd.read_csv(adjective_path)["chinese"].tolist()

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
    concept_matrix = []  # 用于保存CSV矩阵 [N, V]

    # 批量推理
    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]  # 文本，后续将匹配V个形容词
        concept_vector = []
        raw_probs = []

        # 在vllm中不手动分批次，构建一条文本+所有形容词的Chat Template
        prompts = []  # 关于当前文本的所有Chat Template
        for adj in adjectives:
            messages = build_chat_messages(instruction, content, adj)

            # 添加<|im_start|>system,<|im_start|>user,<|im_start|>assistant特殊token
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs
            )
            prompts.append(prompt_text)

        # 批量推理
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)  # 表示对文本+所有形容词的推理结果

        for sample_info in outputs:  # sample_info表示文本+某个形容词的推理结果
            # 提取logprobs
            # sample_info.outputs[0]表示推理的第一个结果（n>1:同样的提示词会推理多个结果）
            # logprobs列表，每个元素是字典，{token_id_1: Logprob对象, token_id_2: Logprob对象, ...}，每个元素表示推理生成的token位置的对数概率
            logprobs = sample_info.outputs[0].logprobs
            # 提取生成的首token的对数概率分布，个数由采样器中logprobs参数决定，转化为概率后表示在整个词表上的归一化概率
            last_token_logprobs = logprobs[0]  # 结构：{token_id_1: Logprob对象, token_id_2: Logprob对象, ...}

            # 将last_token_logprobs中的对数概率转化为概率，存储在字典中
            # 结构：{token_id: 概率值}
            probs_dict = {}
            for token_id in last_token_logprobs:
                logprob_obj = last_token_logprobs[token_id]
                probs_dict[token_id] = math.exp(logprob_obj.logprob)

            # 手动应用temperature（vLLM的logprobs返回原始概率，不受temperature影响）
            if temperature > 0:
                # 反推logits（log(p) = logit - log(sum(exp(logits)))，在同一组内归一化常数相同）
                logits = {tid: math.log(p + 1e-10) for tid, p in probs_dict.items()}
                # 应用temperature
                adjusted_logits = {tid: l / temperature for tid, l in logits.items()}
                # 重新softmax
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

        # 防御性校验，确保每条文本输出的形容词概念长度与形容词数量一致
        if len(concept_vector) != len(adjectives):
            raise RuntimeError(
                f"concept_vector 长度异常：期望 {len(adjectives)}，实际 {len(concept_vector)}"
            )

        # 截断极小值：小于threshold的分数设为0
        truncated_vector = []
        for s in concept_vector:
            if abs(s) >= threshold:
                truncated_vector.append(s)
            else:
                truncated_vector.append(0.0)
        concept_matrix.append(truncated_vector)

        # 保存当前文本的推理结果（保留content、toxic和截断后的concept）
        results.append({"content": content, "toxic": sample["toxic"], "concept": truncated_vector})

    # 保存JSON文件（content + toxic）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"形容词概念向量(JSON)保存到: {output_path}")

    # 保存CSV矩阵文件 [N, V]，横轴为文本，纵轴为形容词
    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"形容词概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(adjectives)}] (文本数, 形容词数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")


def main():
    args = parse_args()  # 解析命令行参数

    config = MLPConfig()  # 加载模型配置

    # 动态生成依赖 dataset_name/model_name 的路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"  # 原始数据集目录
    concept_dir = config.processed_path / args.dataset_name / args.model_name  # 概念向量输出目录
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}.csv"
    # 打印配置信息
    print("\n" + "=" * 60)
    print("形容词概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"提示词模板: {args.template}（已废弃）")
    print(f"当前模式: {args.mode}")
    print(f"量化方法: {args.quantization if args.quantization else '无量化'}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"采样温度: {args.temperature}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model = load_vllm_model(config.models_path, args.model_name, args.gpu_memory_utilization, args.quantization)
    qwen3_flag = is_qwen3_plus(args.model_name)
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")
    multimodal_flag = is_multimodal_model(args.model_name)
    if multimodal_flag:
        print(f"检测到多模态模型({args.model_name})，纯文本推理模式下已跳过视觉编码器")
    generate_adj_concept(data_path, output_path, csv_output_path, config.adjective_path, args.temperature, tokenizer, llm_model, is_qwen3=qwen3_flag, threshold=1e-4)

    print("生成完成")

if __name__ == '__main__':
    main()
