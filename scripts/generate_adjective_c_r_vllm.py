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
# Qwen2.5-7B-Instruct（全量加载，不量化）
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct
# Qwen3.5-9B（多模态模型，仅使用文本推理；全量加载不量化；
#   自动：1)禁用thinking 2)跳过视觉编码器节省显存）
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen3.5-9B
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
# 模型加载配置表
# =============================================================================
# 所有模型相关的加载参数（量化方式、是否多模态、是否 Qwen3+ 等）均集中在
# 此配置表中，以保证 LLM 切换对后续的概念向量生成流程完全透明。
#
# 如需新增模型，只需在本字典中增加一个条目即可，通常无需修改核心推理逻辑：
#
#   "<模型目录名>": {
#       "quantization": None,      # vLLM 量化方式：None / "awq" / "fp8" / "gptq"
#       "is_qwen3": False,         # 是否为 Qwen3+：决定是否禁用 thinking 模式
#       "is_multimodal": False,    # 是否为多模态模型：决定是否跳过视觉编码器
#   }
#
# 若新模型存在目前配置表无法覆盖的特殊加载需求，再视情况扩展 load_vllm_model。
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",       # Qwen首token带空格，已在提示词末尾加空格处理
    },
    "Qwen3.5-9B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": True,
        "prompt_suffix": "",
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "\n",     # GLM-4首token为\n，追加\n使其直接输出数字
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
    """加载vLLM模型和tokenizer
    
    所有模型差异（量化方式、多模态处理、Qwen3+ 标志）均从
    MODEL_LOADING_CONFIG 中读取，保证新增模型时只需改配置表。
    """
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
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    if is_multimodal:
        # 纯文本推理时跳过视觉编码器，释放显存给KV Cache
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt + language_model_only")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


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

def build_chat_messages(instruction, content, adj, adj_definition=None):
    """
    构建Likert Chat Template
    """
    if adj_definition:
        user_content = f"文本内容：{content}\n形容词：{adj}\n定义：{adj_definition}\n该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： "
    else:
        user_content = f"文本内容：{content}\n形容词：{adj}\n该文本在多大程度上体现了\"{adj}\"所描述的特征？回答： "

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages

def generate_adj_concept(data_path, output_path, csv_output_path, adjective_path, temperature, tokenizer, llm_model, is_qwen3=False, prompt_suffix="", threshold=1e-4):
    """生成形容词概念向量。

    此函数对底层 LLM 完全透明：无论加载的是 Qwen2.5、Qwen3.5 还是后续新增的
    DeepSeek 等模型，核心推理流程（构建 prompt -> vLLM 批量推理 -> 提取首 token
    概率 -> Likert 加权期望）均保持一致。唯一的模型相关参数是 is_qwen3，它仅
    控制 tokenizer 的 enable_thinking 开关，不影响概念分数的计算逻辑。
    """
    # 定义Likert verbalizer token（首token id集合）和提示词指令
    likert_tokens = ["1", "2", "3", "4", "5"]
    likert_ids = get_first_token_ids(likert_tokens, tokenizer)

    instruction = ("你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
                   "在评估时，请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
                   "判断文本是否体现了该形容词所描述的特征。\n"
                   "请用1到5的数字评估相关程度，1表示完全不相关，5表示非常相关。只回答一个数字。")

    # 加载形容词词典（含定义）
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

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

    for sample in tqdm(data_set, desc="Processing samples"):
        content = sample["content"]

        # 为当前文本构建所有形容词的prompt
        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            messages = build_chat_messages(instruction, content, adj, adj_def)

            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        # 批量推理：一次性送入当前文本的所有prompt
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        raw_probs = []

        for sample_info in outputs:
            logprobs = sample_info.outputs[0].logprobs
            last_token_logprobs = logprobs[0]

            # 将logprobs转为概率字典
            probs_dict = {}
            for token_id in last_token_logprobs:
                logprob_obj = last_token_logprobs[token_id]
                probs_dict[token_id] = math.exp(logprob_obj.logprob)

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
        if len(concept_vector) != num_adjs:
            raise RuntimeError(
                f"concept_vector 长度异常：期望 {num_adjs}，实际 {len(concept_vector)}"
            )

        # 截断极小值
        truncated_vector = []
        for s in concept_vector:
            if abs(s) >= threshold:
                truncated_vector.append(s)
            else:
                truncated_vector.append(0.0)
        concept_matrix.append(truncated_vector)

        results.append({
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "likert_probs": raw_probs
        })

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
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}.csv"
    # 打印配置信息
    print("\n" + "=" * 60)
    print("形容词概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"采样温度: {args.temperature}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(config.models_path, args.model_name, args.gpu_memory_utilization)
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")
    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")
    generate_adj_concept(data_path, output_path, csv_output_path, config.adjective_path, args.temperature, tokenizer, llm_model, is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=1e-4)

    print("生成完成")

if __name__ == '__main__':
    main()
