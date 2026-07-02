"""生成语用功能标注（Chat Template + vLLM）

核心思路：利用LLM的verbalizer token技术，对文本进行语用功能分类标注。
该标注作为CoSMo3D式规范变换的监督信号，辅助区分"讨论毒性"与"实施毒性"。

【背景】
当前概念向量在"语义相关度"空间评估，无法区分：
- 实施毒性（FP高发区："和黑人结婚生的杂种都应该灭杀掉"）
- 讨论毒性（TP高发区："他说黑人是杂种"）
两者语义相关度相似，但语用功能不同。

CoSMo3D思路：学习从"语义相关度空间"到"规范语用空间"的变换，
需要语用功能标签作为监督信号。

【Verbalizer提取流程】
1. 构建Chat Template prompt → LLM推理（max_tokens=1, temperature=0）
2. 提取首token的Top-20 logprobs → exp()转为概率
3. 从概率分布中提取verbalizer token（"0"~"4"）的概率
4. argmax作为语用功能标签，同时保存完整概率分布

【语用功能类别】
0 = 非毒性表达：无毒性意图，正常交流
1 = 实施毒性：直接实施毒性言语行为（如直接侮辱、威胁、煽动、谩骂）
2 = 讨论毒性：讨论、引用、分析或报道毒性话题，但说话者未直接实施毒性
3 = 反讽毒性：通过反讽、谐音、隐喻等间接方式实施毒性
4 = 调侃毒性：轻度过界的戏谑、调侃或冒犯性玩笑

使用示例：
python scripts/generate_pragmatic_function_labels.py --n_samples 500 --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import math
import os
import sys
import random
from pathlib import Path
import json

# AutoDL环境中OMP_NUM_THREADS可能被设为无效值，导致vLLM报错，需清理
if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

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
        description="生成语用功能标注（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--n_samples', type=int, default=500,
                        help='抽样样本数（分层抽样，保持toxic/non-toxic比例），默认500')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表（与generate_adjective_c_r_vllm.py保持一致）
# =============================================================================
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
# 模型加载（与generate_adjective_c_r_vllm.py保持一致）
# =============================================================================
def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer。"""
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
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 提示词定义
# =============================================================================
# system instruction：引导LLM进行语用功能分类
# 关键区分点：
#   - 实施(1) vs 讨论(2)：前者说话者在执行毒性，后者提及毒性话题
#   - 反讽(3) vs 调侃(4)：前者用间接方式实施毒性，后者轻度过界
SYSTEM_INSTRUCTION = (
    "你是一位语用学分析专家，擅长识别文本的言语行为类型和说话者意图。\n\n"
    "请判断文本的语用功能类别：\n"
    "0 = 非毒性表达：无毒性意图，正常交流\n"
    "1 = 实施毒性：直接实施毒性言语行为（如直接侮辱、威胁、煽动、谩骂）\n"
    "2 = 讨论毒性：讨论、引用、分析或报道毒性话题，但说话者未直接实施毒性\n"
    "3 = 反讽毒性：通过反讽、谐音、隐喻等间接方式实施毒性\n"
    "4 = 调侃毒性：轻度过界的戏谑、调侃或冒犯性玩笑\n\n"
    "判断要点：\n"
    "- 区分\"实施\"与\"讨论\"：前者是说话者在执行毒性，后者是提及毒性话题\n"
    "- 反讽和调侃属于间接实施，但反讽更隐蔽，调侃更轻浮\n"
    "- 非毒性文本归为0类\n\n"
    "只回答一个数字（0-4）。"
)


# =============================================================================
# Prompt构建
# =============================================================================
def build_chat_messages(content):
    """构建语用功能分类的Chat Template messages。

    user_content结构：
        文本内容：{content}
        该文本的语用功能类别是？回答：
    """
    user_content = f"文本内容：{content}\n该文本的语用功能类别是？回答： "

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Verbalizer工具
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（去重）。

    verbalizer技术要求LLM的首token为预定义的token（如"0"~"4"），
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


def extract_pragmatic_function(first_token_logprobs, verbalizer_ids):
    """从首token的logprobs中提取语用功能分类结果。

    Args:
        first_token_logprobs: vLLM返回的首token logprobs字典 {token_id: Logprob对象}
        verbalizer_ids: verbalizer token id列表（对应"0"~"4"）

    Returns:
        (label, probs): argmax标签(int), 5类概率列表
    """
    # logprobs → 概率
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    # 提取0~4各类别的概率
    probs = [probs_dict.get(tid, 0.0) for tid in verbalizer_ids]

    # 归一化（只在verbalizer token范围内归一化）
    total = sum(probs) + 1e-8
    probs_normalized = [p / total for p in probs]

    # argmax作为标签
    label = probs.index(max(probs))

    return label, probs_normalized


# =============================================================================
# 抽样函数
# =============================================================================
def stratified_sample(data, n_samples, seed=42):
    """分层抽样：保持toxic/non-toxic比例。

    Args:
        data: 完整数据集
        n_samples: 抽样数量
        seed: 随机种子

    Returns:
        sampled_data: 抽样后的数据（带original_index字段）
    """
    random.seed(seed)

    toxic_indices = [i for i, d in enumerate(data) if d["toxic"] == 1]
    nontoxic_indices = [i for i, d in enumerate(data) if d["toxic"] == 0]

    n_toxic = int(round(n_samples * len(toxic_indices) / len(data)))
    n_nontoxic = n_samples - n_toxic

    # 确保不超过可用样本数
    n_toxic = min(n_toxic, len(toxic_indices))
    n_nontoxic = min(n_nontoxic, len(nontoxic_indices))

    sampled_toxic = random.sample(toxic_indices, n_toxic)
    sampled_nontoxic = random.sample(nontoxic_indices, n_nontoxic)

    all_sampled = sampled_toxic + sampled_nontoxic
    random.shuffle(all_sampled)

    print(f"分层抽样完成:")
    print(f"  总样本: {len(data)} (Toxic={len(toxic_indices)}, Non-Toxic={len(nontoxic_indices)})")
    print(f"  抽样: {len(all_sampled)} (Toxic={n_toxic}, Non-Toxic={n_nontoxic})")
    print(f"  比例保持: Toxic {n_toxic/len(all_sampled)*100:.1f}% vs 原始 {len(toxic_indices)/len(data)*100:.1f}%")

    return all_sampled


# =============================================================================
# 核心流程：生成语用功能标注
# =============================================================================
def generate_pragmatic_labels(data_path, output_path, sampled_indices,
                               tokenizer, llm_model,
                               is_qwen3=False, prompt_suffix=""):
    """生成语用功能标注。

    流程：
    1. 加载数据集
    2. 对抽样样本构建prompt → vLLM推理 → 提取首token概率 → argmax
    3. 保存结果（JSON含完整信息）
    """
    # verbalizer token id
    verbalizer_tokens = ["0", "1", "2", "3", "4"]
    verbalizer_ids = get_first_token_ids(verbalizer_tokens, tokenizer)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # 准备抽样样本
    sampled_data = [(idx, data_set[idx]) for idx in sampled_indices]

    # 批量推理
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # 先构建所有prompt
    print(f"构建 {len(sampled_data)} 个prompt...")
    prompts = []
    for orig_idx, sample in sampled_data:
        content = sample["content"]
        messages = build_chat_messages(content)
        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    # vLLM批量推理
    print(f"vLLM批量推理开始（{len(prompts)}个样本）...")
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=True)

    # 提取结果
    results = []
    for i, (orig_idx, sample) in enumerate(tqdm(sampled_data, desc="提取结果")):
        first_token_logprobs = outputs[i].outputs[0].logprobs[0]
        label, probs = extract_pragmatic_function(first_token_logprobs, verbalizer_ids)

        result_item = {
            "original_index": orig_idx,
            "content": sample["content"],
            "toxic": sample["toxic"],
            "pragmatic_function": label,
            "pragmatic_probs": probs,
        }
        results.append(result_item)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n语用功能标注保存到: {output_path}")
    print(f"样本数: {len(results)}")

    # 打印分布统计
    print(f"\n语用功能分布统计:")
    label_names = ["0=非毒性", "1=实施毒性", "2=讨论毒性", "3=反讽毒性", "4=调侃毒性"]
    for i, name in enumerate(label_names):
        count = sum(1 for r in results if r["pragmatic_function"] == i)
        pct = count / len(results) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")

    # 按toxic标签分层统计
    print(f"\n按toxic标签分层统计:")
    print(f"  {'类别':<15} {'Toxic样本':<12} {'Non-Toxic样本':<15}")
    for i, name in enumerate(label_names):
        t_count = sum(1 for r in results if r["pragmatic_function"] == i and r["toxic"] == 1)
        n_count = sum(1 for r in results if r["pragmatic_function"] == i and r["toxic"] == 0)
        print(f"  {name:<15} {t_count:<12} {n_count:<15}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    # 构建路径（与generate_adjective_c_r_vllm.py保持一致）
    data_path = config.raw_data_path / args.dataset_name / "train.json"
    if not data_path.exists():
        raise ValueError(f"训练数据不存在: {data_path}")

    # 输出路径
    output_dir = config.processed_path / args.dataset_name / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pragmatic_function_labels_{args.n_samples}.json"

    # 打印配置
    print("\n" + "=" * 60)
    print("语用功能标注生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"抽样数量: {args.n_samples}")
    print(f"随机种子: {args.seed}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    # 抽样
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)
    sampled_indices = stratified_sample(data_set, args.n_samples, args.seed)

    # 加载模型
    model_config = get_model_loading_config(args.model_name)
    tokenizer, llm_model, is_qwen3 = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if is_qwen3:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")
    if model_config["prompt_suffix"]:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(model_config['prompt_suffix'])}")

    # 生成标注
    generate_pragmatic_labels(
        data_path=data_path,
        output_path=output_path,
        sampled_indices=sampled_indices,
        tokenizer=tokenizer,
        llm_model=llm_model,
        is_qwen3=is_qwen3,
        prompt_suffix=model_config["prompt_suffix"],
    )

    print("生成完成")


if __name__ == "__main__":
    main()
