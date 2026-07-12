"""生成改进3级概念向量 + 语用维度概念（Chat Template + vLLM）

核心改进：
  1. 3级锚点从"主体性"转向"危害性"：
     - 1=不相关, 2=存在但无害, 3=有害表达
     - 解决FN：隐晦毒性不再需要"实施"主体性，"有害表达"即可捕获
     - 解决FP：讨论毒性→"存在但无害"而非"涉及"噪声
  
  2. 新增8个语用维度概念（intent+effect类型），使用二元verbalizer：
     - intent: 攻击意图, 偏见表达, 正当批评(anti), 中立讨论(anti)
     - effect: 贬损效果, 恐吓效果, 煽动效果, 无害效果(anti)
     - 形容词概念回答"有什么属性"，语用概念回答"产生什么效果/有什么意图"
     - 3个anti-pattern概念（正当批评、中立讨论、无害效果）提供FP修正信号

逻辑论证：
  当前P(2)="涉及"是噪声，因为LLM对隐晦毒性也给出P(2)
  新P(2)="存在但无害"直接编码危害判断，是FP的关键修正信号
  新P(3)="有害表达"不要求说话者主体性，隐晦毒性也能被捕获

使用示例：
    # 200样本快速验证
    python scripts/generate_harm_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat --num_samples 200

    # 全量生成
    python scripts/generate_harm_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
    python scripts/generate_harm_concept_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
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
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成改进3级危害锚定概念向量 + 语用维度概念（vLLM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--adjective_name', type=str, default='toxic_adjectives_v2.csv',
                        help='形容词词典文件名')
    parser.add_argument('--pragmatic_csv', type=str, default='mixed_concepts_v3.csv',
                        help='语用维度概念词典文件名')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='仅处理前N个样本（快速验证）')
    return parser.parse_args()


# =============================================================================
# 模型加载配置
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Qwen2.5-14B-Instruct": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Qwen3.5-9B": {"quantization": "fp8", "is_qwen3": True, "is_multimodal": True, "prompt_suffix": ""},
    "glm-4-9b-chat": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": "\n"},
    "deepseek-llm-7b-chat": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Baichuan2-7B-Chat": {"quantization": None, "is_qwen3": False, "is_multimodal": False, "prompt_suffix": ""},
    "Qwen3-8B": {"quantization": None, "is_qwen3": True, "is_multimodal": False, "prompt_suffix": ""},
}


def get_model_loading_config(model_name):
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(f"不支持的模型: {model_name}")
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path, model_name, gpu_memory_utilization=0.85):
    llm_path = model_path / model_name
    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path), trust_remote_code=True, dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True, max_model_len=1024,
        max_num_seqs=64, max_num_batched_tokens=16384,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True

    print(f"Loading vLLM from {llm_path}")
    llm = LLM(**llm_kwargs)
    return tokenizer, llm, model_config["is_qwen3"]


# =============================================================================
# 提示词定义
# =============================================================================

# --- 改进3级：危害锚定（用于形容词） ---
HARM_3LEVEL_SYSTEM = (
    "你是一位语言分析专家，擅长判断文本中概念的表达方式是否具有危害性。\n"
    "你的任务是判断文本与指定概念之间的关系，从三个级别中选择最匹配的：\n"
    "1 — 不相关：文本与该概念描述的内容无关\n"
    "2 — 存在但无害：文本涉及该概念，但以提及、讨论、分析、引用、反驳等非有害方式呈现\n"
    "   例如：报道歧视事件、分析辱骂现象、引用他人攻击性言论、批评偏见观点\n"
    "3 — 有害表达：文本以有害方式表达该概念，包括直接表达和隐含表达\n"
    "   直接表达：说话者明确在辱骂、威胁、歧视、煽动等\n"
    "   隐含表达：通过暗示、反讽、谐音、比喻、刻板印象等方式隐含地表达该概念\n"
    "\"存在但无害\"与\"有害表达\"的关键区分：\n"
    "- 存在但无害：概念出现在文本中，但文本本身不是该概念的有害表达\n"
    "- 有害表达：文本本身就是该概念的有害表达，无论说话者是否以第一人称表达\n"
    "请综合考虑文本的字面意思和可能的隐含意思。\n"
    "只回答一个数字（1、2或3）。"
)

# --- 意图概念提示词（二元） ---
INTENT_SYSTEM = (
    "你是一位语言分析专家，擅长识别文本中说话者的意图。\n"
    "你的任务是判断文本中说话者是否具有指定概念描述的意图。\n"
    "请从以下两个选项中选择：\n"
    "1 — 否：说话者不具有该意图\n"
    "2 — 是：说话者具有该意图\n"
    "判断要点：\n"
    "- 关注说话者的真实意图，而非字面含义\n"
    "- 对于隐含意图（如暗示、反讽），需透过字面理解真实意图\n"
    "- 区分'具有某意图'与'提及某意图'：说话者自身具有该意图选'是'，仅客观讨论选'否'\n"
    "只回答一个数字（1或2）。"
)

# --- 效果概念提示词（二元） ---
EFFECT_SYSTEM = (
    "你是一位语言分析专家，擅长判断文本对读者可能产生的影响。\n"
    "你的任务是判断文本是否产生指定概念描述的效果。\n"
    "请从以下两个选项中选择：\n"
    "1 — 否：文本不会产生该效果\n"
    "2 — 是：文本会产生该效果\n"
    "判断要点：\n"
    "- 关注文本本身对受众产生的影响，而非说话者的主观意图\n"
    "- 即使说话者无意伤害，若文本客观上会产生该效果，也应选 '是'\n"
    "- 对于网络用语和特定语境，需考虑其实际效果而非字面含义\n"
    "只回答一个数字（1或2）。"
)


# =============================================================================
# 提示词构建
# =============================================================================
def build_messages(content, concept_name, concept_def, concept_type):
    """根据概念类型构建不同的Chat Template messages。"""

    if concept_type == "adjective":
        system_msg = HARM_3LEVEL_SYSTEM
        user_lines = [
            f"文本内容：{content}",
            f"概念：{concept_name}",
            f"定义：{concept_def}",
            f"文本与\"{concept_name}\"的关系级别（1=不相关 2=存在但无害 3=有害表达）：回答：",
        ]

    elif concept_type == "intent":
        system_msg = INTENT_SYSTEM
        user_lines = [
            f"文本内容：{content}",
            f"意图：{concept_name}",
            f"定义：{concept_def}",
            f"说话者是否具有上述意图？（1=否 2=是）回答：",
        ]

    elif concept_type == "effect":
        system_msg = EFFECT_SYSTEM
        user_lines = [
            f"文本内容：{content}",
            f"效果：{concept_name}",
            f"定义：{concept_def}",
            f"文本是否产生上述效果？（1=否 2=是）回答：",
        ]

    else:
        raise ValueError(f"未知概念类型: {concept_type}")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


# =============================================================================
# Verbalizer工具
# =============================================================================
def get_first_token_ids(word_list, tokenizer):
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("词表中无有效token")
    return list(dict.fromkeys(token_ids))


def extract_score(first_token_logprobs, level_ids):
    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    level_probs = [probs_dict.get(tid, 0.0) for tid in level_ids]
    total = sum(level_probs) + 1e-8
    level_probs = [p / total for p in level_probs]

    return level_probs[-1], level_probs  # 主信号=最后一级, 完整概率


# =============================================================================
# 核心流程
# =============================================================================
def generate_harm_concept(data_path, output_path, adj_csv_path, pragmatic_csv_path,
                          tokenizer, llm_model, is_qwen3=False, prompt_suffix="",
                          threshold=1e-4, num_samples=None):
    """生成改进3级危害锚定概念向量 + 语用维度概念。"""

    # 加载形容词词典
    adj_df = pd.read_csv(adj_csv_path)
    adj_names = adj_df["chinese"].tolist()
    adj_defs = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adj_names)

    # 加载语用概念词典
    prag_df = pd.read_csv(pragmatic_csv_path)
    prag_names = prag_df["name"].tolist()
    prag_types = prag_df["type"].tolist()
    prag_defs = prag_df["definition"].tolist()

    # 合并：形容词 + 语用概念
    all_names = adj_names + prag_names
    all_defs = adj_defs + prag_defs
    all_types = ["adjective"] * len(adj_names) + prag_types
    num_concepts = len(all_names)

    type_counts = {}
    for t in all_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"概念类型分布: {type_counts}")

    # Verbalizer
    three_ids = get_first_token_ids(["1", "2", "3"], tokenizer)
    binary_ids = get_first_token_ids(["1", "2"], tokenizer)
    print(f"3级Verbalizer token IDs: {three_ids}")
    print(f"二元Verbalizer token IDs: {binary_ids}")

    concept_level_ids = []
    for t in all_types:
        concept_level_ids.append(three_ids if t == "adjective" else binary_ids)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)
    if num_samples:
        data_set = data_set[:num_samples]
        print(f"限制处理前 {num_samples} 个样本")

    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []
    concept_matrix = []

    for sample in tqdm(data_set, desc="Processing"):
        content = sample["content"]
        prompts = []
        for i in range(num_concepts):
            messages = build_messages(content, all_names[i], all_defs[i], all_types[i])
            kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        concept_vector = []
        raw_probs = []
        for i, sample_info in enumerate(outputs):
            first_token_logprobs = sample_info.outputs[0].logprobs[0]
            score, level_probs = extract_score(first_token_logprobs, concept_level_ids[i])
            concept_vector.append(score)
            raw_probs.append(level_probs)

        truncated = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated)

        results.append({
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated,
            "level_probs": raw_probs,
            "concept_names": all_names,
            "concept_types": all_types,
        })

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"概念向量保存到: {output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {num_concepts}]")

    # 按类型统计概率分布
    for ctype in ["adjective", "intent", "effect"]:
        indices = [i for i, t in enumerate(all_types) if t == ctype]
        if not indices:
            continue
        n_levels = 3 if ctype == "adjective" else 2
        level_sums = [0.0] * n_levels
        count = 0
        for r in results:
            for idx in indices:
                probs = r["level_probs"][idx]
                for lv in range(min(n_levels, len(probs))):
                    level_sums[lv] += probs[lv]
                count += 1
        if count > 0:
            level_means = [s / count for s in level_sums]
            if n_levels == 2:
                print(f"  {ctype}: P(1=否)={level_means[0]:.4f}, P(2=是)={level_means[1]:.4f}")
            else:
                print(f"  {ctype}: P(1)={level_means[0]:.4f}, P(2)={level_means[1]:.4f}, P(3)={level_means[2]:.4f}")


def main():
    args = parse_args()
    config = MLPConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    adj_csv_path = config.raw_data_path / "adjective" / args.adjective_name
    pragmatic_csv_path = config.raw_data_path / "adjective" / args.pragmatic_csv

    for p in [data_path, adj_csv_path, pragmatic_csv_path]:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    adj_stem = Path(args.adjective_name).stem
    adj_suffix = adj_stem.replace("toxic_adjectives_", "")
    prag_stem = Path(args.pragmatic_csv).stem

    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_suffix}_{prag_stem}_harm.json"

    print("\n" + "=" * 60)
    print("改进3级危害锚定 + 语用维度概念向量生成")
    print("=" * 60)
    print(f"数据集: {args.dataset_name}, 模型: {args.model_name}")
    print(f"形容词词典: {adj_csv_path.name}")
    print(f"语用概念词典: {pragmatic_csv_path.name}")
    print(f"模式: {args.mode}, 输出: {output_path}")
    if args.num_samples:
        print(f"样本限制: 前{args.num_samples}条")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"模型需要追加prompt后缀: {repr(prompt_suffix)}")

    generate_harm_concept(
        data_path, output_path, adj_csv_path, pragmatic_csv_path,
        tokenizer, llm_model, is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix,
        threshold=1e-4, num_samples=args.num_samples,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
