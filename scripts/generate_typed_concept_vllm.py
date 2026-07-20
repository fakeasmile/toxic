"""类型感知概念向量生成脚本（Chat Template + vLLM）。

核心改进：不同类型的概念使用不同的提示词模板和verbalizer
  - behavior（行为型）：二元verbalizer "1=否" "2=是"
  - strategy（策略型）：二元verbalizer "1=否" "2=是"
  - evaluation（评价型）：3级危害锚定 "1=不相关" "2=存在但无害" "3=有害表达"
  - discrimination（歧视型）：3级危害锚定 "1=不相关" "2=存在但无害" "3=有害表达"
  - neutral（中性概念）：3级危害锚定 "1=不相关" "2=存在但无害" "3=有害表达"
  - intent（意图概念）：二元verbalizer "1=否" "2=是"
  - effect（效果概念）：二元verbalizer "1=否" "2=是"

标量分数统一为"有害/肯定"概率[0,1]：二元用P(2)，3级用P(3)。
level_probs保留完整原始概率，供下游灵活使用。

使用示例：
    python scripts/generate_typed_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
    python scripts/generate_typed_concept_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# AutoDL环境中OMP_NUM_THREADS可能被设为无效值，导致vLLM报错，需清理
if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import numpy as np
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
        description="类型感知概念向量生成脚本（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help='train:生成训练集的概念向量，test:生成测试集的概念向量')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--adjective_name', type=str, default='toxic_adjectives_v4.csv',
                        help='概念词典文件名，默认toxic_adjectives_v4.csv')
    parser.add_argument('--data_file', type=str, default=None,
                        help='自定义数据文件名（如train_100.json），默认根据mode自动选择')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='快速验证用，0=全量')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen2.5-14B-Instruct": {
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
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载
# =============================================================================
def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
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
# 提示词模板定义
#
# 设计原则（基于GLM-4-9B-Chat特性优化）：
# 1. 结构化：角色设定 → 任务声明 → 回答规则 → 判断要点 → 输出格式
# 2. 正面指令：用"选2的情形/选3的情形"替代"选否"，减少负面表述
# 3. 消除时态偏差：行为型用"实施了"而非"正在发生"（文本是静态产物）
# 4. 统一user prompt结构：类型化标签+定义 → 文本 → 问题 → 输出要求
# 5. 定义前置：先给出概念定义，再呈现文本，便于模型建立判断框架
# =============================================================================

# --- 行为型提示词（二元判断）---
BEHAVIOR_SYSTEM = """你是一位语言分析专家，擅长识别文本中的有害行为。

任务：判断文本是否实施了指定行为。

回答规则：
1 — 否：文本未实施该行为
2 — 是：文本实施了该行为

判断要点：
- "实施"指文本本身构成了该行为的实例，包括直接实施和以隐含、暗示、反讽方式实施
- 选1的情形：仅提及、讨论、引用、分析该行为，而非实际实施
- 选2的情形：文本本身就是该行为的实例，即使以非字面方式呈现
- 只回答一个数字（1或2）"""

BEHAVIOR_USER = """行为概念：{name}
定义：{definition}

文本：{content}

该文本是否实施了上述行为？只回答一个数字（1=否，2=是）："""

# --- 策略型提示词（二元判断）---
STRATEGY_SYSTEM = """你是一位语言分析专家，擅长识别文本中使用的沟通策略和修辞手段。

任务：判断文本是否使用了指定策略。

回答规则：
1 — 否：文本未使用该策略
2 — 是：文本使用了该策略

判断要点：
- "使用"指文本实际采用了该策略进行表达，包括隐蔽和间接使用
- 选1的情形：仅提及或讨论该策略本身，而非实际采用
- 选2的情形：文本实际采用了该策略，需结合语境判断，策略可能以非字面方式呈现
- 只回答一个数字（1或2）"""

STRATEGY_USER = """策略概念：{name}
定义：{definition}

文本：{content}

该文本是否使用了上述策略？只回答一个数字（1=否，2=是）："""

# --- 评价型提示词（3级危害锚定）---
EVALUATION_SYSTEM = """你是一位语言分析专家，擅长评估文本中负面评价的表达方式。

任务：判断文本与指定评价概念之间的关系级别。

回答规则：
1 — 不相关：文本与该概念无关
2 — 存在但无害：文本涉及该概念，但以非有害方式呈现
3 — 有害表达：文本以有害方式体现了该概念

判断要点：
- 选2的情形：讨论概念含义、分析相关现象、引用他人言论后反驳、学术探讨
- 选3的情形：实际使用贬损词汇攻击他人、表达偏见并以此贬低群体、以有害方式体现该概念
- 即使说话者自称无意，若文本客观上构成有害表达，选3
- 只回答一个数字（1、2或3）"""

EVALUATION_USER = """评价概念：{name}
定义：{definition}

文本：{content}

该文本与上述评价概念的关系级别？1=不相关  2=存在但无害  3=有害表达
只回答一个数字："""

# --- 歧视型提示词（3级危害锚定，强调群体针对性）---
DISCRIMINATION_SYSTEM = """你是一位语言分析专家，擅长识别文本中针对特定群体的歧视表达。

任务：判断文本与指定歧视概念之间的关系级别。

回答规则：
1 — 不相关：文本与该概念无关
2 — 存在但无害：文本涉及该概念，但以非有害方式呈现
3 — 有害表达：文本以有害方式体现了该概念，构成对特定群体的歧视

判断要点：
- 选2的情形：讨论歧视现象、反对歧视、学术分析、引用后批判
- 选3的情形：基于群体身份贬低价值、使用侮辱性称呼攻击特定群体、实施排斥或歧视
- 重点关注是否针对特定群体身份（种族、性别、地域等）进行贬低
- 只回答一个数字（1、2或3）"""

DISCRIMINATION_USER = """歧视概念：{name}
定义：{definition}

文本：{content}

该文本与上述歧视概念的关系级别？1=不相关  2=存在但无害  3=有害表达
只回答一个数字："""

# --- 中性概念提示词（3级危害锚定）---
NEUTRAL_SYSTEM = """你是一位语言分析专家，擅长评估文本的言语特征。

任务：判断文本与指定言语特征之间的关系级别。

回答规则：
1 — 不相关：文本不具有该特征
2 — 存在但无害：文本具有该特征，但未以有害方式呈现
3 — 有害表达：文本以有害方式展现了该特征

判断要点：
- 该特征本身是中性的，关键判断其呈现方式是否有害
- 选2的情形：特征存在但不伤人，如表达直接但不构成攻击
- 选3的情形：特征以有害方式呈现，如构成人身攻击或冒犯
- 只回答一个数字（1、2或3）"""

NEUTRAL_USER = """言语特征：{name}
定义：{definition}

文本：{content}

该文本与上述言语特征的关系级别？1=不相关  2=存在但无害  3=有害表达
只回答一个数字："""

# --- 意图概念提示词（二元判断）---
INTENT_SYSTEM = """你是一位语言分析专家，擅长判断文本中说话者的表达意图。

任务：判断说话者是否具有指定意图。

回答规则：
1 — 否：说话者不具有该意图
2 — 是：说话者具有该意图

判断要点：
- 关注说话者的真实意图，综合字面意思和隐含意思判断
- 选2的情形：文本实质上表达了该意图，即使说话者口头否认
- 对于中立或善意意图，若文本确实客观中立或表达善意，选2
- 只回答一个数字（1或2）"""

INTENT_USER = """意图概念：{name}
定义：{definition}

文本：{content}

该文本的说话者是否具有上述意图？只回答一个数字（1=否，2=是）："""

# --- 效果概念提示词（二元判断）---
EFFECT_SYSTEM = """你是一位语言分析专家，擅长评估文本对读者可能产生的影响。

任务：判断文本是否会产生指定效果。

回答规则：
1 — 否：文本不会产生该效果
2 — 是：文本会产生该效果

判断要点：
- 关注文本客观上可能产生的效果，而非说话者的主观意图
- 选2的情形：文本客观上可能产生该效果，即使说话者无意
- "无害效果"指文本不会对他人造成实质性伤害
- 只回答一个数字（1或2）"""

EFFECT_USER = """效果概念：{name}
定义：{definition}

文本：{content}

该文本是否会产生上述效果？只回答一个数字（1=否，2=是）："""


# =============================================================================
# 提示词与Verbalizer注册表
# =============================================================================
PROMPT_REGISTRY = {
    "behavior":       {"system": BEHAVIOR_SYSTEM,       "user": BEHAVIOR_USER,       "verbalizer": ["1", "2"]},
    "strategy":       {"system": STRATEGY_SYSTEM,       "user": STRATEGY_USER,       "verbalizer": ["1", "2"]},
    "evaluation":     {"system": EVALUATION_SYSTEM,     "user": EVALUATION_USER,     "verbalizer": ["1", "2", "3"]},
    "discrimination": {"system": DISCRIMINATION_SYSTEM, "user": DISCRIMINATION_USER, "verbalizer": ["1", "2", "3"]},
    "neutral":        {"system": NEUTRAL_SYSTEM,        "user": NEUTRAL_USER,        "verbalizer": ["1", "2", "3"]},
    "intent":         {"system": INTENT_SYSTEM,         "user": INTENT_USER,         "verbalizer": ["1", "2"]},
    "effect":         {"system": EFFECT_SYSTEM,         "user": EFFECT_USER,         "verbalizer": ["1", "2"]},
}

THREE_LEVEL_TYPES = {"evaluation", "discrimination", "neutral"}
BINARY_TYPES = {"behavior", "strategy", "intent", "effect"}


# =============================================================================
# 概念词典加载
# =============================================================================
def load_concepts(csv_path):
    concepts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["name"]:
                continue
            concepts.append({
                "name": row["name"],
                "type": row["type"],
                "category": row.get("category", "neutral"),
                "definition": row["definition"],
                "prompt_template": row.get("prompt_template", row["type"]),
            })
    return concepts


# =============================================================================
# Prompt构建
# =============================================================================
def build_prompt(content, concept):
    """构建system_prompt和user_prompt。"""
    ptype = concept["prompt_template"]
    reg = PROMPT_REGISTRY[ptype]
    system_prompt = reg["system"]
    user_prompt = reg["user"].format(
        name=concept["name"],
        definition=concept["definition"],
        content=content,
    )
    return system_prompt, user_prompt


# =============================================================================
# 核心流程：类型感知概念向量生成
# =============================================================================
def generate_typed_concept(data_path, output_path, adjective_path,
                           tokenizer, llm_model,
                           is_qwen3=False, prompt_suffix="",
                           num_samples=0):
    """生成类型感知概念向量。

    对数据集中每条文本，遍历所有概念，根据概念类型使用不同的提示词和verbalizer，
    提取概率分布，构建概念向量。

    标量分数：二元用P(2)，3级用P(3)，统一为[0,1]范围。
    level_probs：保留完整原始概率，供下游灵活使用。
    """
    # 加载概念词典
    concepts = load_concepts(adjective_path)
    num_concepts = len(concepts)
    type_counts = {}
    for c in concepts:
        ptype = c["prompt_template"]
        v_type = "3级" if ptype in THREE_LEVEL_TYPES else "二元"
        type_counts.setdefault((ptype, v_type), 0)
        type_counts[(ptype, v_type)] += 1
    print(f"概念总数: {num_concepts}")
    for (ptype, v_type), count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}概念 ({v_type})")

    # 验证所有概念的prompt_template都已在注册表中
    for c in concepts:
        if c["prompt_template"] not in PROMPT_REGISTRY:
            raise ValueError(f"概念'{c['name']}'的prompt_template='{c['prompt_template']}'不在PROMPT_REGISTRY中")

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)
    if num_samples > 0:
        data_set = data_set[:num_samples]
        print(f"快速验证模式: 使用前{num_samples}条样本")
    print(f"数据集大小: {len(data_set)}条")

    # 推理配置
    sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    results = []

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]

        # 为当前文本构建所有概念的prompt
        prompts = []
        for concept in concepts:
            system_prompt, user_prompt = build_prompt(content, concept)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        # 批量推理：一次性送入当前文本的所有概念prompt
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        # 防御性校验
        if len(outputs) != num_concepts:
            raise RuntimeError(f"推理输出数量异常：期望{num_concepts}，实际{len(outputs)}")

        # 解析结果
        concept_scores = [0.0] * num_concepts
        level_probs_list = [None] * num_concepts

        for ci, (output, concept) in enumerate(zip(outputs, concepts)):
            ptype = concept["prompt_template"]
            verbalizer = PROMPT_REGISTRY[ptype]["verbalizer"]

            # 提取首token logprobs
            token_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_token_logprobs = output.outputs[0].logprobs[0]
                for token_id, logprob_info in first_token_logprobs.items():
                    token_text = logprob_info.decoded_token.strip()
                    if token_text in verbalizer:
                        token_logprobs[token_text] = logprob_info.logprob

            # 计算概率（softmax归一化）
            probs = {}
            if token_logprobs:
                max_logprob = max(token_logprobs.values())
                exp_sum = sum(np.exp(lp - max_logprob) for lp in token_logprobs.values())
                for v in verbalizer:
                    if v in token_logprobs:
                        probs[v] = np.exp(token_logprobs[v] - max_logprob) / exp_sum
                    else:
                        probs[v] = 0.0
            else:
                for v in verbalizer:
                    probs[v] = 1.0 / len(verbalizer)

            # 标量分数：统一为"有害/肯定"概率[0,1]
            if ptype in BINARY_TYPES:
                concept_scores[ci] = probs.get("2", 0.0)
                level_probs_list[ci] = [probs.get("1", 0.0), probs.get("2", 0.0)]
            else:
                concept_scores[ci] = probs.get("3", 0.0)
                level_probs_list[ci] = [probs.get("1", 0.0), probs.get("2", 0.0), probs.get("3", 0.0)]

        # 组装结果
        results.append({
            "content": sample["content"],
            "toxic": sample.get("toxic", -1),
            "concept_scores": concept_scores,
            "level_probs": level_probs_list,
        })

    # 保存结果（含概念元信息）
    meta = {
        "num_concepts": num_concepts,
        "concept_names": [c["name"] for c in concepts],
        "concept_types": [c["prompt_template"] for c in concepts],
        "adjective_file": adjective_path.name,
        "num_samples": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_data = {
        "meta": meta,
        "data": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n概念向量已保存至: {output_path}")
    print(f"总概念数: {num_concepts}, 总样本数: {len(results)}")

    # 覆盖率统计
    type_coverage = {ptype: {"total": 0, "covered": 0} for ptype in PROMPT_REGISTRY}
    for item in results:
        for ci, concept in enumerate(concepts):
            ptype = concept["prompt_template"]
            lp = item["level_probs"][ci]
            type_coverage[ptype]["total"] += 1
            if any(p > 0.01 for p in lp):
                type_coverage[ptype]["covered"] += 1

    print("\nVerbalizer覆盖率:")
    for ptype, cov in type_coverage.items():
        if cov["total"] > 0:
            rate = cov["covered"] / cov["total"] * 100
            print(f"  {ptype}: {rate:.2f}% ({cov['covered']}/{cov['total']})")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()
    config = MLPConfig()

    # 构建路径
    data_dir = config.raw_data_path / args.dataset_name
    if args.data_file:
        data_path = data_dir / args.data_file
    else:
        data_path = data_dir / f"{args.mode}.json"

    adjective_path = config.raw_data_path / "adjective" / args.adjective_name
    if not adjective_path.exists():
        raise FileNotFoundError(f"概念词典不存在: {adjective_path}")

    # 输出路径
    adj_suffix = Path(args.adjective_name).stem.replace("toxic_adjectives_", "")
    concept_dir = config.processed_path / args.dataset_name / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_typed_{adj_suffix}.json"

    # 打印配置
    print("\n" + "=" * 60)
    print("类型感知概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"概念词典: {adjective_path.name}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print("=" * 60 + "\n")

    # 加载模型
    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")

    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")

    # 执行概念向量生成
    generate_typed_concept(
        data_path, output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix,
        num_samples=args.num_samples,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
