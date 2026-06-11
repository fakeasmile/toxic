"""生成语用推理结果（vLLM版本）

对每条文本进行7维度结构化语用推理，生成JSON格式的推理结果。
推理结果后续由 encode_reasoning_bge.py 编码为BGE嵌入。

7个推理维度：
1. expression_strategy  - 表达策略
2. implicit_intent      - 隐含意图
3. encoding_strategy    - 编码策略
4. attack_target        - 攻击目标
5. emotional_tone       - 情感基调
6. pragmatic_effect     - 语用效果
7. topic_distinction    - 话题区分

使用示例：
python scripts/generate_pragmatic_reasoning.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ
"""

import argparse
import json
import os
import sys
from pathlib import Path

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.PCCG_config import PCCGConfig
from scripts.generate_adjective_c_r_vllm import (
    MODEL_LOADING_CONFIG, get_model_loading_config, load_vllm_model
)

# =============================================================================
# 维度定义与候选概念（用于解析校验，不再全部写入提示词）
# =============================================================================
DIMENSION_CONFIG = {
    "expression_strategy": {
        "label": "表达策略",
        "concepts": [
            "攻击性的", "侮辱性的", "威胁性的", "挑衅性的", "煽动性的",
            "嘲讽的", "贬低的", "蔑视的", "恐吓的", "羞辱性的",
            "骚扰性的", "粗俗的", "下流的", "无特殊策略"
        ]
    },
    "implicit_intent": {
        "label": "隐含意图",
        "concepts": [
            "反讽的", "阴阳怪气的", "暗示歧视的", "话中有话的",
            "伪善的", "暗含攻击的", "正话反说的", "捧杀的",
            "伪关心的", "煤气灯操纵的", "冷暴力的",
            "似是而非的", "隐晦的", "情感操控的", "无隐含意图"
        ]
    },
    "encoding_strategy": {
        "label": "编码策略",
        "concepts": [
            "使用谐音的", "使用暗语的", "使用缩写的", "使用表情替代的",
            "使用反串的", "使用谐音替换的", "使用拼音的", "无编码策略"
        ]
    },
    "attack_target": {
        "label": "攻击目标",
        "concepts": [
            "地域黑的", "厌女的", "厌男的", "种族偏见的", "排外的",
            "物化女性的", "物化的", "非人化的", "标签化的", "扣帽子的",
            "针对个人的", "针对群体的", "歧视性的", "性别偏见的",
            "性骚扰的", "反农村人的", "反残疾人的",
            "反同性恋的", "反跨性别的", "阶级歧视的", "外貌歧视的",
            "肥胖羞辱的", "学历羞辱的", "职业羞辱的", "年龄歧视的",
            "身体羞辱的", "无特定目标"
        ]
    },
    "emotional_tone": {
        "label": "情感基调",
        "concepts": [
            "仇恨的", "恶意的", "愤怒的", "厌恶的", "怨恨的",
            "偏见的", "刻板的", "极端的", "暴力的", "诅咒死亡的",
            "中性"
        ]
    },
    "pragmatic_effect": {
        "label": "语用效果",
        "concepts": [
            "引战的", "挑拨的", "带节奏的", "分裂性的", "误导性的",
            "破坏性的", "排斥的", "受害者有罪论的",
            "稻草人攻击的", "比烂主义的", "无特殊效果"
        ]
    },
    "topic_distinction": {
        "label": "话题区分",
        "concepts": [
            "讨论敏感话题的", "表达不满的", "立场鲜明的", "情绪化表达的",
            "批评的", "幽默的", "讽刺的", "非敏感话题"
        ]
    }
}

DIMENSION_NAMES = list(DIMENSION_CONFIG.keys())


def build_reasoning_prompt(content: str) -> list:
    """构建语用推理的Chat Template消息（精简版）

    不在提示词中列出所有候选概念，而是给出维度描述让LLM自由选择。
    LLM本身具备足够的语言理解能力，无需枚举候选列表。
    这样可将提示词从~800 tokens压缩到~200 tokens。
    """
    system_msg = (
        "分析文本的语用特征，从7个维度各选1个最匹配的形容词并简述理由。\n"
        "维度：1表达策略 2隐含意图 3编码策略 4攻击目标 5情感基调 6语用效果 7话题区分\n"
        "编码策略指谐音/暗语/缩写/反串等隐晦表达手段；话题区分指区分攻击与讨论敏感话题。\n"
        "严格按JSON输出：\n"
        '{"expression_strategy":{"concept":"形容词","reason":"理由"},'
        '"implicit_intent":{"concept":"形容词","reason":"理由"},'
        '"encoding_strategy":{"concept":"形容词","reason":"理由"},'
        '"attack_target":{"concept":"形容词","reason":"理由"},'
        '"emotional_tone":{"concept":"形容词","reason":"理由"},'
        '"pragmatic_effect":{"concept":"形容词","reason":"理由"},'
        '"topic_distinction":{"concept":"形容词","reason":"理由"}}'
    )

    user_msg = f"文本：{content}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_reasoning_output(text: str) -> dict:
    """解析LLM输出的JSON推理结果

    Returns:
        dict: 7维度的推理结果，每个维度包含concept和reason。
              解析失败时返回全维度为"解析失败"的默认值。
    """
    default = {
        dim: {"concept": "解析失败", "reason": "解析失败"}
        for dim in DIMENSION_NAMES
    }

    # 提取JSON内容：找到第一个{到最后一个}之间的内容
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        return default

    json_str = text[first_brace:last_brace + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return default

    # 验证并填充缺失维度
    result = {}
    for dim in DIMENSION_NAMES:
        if dim in parsed and isinstance(parsed[dim], dict):
            concept = parsed[dim].get("concept", "解析失败")
            reason = parsed[dim].get("reason", "解析失败")
            # 校验concept是否在候选列表中（模糊匹配）
            valid_concepts = DIMENSION_CONFIG[dim]["concepts"]
            if concept not in valid_concepts:
                matched = False
                for vc in valid_concepts:
                    if vc in concept or concept in vc:
                        concept = vc
                        matched = True
                        break
                if not matched:
                    concept = "解析失败"
            result[dim] = {"concept": concept, "reason": reason}
        else:
            result[dim] = {"concept": "解析失败", "reason": "解析失败"}

    return result


def generate_reasoning(data_path, output_path, tokenizer, llm_model, is_qwen3=False):
    """生成语用推理结果（批量推理）

    一次性构建所有prompt，让vLLM自动调度批量推理。

    Args:
        data_path: 原始数据集路径
        output_path: 推理结果输出路径
        tokenizer: tokenizer
        llm_model: vLLM模型
        is_qwen3: 是否为Qwen3+模型
    """
    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    # vLLM采样配置
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.3,
        top_p=0.9,
    )

    # 一次性构建所有prompt
    chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
    prompts = []
    for sample in data_set:
        messages = build_reasoning_prompt(sample["content"])
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs
        )
        prompts.append(prompt_text)

    print(f"共 {len(prompts)} 条prompt，开始批量推理...")

    # 批量推理（vLLM自动调度）
    outputs = llm_model.generate(prompts, sampling_params)

    # 解析结果
    results = []
    parse_fail_count = 0

    for i, (sample, output) in enumerate(zip(data_set, outputs)):
        generated_text = output.outputs[0].text.strip()
        reasoning = parse_reasoning_output(generated_text)

        if any(r["concept"] == "解析失败" for r in reasoning.values()):
            parse_fail_count += 1

        results.append({
            "content": sample["content"],
            "toxic": sample["toxic"],
            "reasoning": reasoning,
        })

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(data_set)
    print(f"语用推理结果保存到: {output_path}")
    print(f"总样本数: {total}, 解析失败数: {parse_fail_count} ({parse_fail_count/total*100:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成语用推理结果（vLLM版本）",
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'], default='test',
        help='train:生成训练集推理结果，test:生成测试集推理结果'
    )
    parser.add_argument(
        '--dataset_name', type=str, required=True,
        help='数据集名称(TOXICN/COLD)'
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        help='LLM模型名称'
    )
    parser.add_argument(
        '--gpu_memory_utilization', type=float, default=0.85,
        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PCCGConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    output_dir = config.processed_path / args.dataset_name / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pragmatic_reasoning_{args.mode}.json"

    print("\n" + "=" * 60)
    print("语用推理生成 - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        config.models_path, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式")

    generate_reasoning(data_path, output_path, tokenizer, llm_model, is_qwen3=qwen3_flag)
    print("生成完成")


if __name__ == '__main__':
    main()
