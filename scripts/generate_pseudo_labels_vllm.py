import argparse
import sys
import re
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.KISCB_config import KISCBConfig

INTENT_MAP = {"贬低": 0, "歧视": 1, "煽动": 2, "物化": 3, "无": 4}
TONE_MAP = {"愤怒敌对": 0, "蔑视轻蔑": 1, "冷漠戏谑": 2, "中性": 3}

SYSTEM_INSTRUCTION = "你是一位语言分析专家。请分析以下文本的攻击意图和情感基调。"

USER_TEMPLATE = """文本：{content}

请从以下选项中选择（可多选）：
攻击意图：[贬低/歧视/煽动/物化/无]
情感基调：[愤怒敌对/蔑视轻蔑/冷漠戏谑/中性]

以JSON格式输出，例如：{{"intent": ["贬低", "歧视"], "tone": "愤怒敌对"}}"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成攻击意图和情感基调伪标签（vLLM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        default='test',
        help='train:生成训练集伪标签，test:生成测试集伪标签'
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
        default='Qwen2.5-7B-Instruct-AWQ',
        help='LLM模型名称'
    )

    parser.add_argument(
        '--quantization',
        type=str,
        default=None,
        choices=[None, 'awq', 'fp8'],
        help='量化方法：awq/fp8，None表示不使用量化（默认）'
    )

    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.85,
        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85'
    )

    return parser.parse_args()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85, quantization: str = None):
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading vLLM model from {llm_path}")
    llm = LLM(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )

    return tokenizer, llm


def parse_llm_response(response_text: str):
    default_intent = [0, 0, 0, 0, 1]
    default_tone = 3

    try:
        json_match = re.search(r'\{[^{}]+\}', response_text)
        if not json_match:
            return default_intent, default_tone

        parsed = json.loads(json_match.group())

        intent_vec = [0, 0, 0, 0, 0]
        raw_intents = parsed.get("intent", [])
        if isinstance(raw_intents, str):
            raw_intents = [raw_intents]
        has_valid_intent = False
        for item in raw_intents:
            if item in INTENT_MAP:
                intent_vec[INTENT_MAP[item]] = 1
                has_valid_intent = True
        if not has_valid_intent:
            intent_vec[4] = 1

        raw_tone = parsed.get("tone", "")
        tone_val = TONE_MAP.get(raw_tone, default_tone)

        return intent_vec, tone_val
    except (json.JSONDecodeError, KeyError, TypeError):
        return default_intent, default_tone


def generate_pseudo_labels(data_path, output_path, tokenizer, llm_model):
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0,
    )

    prompts = []
    for sample in data_set:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": USER_TEMPLATE.format(content=sample["content"])},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=True)

    results = []
    for sample, output in zip(data_set, outputs):
        response_text = output.outputs[0].text.strip()
        intent_vec, tone_val = parse_llm_response(response_text)
        results.append({
            "content": sample["content"],
            "toxic": sample["toxic"],
            "intent": intent_vec,
            "tone": tone_val,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"伪标签保存到: {output_path}")


def main():
    args = parse_args()

    config = KISCBConfig()

    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    output_path = config.processed_path / args.dataset_name / args.model_name / "ki_scb" / f"pseudo_labels_{args.mode}.json"

    print("\n" + "=" * 60)
    print("伪标签生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"当前模式: {args.mode}")
    print(f"量化方法: {args.quantization if args.quantization else '无量化'}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model = load_vllm_model(config.models_path, args.model_name, args.gpu_memory_utilization, args.quantization)
    generate_pseudo_labels(data_path, output_path, tokenizer, llm_model)

    print("生成完成")


if __name__ == '__main__':
    main()
