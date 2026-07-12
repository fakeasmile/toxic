"""类型感知概念向量生成脚本（Chat Template + vLLM）。

核心改进：不同类型的概念使用不同的提示词模板和verbalizer
  - behavior（行为型）：二元verbalizer "1=否" "2=是"
  - strategy（策略型）：二元verbalizer "1=否" "2=是"
  - evaluation（评价型）：3级危害锚定 "1=不相关" "2=存在但无害" "3=有害表达"
  - discrimination（歧视型）：3级危害锚定 "1=不相关" "2=存在但无害" "3=有害表达"
  - neutral（中性概念）：3级危害锚定 "1=不相关" "2=存在但无害" "3=有害表达"
  - intent（意图概念）：二元verbalizer "1=否" "2=是"
  - effect（效果概念）：二元verbalizer "1=否" "2=是"

使用示例：
    # 全量生成（训练集）
    python scripts/generate_typed_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat

    # 快速验证（200样本）
    python scripts/generate_typed_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat --num_samples 200

    # 生成测试集
    python scripts/generate_typed_concept_vllm.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================
# 提示词模板定义
# ============================================================

# 行为型提示词（二元判断）
BEHAVIOR_SYSTEM = """你是一位语言分析专家，擅长识别文本中的有害行为。
你的任务是判断文本中是否正在发生指定行为。

回答规则：
1 — 否：文本中未发生该行为
2 — 是：文本中正在发生该行为

判断要点：
- 关注文本是否实际实施了该行为，而非仅仅提及或讨论
- 即使行为以隐含、暗示、反讽方式表达，只要实质上构成该行为，就选"是"
- 报道、引用、分析、反驳他人该行为的，选"否"
- 只回答一个数字（1或2）"""

BEHAVIOR_USER_TEMPLATE = """请判断以下文本中是否正在发生"{adj_chinese}"（{definition}）：

文本：{content}

只回答一个数字（1=否，2=是）："""

# 策略型提示词（二元判断）
STRATEGY_SYSTEM = """你是一位语言分析专家，擅长识别文本中的沟通策略和修辞手段。
你的任务是判断文本是否使用了指定的沟通策略。

回答规则：
1 — 否：文本未使用该策略
2 — 是：文本使用了该策略

判断要点：
- 关注文本的实际表达策略，而非字面含义
- 策略可能以隐蔽方式使用，需结合语境判断
- 仅引用或讨论该策略但不实际使用的，选"否"
- 只回答一个数字（1或2）"""

STRATEGY_USER_TEMPLATE = """请判断以下文本是否使用了"{adj_chinese}"策略（{definition}）：

文本：{content}

只回答一个数字（1=否，2=是）："""

# 评价型提示词（3级危害锚定）
EVALUATION_SYSTEM = """你是一位语言分析专家，擅长评估文本中负面评价的表达方式。
你的任务是判断文本与指定评价概念之间的关系级别。

回答规则：
1 — 不相关：文本与该评价概念无关
2 — 存在但无害：文本中存在该评价概念，但以非有害方式呈现（如讨论、引用、分析、反驳）
3 — 有害表达：文本以有害方式表达该评价（即使隐含、暗示、反讽也算）

判断要点：
- "存在但无害"指文本虽然涉及该概念，但说话者并非在实施有害表达
  例如：讨论贬义词汇的含义、分析偏见现象、引用他人言论后反驳
- "有害表达"指文本本身就是该评价的有害实现
  例如：实际使用贬义词汇攻击他人、表达偏见并以此贬低群体
- 即使说话者无意伤害，若文本客观上以有害方式表达该评价，也应选3
- 只回答一个数字（1、2或3）"""

EVALUATION_USER_TEMPLATE = """请判断以下文本与"{adj_chinese}"（{definition}）的关系级别：

文本：{content}

1=不相关  2=存在但无害  3=有害表达
只回答一个数字："""

# 歧视型提示词（3级危害锚定，强调群体针对性）
DISCRIMINATION_SYSTEM = """你是一位语言分析专家，擅长识别文本中针对特定群体的歧视表达。
你的任务是判断文本与指定歧视概念之间的关系级别。

回答规则：
1 — 不相关：文本与该歧视概念无关
2 — 存在但无害：文本中存在该歧视概念，但以非有害方式呈现（如讨论歧视现象、反对歧视、学术分析）
3 — 有害表达：文本以有害方式表达该歧视（针对特定群体的贬低、排斥、侮辱）

判断要点：
- "存在但无害"指文本虽然涉及歧视话题，但说话者并非在实施歧视
  例如：讨论性别歧视现象、反对地域偏见、分析歧视的社会原因
- "有害表达"指文本本身构成对特定群体的歧视性攻击
  例如：基于身份贬低群体价值、使用侮辱性称呼攻击特定群体
- 重点关注是否针对特定群体身份（种族、性别、地域等）进行贬低
- 只回答一个数字（1、2或3）"""

DISCRIMINATION_USER_TEMPLATE = """请判断以下文本与"{adj_chinese}"（{definition}）的关系级别：

文本：{content}

1=不相关  2=存在但无害  3=有害表达
只回答一个数字："""

# 中性概念提示词（3级危害锚定）
NEUTRAL_SYSTEM = """你是一位语言分析专家，擅长评估文本的言语特征。
你的任务是判断文本与指定言语特征之间的关系级别。

回答规则：
1 — 不相关：文本不具有该特征
2 — 存在但无害：文本具有该特征，但未以有害方式呈现
3 — 有害表达：文本以有害方式展现该特征

判断要点：
- 该特征本身是中性的，需判断其呈现方式是否有害
- 例如"粗鲁的"可能只是表达方式直接但不伤人（选2），也可能构成人身攻击（选3）
- 只回答一个数字（1、2或3）"""

NEUTRAL_USER_TEMPLATE = """请判断以下文本与"{adj_chinese}"（{definition}）的关系级别：

文本：{content}

1=不相关  2=存在但无害  3=有害表达
只回答一个数字："""

# 意图概念提示词（二元判断）
INTENT_SYSTEM = """你是一位语言分析专家，擅长判断文本中说话者的表达意图。
你的任务是判断说话者是否具有指定的表达意图。

回答规则：
1 — 否：说话者不具有该意图
2 — 是：说话者具有该意图

判断要点：
- 关注说话者的真实意图，而非字面含义
- 即使说话者否认，若文本实质上表达了该意图，也应选"是"
- 对于中立/善意意图，若文本确实客观中立或表达善意，选"是"
- 只回答一个数字（1或2）"""

INTENT_USER_TEMPLATE = """请判断以下文本的说话者是否具有"{adj_chinese}"（{definition}）：

文本：{content}

只回答一个数字（1=否，2=是）："""

# 效果概念提示词（二元判断）
EFFECT_SYSTEM = """你是一位语言分析专家，擅长评估文本对读者可能产生的影响。
你的任务是判断文本是否会产生指定效果。

回答规则：
1 — 否：文本不会产生该效果
2 — 是：文本会产生该效果

判断要点：
- 关注文本客观上可能产生的效果，而非说话者的主观意图
- 即使说话者无意，若文本客观上可能产生该效果，也应选"是"
- "无害效果"指文本不会对他人造成实质性伤害
- 只回答一个数字（1或2）"""

EFFECT_USER_TEMPLATE = """请判断以下文本是否会产生"{adj_chinese}"（{definition}）：

文本：{content}

只回答一个数字（1=否，2=是）："""


# 模板映射
SYSTEM_PROMPTS = {
    "behavior": BEHAVIOR_SYSTEM,
    "strategy": STRATEGY_SYSTEM,
    "evaluation": EVALUATION_SYSTEM,
    "discrimination": DISCRIMINATION_SYSTEM,
    "neutral": NEUTRAL_SYSTEM,
    "intent": INTENT_SYSTEM,
    "effect": EFFECT_SYSTEM,
}

USER_TEMPLATES = {
    "behavior": BEHAVIOR_USER_TEMPLATE,
    "strategy": STRATEGY_USER_TEMPLATE,
    "evaluation": EVALUATION_USER_TEMPLATE,
    "discrimination": DISCRIMINATION_USER_TEMPLATE,
    "neutral": NEUTRAL_USER_TEMPLATE,
    "intent": INTENT_USER_TEMPLATE,
    "effect": EFFECT_USER_TEMPLATE,
}

# Verbalizer映射：二元用{"1","2"}，3级用{"1","2","3"}
VERBALIZERS = {
    "behavior": ["1", "2"],
    "strategy": ["1", "2"],
    "evaluation": ["1", "2", "3"],
    "discrimination": ["1", "2", "3"],
    "neutral": ["1", "2", "3"],
    "intent": ["1", "2"],
    "effect": ["1", "2"],
}

# 3级类型（保存level_probs）
THREE_LEVEL_TYPES = {"evaluation", "discrimination", "neutral"}
# 二元类型（保存binary_prob）
BINARY_TYPES = {"behavior", "strategy", "intent", "effect"}


def load_concepts(csv_path):
    """加载概念词典，返回概念列表。"""
    concepts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concepts.append({
                "name": row["name"],
                "chinese": row.get("chinese", row["name"]),
                "type": row["type"],
                "category": row.get("category", "neutral"),
                "definition": row["definition"],
                "prompt_template": row.get("prompt_template", row["type"]),
            })
    return concepts


def build_prompts(content, concept):
    """根据概念类型构建对应的system和user prompt。"""
    ptype = concept["prompt_template"]
    system_prompt = SYSTEM_PROMPTS[ptype]
    user_prompt = USER_TEMPLATES[ptype].format(
        adj_chinese=concept["chinese"],
        definition=concept["definition"],
        content=content,
    )
    return system_prompt, user_prompt


def get_verbalizer(concept):
    """根据概念类型返回verbalizer词表。"""
    ptype = concept["prompt_template"]
    return VERBALIZERS[ptype]


def parse_args():
    parser = argparse.ArgumentParser(description="类型感知概念向量生成脚本")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adjective_name", type=str, default="toxic_adjectives_v4.csv")
    parser.add_argument("--data_file", type=str, default=None, help="自定义数据文件名（如train_100.json），默认根据mode自动选择")
    parser.add_argument("--num_samples", type=int, default=0, help="快速验证用，0=全量")
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--logprobs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    # 延迟导入vLLM
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # 路径
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "data" / "raw" / args.dataset_name
    processed_dir = base_path / "data" / "processed" / args.dataset_name / args.model_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    adj_path = base_path / "data" / "raw" / "adjective" / args.adjective_name

    # 加载概念
    concepts = load_concepts(adj_path)
    print(f"概念总数: {len(concepts)}")
    type_counts = {}
    for c in concepts:
        ptype = c["prompt_template"]
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    for ptype, count in sorted(type_counts.items()):
        v_type = "3级" if ptype in THREE_LEVEL_TYPES else "二元"
        print(f"  {ptype}: {count}概念 ({v_type})")

    # 加载数据
    if args.data_file:
        data_file = data_dir / args.data_file
    else:
        if args.mode == "train":
            data_file = data_dir / "train.json"
        else:
            data_file = data_dir / "test.json"

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if args.num_samples > 0:
        raw_data = raw_data[:args.num_samples]
        print(f"快速验证模式: 使用前{args.num_samples}条样本")

    print(f"数据: {len(raw_data)}条 ({args.mode})")

    # 初始化模型
    model_path = base_path / "models" / args.model_name
    print(f"加载模型: {model_path}")

    llm = LLM(model=str(model_path), trust_remote_code=True, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # 判断是否有chat template
    has_chat_template = tokenizer.chat_template is not None
    print(f"Chat template: {'有' if has_chat_template else '无'}")

    # 预构建所有prompt
    print("构建提示词...")
    all_prompts = []
    all_verbalizers = []
    all_ptypes = []
    # 索引映射：prompt_idx -> (sample_idx, concept_idx)
    prompt_map = []

    for idx, item in enumerate(raw_data):
        content = item["content"]
        for ci, concept in enumerate(concepts):
            system_prompt, user_prompt = build_prompts(content, concept)
            verbalizer = get_verbalizer(concept)
            ptype = concept["prompt_template"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if has_chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    prompt_suffix="\n"
                )
            else:
                prompt_text = system_prompt + "\n\n" + user_prompt + "\n"

            all_prompts.append(prompt_text)
            all_verbalizers.append(verbalizer)
            all_ptypes.append(ptype)
            prompt_map.append((idx, ci))

    print(f"总prompt数: {len(all_prompts)}")

    # 批量生成
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        logprobs=args.logprobs,
    )

    batch_size = 256
    all_outputs = []
    for start in range(0, len(all_prompts), batch_size):
        end = min(start + batch_size, len(all_prompts))
        batch = all_prompts[start:end]
        print(f"  生成中: {start}-{end}/{len(all_prompts)}")
        outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(outputs)

    # 解析结果
    print("解析结果...")
    results = [None] * len(raw_data)
    for i in range(len(raw_data)):
        results[i] = {"concept_scores": [0.0] * len(concepts), "level_probs": [None] * len(concepts)}

    for pi, output in enumerate(all_outputs):
        idx, ci = prompt_map[pi]
        verbalizer = all_verbalizers[pi]
        ptype = all_ptypes[pi]

        # 提取logprobs
        token_logprobs = {}
        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0]
            for token_id, logprob_info in first_token_logprobs.items():
                token_text = logprob_info.decoded_token.strip()
                if token_text in verbalizer:
                    token_logprobs[token_text] = logprob_info.logprob

        # 计算概率
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

        # 计算标量分数
        if ptype in BINARY_TYPES:
            score = probs.get("2", 0.0)
            results[idx]["concept_scores"][ci] = score
            results[idx]["level_probs"][ci] = [probs.get("1", 0.0), probs.get("2", 0.0)]
        else:
            score = (probs.get("2", 0.0) * 2 + probs.get("3", 0.0) * 3) / (probs.get("1", 0.0) + probs.get("2", 0.0) + probs.get("3", 0.0) + 1e-8)
            results[idx]["concept_scores"][ci] = score
            results[idx]["level_probs"][ci] = [probs.get("1", 0.0), probs.get("2", 0.0), probs.get("3", 0.0)]

    # 填充content和toxic
    for idx, item in enumerate(raw_data):
        results[idx]["content"] = item["content"]
        results[idx]["toxic"] = item.get("toxic", -1)

    # 保存结果
    adj_suffix = Path(args.adjective_name).stem.replace("toxic_adjectives_", "")
    suffix = f"typed_{adj_suffix}"
    output_path = processed_dir / f"concept_{args.mode}_{args.model_name.replace('/', '-')}_{suffix}.json"

    # 保存概念元信息
    meta = {
        "num_concepts": len(concepts),
        "concept_names": [c["name"] for c in concepts],
        "concept_types": [c["prompt_template"] for c in concepts],
        "concept_chinese": [c.get("chinese", c["name"]) for c in concepts],
        "adjective_file": args.adjective_name,
        "num_samples": len(results),
        "mode": args.mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_data = {
        "meta": meta,
        "data": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n概念向量已保存至: {output_path}")
    print(f"总概念数: {len(concepts)}, 总样本数: {len(results)}")

    # 覆盖率统计
    type_coverage = {ptype: {"total": 0, "covered": 0} for ptype in VERBALIZERS}
    for item in results:
        for i, concept in enumerate(concepts):
            ptype = concept["prompt_template"]
            lp = item["level_probs"][i]
            if ptype in BINARY_TYPES:
                type_coverage[ptype]["total"] += 1
                if lp[0] > 0.01 or lp[1] > 0.01:
                    type_coverage[ptype]["covered"] += 1
            else:
                type_coverage[ptype]["total"] += 1
                if lp[0] > 0.01 or lp[1] > 0.01 or lp[2] > 0.01:
                    type_coverage[ptype]["covered"] += 1

    print("\nVerbalizer覆盖率:")
    for ptype, cov in type_coverage.items():
        if cov["total"] > 0:
            rate = cov["covered"] / cov["total"] * 100
            print(f"  {ptype}: {rate:.2f}% ({cov['covered']}/{cov['total']})")


if __name__ == "__main__":
    main()
