"""Qwen模型提示词模板分析工具

主要功能：
模拟 generate_adjective_c_r.py 的单样本推理流程，快速调试提示词和 Verbalizer，确认分数计算是否符合预期
该脚本用于分析和调试不同提示词模板（binary / likert / ICL）在Qwen系列模型上的推理行为，
帮助评估提示词模板的质量和合理性，为生成形容词概念向量提供调试支持。
后续在论文中可以在末尾简短对提示词进行分析，以证明我们的工作量。

具体分析项：
1. 首token概率分布 Top-10：观察模型对第一个输出token的预测偏好，用于判断提示词
   是否将模型输出约束到预期方向；
2. 模型生成序列（10个token，贪心解码）：观察模型实际输出的文本是否通顺、符合模板要求；
3. Verbalizer概率分析：统计预定义Verbalizer词表占总概率的比例，评估提示词对模型
   输出的约束强度（理想区间约70%-90%）。

支持的模板类型（与generate_adjective_c_r.py保持一致）：
- binary: 二元判断（是/否），用于判断形容词是否准确描述文本；
- likert: 1-5程度量化，用于评估文本具有形容词特征的程度；
- ICL: 基于形容词定义的判断（In-Context Learning），引入形容词定义进行推理。

使用方法：
直接修改下方 CONFIG 区域的变量（模型名、模板类型、文本内容、形容词等），然后运行：
python scripts/inspect_prompt_template.py
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "Qwen2.5-3B-Instruct"  # models目录下的模型文件夹名

# 提示词模板类型，可选: "binary", "likert", "ICL"
PROMPT_TEMPLATE = "ICL"

# 文本内容和形容词（直接修改即可）
TEXT_CONTENT = "中国男性根本就配不上中国女性"
ADJECTIVE = "包容的"

# ICL模板专用：形容词定义（仅ICL模板使用，不需要时可留空）
ADJECTIVE_DEFINITION = "尊重和接纳不同群体或观点的态度，不对他人身份、地域或取向进行排斥或歧视的言论。"
# ===================================================================


def load_qwen_model(model_path: Path, model_name: str):
    """加载模型和分词器（复用generate_adjective_c_r逻辑）"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Qwen model from {llm_path}")
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def get_first_token_ids(word_list, tokenizer, device):
    """获取词表中每个词的首token id"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    token_ids = sorted(set(token_ids))
    return torch.tensor(list(dict.fromkeys(token_ids)), device=device, dtype=torch.long)


def build_prompt_and_verbalizer(template, text_content, adjective, adj_definition=""):
    """
    根据模板类型构建完整的提示词和对应的Verbalizer词表。
    逻辑与 generate_adjective_c_r.py 中的模板构建保持一致。

    返回:
        prompt: 完整的提示词文本
        verbalizer_words: 用于显示的verbalizer词列表
        score_tokens: 用于分数计算的token分组字典
            - binary/ICL: {"affirmative": [...], "negative": [...]}
            - likert: {"likert": ["1", "2", "3", "4", "5"]}
    """
    if template in ["binary", "ICL"]:
        verbalizer_words = ["是", " 是", "Yes", " Yes", "yes", " yes",
                            "否", "不", " 不", "不能", "无", "No", " No", "no", " no"]
        score_tokens = {
            "affirmative": ["是", " 是", "Yes", " Yes", "yes", "yes"],
            "negative": ["否", "不", " 不", "不能", "无", "No", " No", "no", " no"],
        }

        if template == "binary":
            instruction = "你是一位社会心理学专家。请判断给定形容词是否准确描述文本，只回答是或否。"
            prompt = f"{instruction}\n文本内容：{text_content}\n形容词：「{adjective}」描述是否准确？回答： "
        elif template == "ICL":
            instruction = "请根据形容词的定义，判断该文本是否表现出该形容词所描述的特征，只回答是或否。"
            prompt = (f"{instruction}\n"
                      f"文本内容：{text_content}\n"
                      f"形容词「{adjective}」的定义：{adj_definition}\n"
                      f"根据上述定义，该文本是否表现出该「{adjective}」所描述的特征？回答： ")

    elif template == "likert":
        instruction = ("你是一位社会心理学专家。请评估以下文本具有给定形容词所描述特征的程度。\n"
                       "请严格按照1到5的等级进行评估：\n"
                       "1 = 完全不具有该特征\n"
                       "2 = 略微具有该特征\n"
                       "3 = 中等程度具有该特征\n"
                       "4 = 较强程度具有该特征\n"
                       "5 = 非常强烈地具有该特征")
        prompt = f"{instruction}\n文本内容：{text_content}\n形容词「{adjective}」的程度等级（直接回答数字）： "
        verbalizer_words = ["1", "2", "3", "4", "5"]
        score_tokens = {
            "likert": ["1", "2", "3", "4", "5"],
        }

    else:
        raise ValueError(f"不支持的模板类型: {template}，可选: binary, likert, ICL")

    return prompt, verbalizer_words, score_tokens


def main():
    config = MLPConfig()

    tokenizer, model = load_qwen_model(config.models_path, MODEL_NAME)
    device = next(model.parameters()).device

    # 根据模板构建提示词和Verbalizer
    prompt, verbalizer_words, score_tokens = build_prompt_and_verbalizer(
        PROMPT_TEMPLATE, TEXT_CONTENT, ADJECTIVE, ADJECTIVE_DEFINITION
    )

    print("\n" + "=" * 60)
    print("模型推理调试")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"模板类型: {PROMPT_TEMPLATE}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词: {ADJECTIVE}")
    print(f"提示词: {prompt}")

    # 编码提示词
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    print(f"\n提示词token数: {inputs['input_ids'].shape[1]}")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    # 取最后一个token位置的logits
    last_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(last_logits.float(), dim=-1)

    # 输出概率最高的前10个token
    topk = 10
    top_probs, top_indices = torch.topk(probs, topk)
    print(f"\n首token概率分布 Top-{topk}:")
    print(f"{'排名':<4} {'Token ID':<10} {'Token文本':<12} {'概率':<12} {'累计概率':<10}")
    cumsum = 0.0
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_text = tokenizer.decode([idx.item()])
        cumsum += prob.item()
        print(f"{rank:<4} {idx.item():<10} {repr(token_text):<12} {prob.item():<12.6f} {cumsum:<10.6f}")

    # 模型生成的10个词（贪心解码）
    print(f"\n模型生成 Top-10（贪心解码，每次取概率最高的token）:")
    generated_ids = []
    current_input_ids = inputs["input_ids"].clone()
    for step in range(10):
        with torch.no_grad():
            out = model(input_ids=current_input_ids, use_cache=False)
        next_logits = out.logits[0, -1, :]
        next_token_id = torch.argmax(next_logits).item()
        generated_ids.append(next_token_id)
        current_input_ids = torch.cat([current_input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"生成token序列: {generated_ids}")
    print(f"生成文本: {repr(generated_text)}")

    # Verbalizer分析
    if verbalizer_words:
        verbalizer_ids = get_first_token_ids(verbalizer_words, tokenizer, device)

        print(f"\nVerbalizer分析 ({len(verbalizer_words)}个词 -> {len(verbalizer_ids)}个唯一token):")
        print(f"{'词':<10} {'Token ID':<10} {'概率':<12}")
        verbalizer_prob_list = []
        for word in verbalizer_words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if encoded:
                tid = encoded[0]
                p = probs[tid].item()
                verbalizer_prob_list.append((word, tid, p))
                print(f"{word:<10} {tid:<10} {p:<12.6f}")

        # verbalizer概率统计
        total_vprob = sum(p for _, _, p in verbalizer_prob_list)
        print(f"\nVerbalizer概率总和: {total_vprob:.6f}")
        print(f"Verbalizer占总概率比例: {total_vprob:.2%}")

        # 分数计算（与generate_adjective_c_r.py保持一致）
        print(f"\n{'=' * 60}")
        print("概念向量分数计算")
        print(f"{'=' * 60}")

        if PROMPT_TEMPLATE == "binary" or PROMPT_TEMPLATE == "ICL":
            affirmative_ids = get_first_token_ids(score_tokens["affirmative"], tokenizer, device)
            negative_ids = get_first_token_ids(score_tokens["negative"], tokenizer, device)

            pos_prob = probs[affirmative_ids].sum().item()
            neg_prob = probs[negative_ids].sum().item()
            total = pos_prob + neg_prob + 1e-8
            score = pos_prob / total

            print(f"肯定词概率: {pos_prob:.6f}")
            print(f"否定词概率: {neg_prob:.6f}")
            print(f"归一化后概念分数: {score:.6f}")

        elif PROMPT_TEMPLATE == "likert":
            likert_ids = get_first_token_ids(score_tokens["likert"], tokenizer, device)
            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

            level_probs = probs[likert_ids]
            total_level_prob = level_probs.sum() + 1e-8
            score = (weights * level_probs / total_level_prob).sum().item()

            print(f"Level 1 (权重0.00): {level_probs[0].item():.6f}")
            print(f"Level 2 (权重0.25): {level_probs[1].item():.6f}")
            print(f"Level 3 (权重0.50): {level_probs[2].item():.6f}")
            print(f"Level 4 (权重0.75): {level_probs[3].item():.6f}")
            print(f"Level 5 (权重1.00): {level_probs[4].item():.6f}")
            print(f"Likert概率总和: {total_level_prob.item():.6f}")
            print(f"加权期望概念分数: {score:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
