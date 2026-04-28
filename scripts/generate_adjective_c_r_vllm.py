"""生成形容词概念向量（vLLM版本）

基于Chat Template生成形容词概念向量，使用vLLM加速批量推理。

使用示例：
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template binary
"""

import argparse
import math
import sys
from pathlib import Path
import json

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
        epilog="使用示例:"
               "python scripts/generate_adjective_c_r_vllm.py --mode test --dataset_name TOXICN --model_name ... --template binary"
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
        choices=['binary', 'likert', 'ICL'],
        default='binary',
        help='提示词模板类型：binary=二元判断(原始), likert=Likert程度量化, ICL=引入形容词解释'
    )

    return parser.parse_args()


def load_vllm_model(model_path: Path, model_name: str):
    """加载vLLM模型和tokenizer"""
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
        dtype="float16",
        quantization=None,  # 不使用量化
        gpu_memory_utilization=0.7,  # 最多占用GPU显存的比例
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=256,
        max_num_batched_tokens=4096,
    )

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

def build_chat_messages(template, instruction, content, adj, adj_definition=""):
    """
    根据模板类型构建Chat Template（手动拼接方式在语义上等价）
    """
    if template == "binary":
        user_content = f"文本内容：{content}\n形容词：「{adj}」描述是否准确？回答： "
    elif template == "likert":
        user_content = f"文本内容：{content}\n形容词「{adj}」的程度等级（直接回答数字）： "
    elif template == "ICL":
        user_content = (
            f"文本内容：{content}\n"
            f"形容词「{adj}」的定义：{adj_definition}\n"
            f"根据上述定义，该文本是否表现出该形容词「{adj}」所描述的特征？回答： "
        )
    else:
        raise ValueError(f"不支持的模板类型: {template}")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages

def generate_adj_concept(data_path, output_path, adjective_path, template, tokenizer, llm_model):
    # 根据模板类型定义verbalizer token（首token id集合）和提示词指令
    if template in ["binary", "ICL"]:
        # verbalizer tokens
        affirmative_tokens = ["是", " 是", "Yes", " Yes", "yes", " yes"]
        negative_tokens = ["否", "不", " 不", "不能", "无", "No", " No", "no", " no"]
        affirmative_ids = get_first_token_ids(affirmative_tokens, tokenizer)
        negative_ids = get_first_token_ids(negative_tokens, tokenizer)

        if template == "binary":
            instruction = "你是一位社会心理学专家。请判断给定形容词是否准确描述文本，只回答是或否。"
        elif template == "ICL":
            instruction = "请根据形容词的定义，判断该文本是否表现出该形容词所描述的特征，只回答是或否。"
            # 加载形容词解释
            definition = pd.read_csv(adjective_path)["definition"].tolist()

    elif template == "likert":
        likert_tokens = ["1", "2", "3", "4", "5"]
        likert_ids = get_first_token_ids(likert_tokens, tokenizer)

        instruction = ("你是一位社会心理学专家。请评估以下文本具有给定形容词所描述特征的程度。\n"
                       "请严格按照1到5的等级进行评估：\n"
                       "1 = 完全不具有该特征\n"
                       "2 = 略微具有该特征\n"
                       "3 = 中等程度具有该特征\n"
                       "4 = 较强程度具有该特征\n"
                       "5 = 非常强烈地具有该特征")

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

    # 批量推理
    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]  # 文本，后续将匹配V个形容词
        concept_vector = []
        raw_probs = []

        # 在vllm中不手动分批次，构建一条文本+所有形容词的Chat Template
        prompts = []  # 关于当前文本的所有Chat Template
        for index, adj in enumerate(adjectives):
            if template == "ICL":
                messages = build_chat_messages(template, instruction, content, adj, definition[index])
            else:
                messages = build_chat_messages(template, instruction, content, adj)

            # 添加<|im_start|>system,<|im_start|>user,<|im_start|>assistant特殊token
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt_text)

        # 批量推理
        outputs = llm_model.generate(prompts, sampling_params)  # 表示对文本+所有形容词的推理结果

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

            # 根据模板类型计算score
            if template == "binary":
                # 计算肯定词的概率之和
                pos_prob = 0.0
                for tid in affirmative_ids:
                    pos_prob = pos_prob + probs_dict.get(tid, 0.0)
                # 计算否定词的概率之和
                neg_prob = 0.0
                for tid in negative_ids:
                    neg_prob = neg_prob + probs_dict.get(tid, 0.0)
                # 归一化计算score
                total = pos_prob + neg_prob + 1e-8
                score = pos_prob / total
                raw_probs.append([pos_prob, neg_prob])
            elif template == "ICL":
                # 计算肯定词的概率之和
                pos_prob = 0.0
                for tid in affirmative_ids:
                    pos_prob = pos_prob + probs_dict.get(tid, 0.0)
                # 计算否定词的概率之和
                neg_prob = 0.0
                for tid in negative_ids:
                    neg_prob = neg_prob + probs_dict.get(tid, 0.0)
                # 归一化计算score
                total = pos_prob + neg_prob + 1e-8
                score = pos_prob / total
                raw_probs.append([pos_prob, neg_prob])
            elif template == "likert":
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

        # 保存当前文本的推理结果
        results.append({"content": content, "toxic": sample["toxic"], "concept": concept_vector, "raw_probs": raw_probs})

    # 将所有结果保存到JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"形容词概念向量保存到: {output_path}")


def main():
    args = parse_args()  # 解析命令行参数

    config = MLPConfig()  # 加载模型配置

    # 动态生成依赖 dataset_name/model_name 的路径
    data_path = config.raw_data_path / args.dataset_name / f"mini_{args.mode}.json"  # 原始数据集目录
    concept_dir = config.processed_path / args.dataset_name / args.model_name / args.template  # 概念向量输出目录
    concept_dir.mkdir(parents=True, exist_ok=True)
    output_path = concept_dir / f"concept_{args.mode}.json"
    # 打印配置信息
    print("\n" + "=" * 60)
    print("形容词概念向量生成(vLLM) - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"提示词模板: {args.template}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    tokenizer, llm_model = load_vllm_model(config.models_path, args.model_name)
    generate_adj_concept(data_path, output_path, config.adjective_path, args.template, tokenizer, llm_model)

    print("生成完成")

if __name__ == '__main__':
    main()
