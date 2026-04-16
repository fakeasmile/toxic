"""生成形容词概念向量（基于肯定词概率和）。"""
import argparse
import sys
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成形容词概念向量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="使用示例:"
               "python scripts/generate_adjective_c_r.py --mode {train/test} --dataset_name TOXICN --model_name ..."
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
        default=None,
        help='数据集名称(TOXICN/COLD)'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='LLM模型名称'
    )

    return parser.parse_args()

def dynamic_load_parameters(args):
    """用命令行参数覆盖MLP_config参数
    
    注意：必须先更新 dataset_name 和 model_name，再更新依赖它们的路径
    """
    mlp_config = MLPConfig()
    
    # 第一步：更新基础配置
    if args.dataset_name is not None:
        mlp_config.dataset_name = args.dataset_name
    if args.model_name is not None:
        mlp_config.model_name = args.model_name

    # 第二步：重新计算所有依赖 dataset_name 和 model_name 的路径
    # 确保路径与最新的数据集/模型名称一致
    mlp_config.train_path = mlp_config.base_path / "data" / "raw" / mlp_config.dataset_name / "train.json"
    mlp_config.test_path = mlp_config.base_path / "data" / "raw" / mlp_config.dataset_name / "test.json"
    mlp_config.train_concept_path = mlp_config.processed_path / f"train_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json"
    mlp_config.test_concept_path = mlp_config.processed_path / f"test_with_concepts({mlp_config.dataset_name})({mlp_config.model_name}).json"
    
    # 打印实际使用的配置，便于调试和验证
    print("\n" + "=" * 60)
    print("形容词概念向量生成 - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {mlp_config.dataset_name}")
    print(f"LLM模型名称: {mlp_config.model_name}")
    print(f"训练集路径: {mlp_config.train_path}")
    print(f"测试集路径: {mlp_config.test_path}")
    print(f"形容词词典路径: {mlp_config.adjective_path}")
    print("=" * 60 + "\n")

    return mlp_config

def load_qwen_model(model_path: Path, model_name: str):
    """加载模型和分词器"""
    llm_path = model_path / model_name  # LLM模型路径
    # 判断llm目录是否存在
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit量化
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  #
        bnb_4bit_compute_dtype=torch.float16,  # 半精度
    )

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",  # PAD右填充
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
    """获取词表中每个词的首 token id（去重后返回）。"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR")

    return torch.tensor(sorted(set(token_ids)), device=device, dtype=torch.long)


def _expand_prefix_cache(base_cache, batch_size: int):
    """
    将单样本 prefix cache:[1, num_heads, L_prefix, head_dim] 扩展到 batch 维度。
    同时兼容 legacy tuple cache 与 DynamicCache。
    """
    if isinstance(base_cache, DynamicCache):
        legacy_cache = base_cache.to_legacy_cache()
        expanded = DynamicCache()
        for layer_idx, kv in enumerate(legacy_cache):
            k, v = kv[0], kv[1]
            k_expanded = k.expand(batch_size, -1, -1, -1).contiguous()  # [batch_size, num_heads, L_prefix, head_dim]
            v_expanded = v.expand(batch_size, -1, -1, -1).contiguous()
            expanded.update(k_expanded, v_expanded, layer_idx)
        return expanded

    expanded_layers = []
    for kv in base_cache:
        k, v = kv[0], kv[1]
        k_expanded = k.expand(batch_size, -1, -1, -1).contiguous()
        v_expanded = v.expand(batch_size, -1, -1, -1).contiguous()
        expanded_layers.append((k_expanded, v_expanded))
    return tuple(expanded_layers)


def generate_adj_concept(mlp_config, mode):
    # 加载LLM和分词器
    tokenizer, model = load_qwen_model(mlp_config.models_path, mlp_config.model_name)
    device = next(model.parameters()).device

    # 定义肯定词/否定词列表
    affirmative_tokens = ["是", "是的", "对", "准确", "正确", "Yes", "yes"]
    negative_tokens = ["否", "不", "错误", "偏差", "No", "no"]
    # 获取肯定词/否定词token id
    affirmative_ids = get_first_token_ids(affirmative_tokens, tokenizer, device)
    negative_ids = get_first_token_ids(negative_tokens, tokenizer, device)

    # 读取形容词列表，提取中文列
    adjectives = pd.read_csv(mlp_config.adjective_path)["chinese"].tolist()

    # 读取数据集
    if mode == "train":
        with open(mlp_config.train_path, "r", encoding="utf-8") as f:
            data_set = json.load(f)
        output_concept_path = mlp_config.train_concept_path  # 形容词概念向量保存路径
    elif mode == "test":
        with open(mlp_config.test_path, "r", encoding="utf-8") as f:
            data_set = json.load(f)
        output_concept_path = mlp_config.test_concept_path
    else:
        raise ValueError("dataset_name must be 'train' or 'test'")

    print(output_concept_path)
    return
    results = []
    batch_size = 16  # 形容词批量推理大小

    # “公共前缀缓存 + 变动后缀批量推理”
    instruction = "你是一个语义分析专家。请判断给定形容词是否准确描述文本，只回答‘是’或‘否’。"

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples"), start=1):
        content = sample["content"]  # 文本内容
        concept_vector = []  # 存储这条文本对应的形容词概念向量，形状：[V]

        # 指令+文本内容作为公共前缀
        prefix_text = f"{instruction}\n文本内容：{content}\n"
        # tokenizer分词，input_ids形状[1, L_prefix]，attention_mask：[1, L_prefix]，全1
        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).to(device)

        # 计算前缀的KV缓存
        with torch.no_grad():
            prefix_outputs = model(**prefix_inputs, use_cache=True)
            base_cache = prefix_outputs.past_key_values  # k,v形状：[1, num_heads, L_prefix, head_dim]
            prefix_len = prefix_inputs["input_ids"].shape[1]  # 获取前缀token的长度

        for i in range(0, len(adjectives), batch_size):
            adj_batch = adjectives[i: i + batch_size]  # 批量大小的形容词（最后一组可能不足）
            curr_bsz = len(adj_batch)  # 当前形容词数量

            # 后缀token
            suffix_texts = []
            for adj in adj_batch:
                suffix_texts.append(f"形容词：‘{adj}’描述是否准确？回答：")
            suffix_inputs = tokenizer(
                suffix_texts,
                return_tensors="pt",
                padding=True,  # 填充到批次内最大长度
                add_special_tokens=False,
            ).to(device)
            # suffix_inputs结构：
            # - input_ids: 形状[curr_bsz, L_suffix]（L_suffix=后缀最长Token数）
            # - attention_mask: 形状[curr_bsz, L_suffix]（1=有效Token，0=填充）

            suffix_len = suffix_inputs["input_ids"].shape[1]  # 后缀长度L_suffix

            # 构建position_ids
            # 生成递增序列[L_prefix, L_prefix+1,...,L_prefix+L_suffix-1]，形状：[L_suffix]
            position_ids = torch.arange(prefix_len, prefix_len + suffix_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(curr_bsz, -1).contiguous()  # 扩展为[curr_bsz, L_suffix]

            # 扩展前缀缓存到当前batch大小
            expanded_cache = _expand_prefix_cache(base_cache, curr_bsz)

            # 构建完整的attention_mask，前缀掩码全1，形状[curr_bsz, L_prefix]
            prefix_mask = torch.ones(
                (curr_bsz, prefix_len),
                device=device,
                dtype=suffix_inputs["attention_mask"].dtype,
            )
            # 形状：[curr_bsz, L_prefix + L_suffix]
            full_mask = torch.cat([prefix_mask, suffix_inputs["attention_mask"]], dim=1)

            # 此时构建好前缀kv缓存，完整的注意力掩码和位置编码
            '''
            attention_mask:
            --prefix--    --suffix--
            1,1,1,1,1,1,,..,1,1,1,0
            1,1,1,1,1,1,...,1,0,0,0
            1,1,1,1,1,1,...,1,1,1,1
            position_ids:
            --suffix--
            L_p, L_p+1, L_p+2, ..., L_p+i, L_p+i
            L_p, L_p+1, L_p+2, ..., L_p+j, L_p+j
            L_p, L_p+1, L_p+2, ..., L_p+k, L_p+k
            ...
            '''
            with torch.no_grad():
                outputs = model(
                    input_ids=suffix_inputs["input_ids"],  # input_ids:[curr_bsz, L_suffix]
                    attention_mask=full_mask,              # [curr_bsz, L_prefix+L_suffix]
                    position_ids=position_ids,             # [curr_bsz, L_suffix]
                    past_key_values=expanded_cache,        # 前缀缓存
                    use_cache=False,
                )

            logits = outputs.logits  # 形状：[curr_bsz, L_suffix, vocab_size]
            # 根据最后一个token位置获取生成的第一个token的概率分布
            last_token_indices = suffix_inputs["attention_mask"].sum(dim=1) - 1  # 分词器为right右填充，形状[curr_bsz]

            for j, last_idx in enumerate(last_token_indices):
                # last_idx表示这个批次中第j个样本的有效token长度-1
                target_logits = logits[j, last_idx, :]  # 形状[V]，表示最后一个有效token的为归一化分数

                # softmax 前转 float32，避免半精度下极小概率数值不稳定
                probs = torch.softmax(target_logits.float(), dim=-1)  # 和为1

                pos_prob = probs[affirmative_ids].sum()  # 计算肯定词的概率和
                neg_prob = probs[negative_ids].sum()  # 计算否定词的概率和
                total = pos_prob + neg_prob + 1e-8
                score = (pos_prob / total).item()

                concept_vector.append(score)

        # 防御性校验，确保每条文本输出的形容词概念长度与形容词数量一致
        if len(concept_vector) != len(adjectives):
            raise RuntimeError(
                f"concept_vector 长度异常：期望 {len(adjectives)}，实际 {len(concept_vector)}"
            )

        results.append({"content": content, "concept": concept_vector})

        # 间隔清理缓存
        if torch.cuda.is_available() and sample_idx % 128 == 0:
            torch.cuda.empty_cache()

    with open(output_concept_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"形容词概念向量保存到: {output_concept_path}")


def main():
    args = parse_args()  # 解析命令行参数

    mlp_config = dynamic_load_parameters(args)
    generate_adj_concept(mlp_config, args.mode)


if __name__ == '__main__':
    main()
