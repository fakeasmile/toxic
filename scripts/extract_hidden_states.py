"""提取LLM中间层Hidden State（transformers版本）

【功能】
使用transformers加载LLM，对每条文本构建与Likert评分相同的提示词模板，
前向传播提取指定层的hidden state，保存为.pt文件供ICB-CBM训练使用。

【与vLLM概念向量生成的关系】
- vLLM生成Likert标量概念向量（首token概率 → 加权期望）
- 本脚本生成LLM hidden state（中间层激活 → 稠密向量）
- 两者使用相同的提示词模板和模型权重，确保信息一致性
- 两者独立运行，互不影响

【执行流程】
1. 加载transformers模型和tokenizer
2. 对数据集中每条文本，构建Chat Template prompt
3. 批量前向传播，提取指定层的最后一个token的hidden state
4. 保存为.pt文件

使用示例：
# Qwen2.5-7B-Instruct-GPTQ-Int8
python scripts/extract_hidden_states.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-GPTQ-Int8 --layer 16

# Qwen3.5-9B
python scripts/extract_hidden_states.py --mode train --dataset_name TOXICN --model_name Qwen3.5-9B --layer 20
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

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.MLP_config import MLPConfig

# =============================================================================
# 模型加载配置表（与generate_adjective_c_r_vllm.py保持一致）
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct-GPTQ-Int8": {
        "quantization": "gptq",
        "is_qwen3": False,
        "is_multimodal": False,
    },
    "Qwen3.5-9B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": True,
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
    },
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="提取LLM中间层Hidden State",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:提取训练集hidden state，test:提取测试集hidden state')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称(TOXICN/COLD)')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称')
    parser.add_argument('--layer', type=int, default=16, help='提取hidden state的层号（从0开始）')
    parser.add_argument('--batch_size', type=int, default=8, help='批量前向传播的批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='输入最大token长度')

    return parser.parse_args()


def load_transformers_model(model_path: Path, model_name: str):
    """使用transformers加载LLM模型和tokenizer

    与vLLM使用相同的模型权重，但通过transformers API加载以支持hidden state提取。
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = MODEL_LOADING_CONFIG.get(model_name)
    if model_config is None:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )

    quantization = model_config["quantization"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="left",  # 批量推理时左侧padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading transformers model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if quantization == "gptq":
        # 预量化权重，让transformers自动检测量化配置，不手动指定bits
        from transformers import GPTQConfig
        model_kwargs["quantization_config"] = GPTQConfig()

    model = AutoModelForCausalLM.from_pretrained(llm_path, **model_kwargs)
    model.eval()

    # 获取模型层数和hidden dim
    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  模型层数: {num_layers}, Hidden dim: {hidden_dim}")

    return tokenizer, model, model_config["is_qwen3"], num_layers, hidden_dim


def build_chat_messages(content):
    """构建与Likert评分相同的提示词模板（不含形容词，仅用于提取文本表示）

    注意：这里使用一个通用的系统指令，不包含具体形容词。
    目的是提取文本在LLM中的语义表示，而非针对某个形容词的判断。
    """
    instruction = ("你是一位语言分析专家。请分析以下文本的语义特征。\n"
                   "直接回答。")

    user_content = f"文本内容：{content}\n回答： "

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    return messages


@torch.no_grad()
def extract_hidden_states(data_path, output_path, tokenizer, model, is_qwen3,
                         layer, batch_size, max_length):
    """提取数据集中每条文本的LLM中间层hidden state

    Args:
        data_path: 原始数据集路径
        output_path: hidden state保存路径（.pt文件）
        tokenizer: 分词器
        model: LLM模型
        is_qwen3: 是否为Qwen3+模型
        layer: 提取hidden state的层号
        batch_size: 批量大小
        max_length: 最大token长度
    """
    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    all_hidden_states = []
    contents = [sample["content"] for sample in data_set]

    # 批量处理
    for start_idx in tqdm(range(0, len(contents), batch_size), desc="Extracting hidden states"):
        batch_contents = contents[start_idx:start_idx + batch_size]

        # 构建Chat Template prompts
        prompts = []
        for content in batch_contents:
            messages = build_chat_messages(content)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs
            )
            prompts.append(prompt_text)

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        # 前向传播，获取所有层的hidden states
        outputs = model(**inputs, output_hidden_states=True)

        # 提取指定层的hidden state
        # hidden_states: tuple of (num_layers+1,) 个元素，每个 [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]

        # 取最后一个非padding token的hidden state
        # 通过attention_mask确定每个样本的实际长度
        attention_mask = inputs["attention_mask"]  # [batch, seq_len]
        # 找到每个样本最后一个非padding token的位置
        # 由于padding_side="left"，实际内容在右侧，最后一个1的位置就是最后一个token
        sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch]

        # 提取每个样本最后一个token的hidden state
        batch_size_actual = hidden_states.shape[0]
        for i in range(batch_size_actual):
            last_token_idx = sequence_lengths[i].item()
            h = hidden_states[i, last_token_idx, :]  # [hidden_dim]
            all_hidden_states.append(h.cpu())

    # 保存为.pt文件（转为float32以避免后续训练dtype问题）
    hidden_tensor = torch.stack(all_hidden_states).float()  # [N, hidden_dim], float32
    torch.save(hidden_tensor, output_path)
    print(f"Hidden states保存到: {output_path}")
    print(f"  形状: {hidden_tensor.shape}, dtype: {hidden_tensor.dtype}")


def main():
    args = parse_args()

    config = MLPConfig()

    # 动态生成路径
    data_path = config.raw_data_path / args.dataset_name / f"{args.mode}.json"
    hidden_dir = config.processed_path / args.dataset_name / args.model_name
    hidden_dir.mkdir(parents=True, exist_ok=True)
    output_path = hidden_dir / f"hidden_{args.mode}.pt"

    # 加载模型
    tokenizer, model, is_qwen3, num_layers, hidden_dim = load_transformers_model(
        config.models_path, args.model_name
    )

    # 校验层号
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"层号 {args.layer} 超出范围 [0, {num_layers - 1}]")

    # 打印配置信息
    print("\n" + "=" * 60)
    print("LLM Hidden State提取 - 配置信息")
    print("=" * 60)
    print(f"数据集名称: {args.dataset_name}")
    print(f"LLM模型名称: {args.model_name}")
    print(f"当前模式: {args.mode}")
    print(f"提取层号: {args.layer} / {num_layers - 1}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"批量大小: {args.batch_size}")
    print(f"数据集路径: {data_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")

    # 提取hidden states
    extract_hidden_states(
        data_path, output_path, tokenizer, model, is_qwen3,
        args.layer, args.batch_size, args.max_length
    )

    print("提取完成")


if __name__ == '__main__':
    main()
