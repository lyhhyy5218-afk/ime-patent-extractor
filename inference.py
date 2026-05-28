#!/usr/bin/env python3
"""IME Patent Extractor with 元易九维九法分类

从专利摘要中抽取 IME 三元组并自动对 Method 进行元易九维九法分类。

用法:
    python inference.py --text "专利摘要文本"
    python inference.py --file patents.txt
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
LORA_PATH = "model"

INSTRUCTION = (
    "从以下专利摘要中提取IME三元组及Method的元易九维九法分类。\n"
    "要求：只提取原文中明确写出的内容，不要推断。Effect必须包含具体技术特征和作用。\n"
    "输出格式：\n"
    "Innovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx"
)


def load_model(base_model=BASE_MODEL, lora_path=LORA_PATH):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def extract(text, model, tokenizer):
    messages = [
        {"role": "system", "content": "你是一位专利技术分析专家，精通元易创新方法的九维九法分类体系。"},
        {"role": "user", "content": f"{INSTRUCTION}\n\n专利摘要：\n{text[:500]}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="IME Patent Extractor with 元易九维九法")
    parser.add_argument("--text", type=str, help="专利摘要文本")
    parser.add_argument("--file", type=str, help="包含专利摘要的文件（每行一条）")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="基座模型路径")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("请提供 --text 或 --file 参数")
        return

    print("加载模型...")
    model, tokenizer = load_model(args.base_model, LORA_PATH)

    if args.text:
        result = extract(args.text, model, tokenizer)
        print(f"\n输入: {args.text[:100]}...")
        print(f"\n输出:\n{result}")
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, text in enumerate(lines):
            result = extract(text, model, tokenizer)
            print(f"\n[{i+1}] {text[:60]}...")
            print(f"    -> {result}")


if __name__ == "__main__":
    main()
