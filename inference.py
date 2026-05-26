#!/usr/bin/env python3
"""IME Patent Extractor - 从专利摘要中抽取IME三元组

用法:
    python inference.py --text "专利摘要文本"
    python inference.py --file patents.txt
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
LORA_PATH = "model"  # 本地adapter路径

INSTRUCTION = (
    "从以下专利摘要中抽取IME三元组。要求：只提取原文中明确写出的内容，"
    "不要推断、扩展或添加原文没有的信息。每个字段用一句话简短概括。\n"
    "Innovation(创新主体): xxx | Method(技术手段): xxx | Effect(技术效果): xxx"
)


def load_model(base_model=LORA_PATH if False else BASE_MODEL, lora_path=LORA_PATH):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def extract(text, model, tokenizer):
    prompt = (
        f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{text[:500]}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=55,
            do_sample=False,
            repetition_penalty=1.15,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return response


def main():
    parser = argparse.ArgumentParser(description="IME Patent Extractor")
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
        print(f"输出: {result}")
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, text in enumerate(lines):
            result = extract(text, model, tokenizer)
            print(f"\n[{i+1}] {text[:60]}...")
            print(f"    -> {result}")


if __name__ == "__main__":
    main()
