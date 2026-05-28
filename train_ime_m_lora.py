#!/usr/bin/env python3
"""
IME+M九维九法分类 LoRA训练脚本
基于Qwen2.5-1.5B，用Qwen-Max蒸馏数据进行LoRA微调
输出格式: Innovation + Method + 维度 + 法则 + Effect
"""
import os
import json
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
MODEL_DIR = '/home/jiyong/桌面/ime论文/experiments/model_cache/Qwen/Qwen2.5-1.5B'
DATA_DIR = '/tmp/ime-patent-extractor/lora_data'
OUTPUT_DIR = '/tmp/ime-patent-extractor/lora_output'
MAX_SEQ_LEN = 768
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def load_data():
    """加载训练数据"""
    train_path = os.path.join(DATA_DIR, 'train.json')
    val_path = os.path.join(DATA_DIR, 'val.json')

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    # 统计维度和法则分布
    weidu_count = {}
    faze_count = {}
    for item in train_data:
        output = item['output']
        for line in output.split('\n'):
            if line.startswith('维度:'):
                w = line.split(':', 1)[1].strip()
                weidu_count[w] = weidu_count.get(w, 0) + 1
            elif line.startswith('法则:'):
                f = line.split(':', 1)[1].strip()
                faze_count[f] = faze_count.get(f, 0) + 1

    print("\n维度分布:")
    for k, v in sorted(weidu_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print("\n法则分布:")
    for k, v in sorted(faze_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return train_data, val_data


def format_sample(tokenizer, instruction, input_text, output_text, max_seq_len=MAX_SEQ_LEN):
    """格式化单条训练样本"""
    # Qwen2.5 chat template
    messages = [
        {"role": "system", "content": "你是一位专利技术分析专家，精通元易创新方法的九维九法分类体系。"},
        {"role": "user", "content": f"{instruction}\n\n专利摘要：\n{input_text}"},
        {"role": "assistant", "content": output_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Tokenize
    enc = tokenizer(text, truncation=True, max_length=max_seq_len, padding=False)
    input_ids = enc['input_ids']

    # 构建labels：只对assistant回复部分计算loss
    # 找到assistant回复的起始位置
    assistant_text = output_text
    assistant_enc = tokenizer(assistant_text, add_special_tokens=False)
    assistant_len = len(assistant_enc['input_ids'])

    labels = [-100] * (len(input_ids) - assistant_len) + input_ids[-assistant_len:]

    # 对齐长度
    if len(labels) < len(input_ids):
        labels = [-100] * len(input_ids)
    labels = labels[:len(input_ids)]

    return {
        'input_ids': input_ids,
        'attention_mask': enc['attention_mask'],
        'labels': labels,
    }


def train():
    """LoRA训练主流程"""
    print("=" * 60)
    print("IME+M九维九法 LoRA训练")
    print("=" * 60)

    # Step 1: 加载数据
    train_data, val_data = load_data()

    if len(train_data) < 10:
        print("训练数据太少，无法训练")
        return

    # Step 2: 加载模型和tokenizer
    print("\n[加载模型]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    # Step 3: LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 4: 格式化数据
    print("\n[格式化数据]")
    train_samples = []
    for item in train_data:
        try:
            sample = format_sample(tokenizer, item['instruction'], item['input'], item['output'])
            if len(sample['input_ids']) > 50:  # 过滤太短的
                train_samples.append(sample)
        except Exception as e:
            pass

    val_samples = []
    for item in val_data:
        try:
            sample = format_sample(tokenizer, item['instruction'], item['input'], item['output'])
            if len(sample['input_ids']) > 50:
                val_samples.append(sample)
        except Exception as e:
            pass

    print(f"有效训练样本: {len(train_samples)}")
    print(f"有效验证样本: {len(val_samples)}")

    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    # Step 5: 训练
    print("\n[开始训练]")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # Step 6: 保存
    final_dir = os.path.join(OUTPUT_DIR, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n{'=' * 60}")
    print(f"训练完成！模型保存: {final_dir}")
    print(f"{'=' * 60}")

    return model, tokenizer


if __name__ == '__main__':
    train()
