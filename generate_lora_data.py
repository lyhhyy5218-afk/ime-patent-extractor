#!/usr/bin/env python3
"""
批量生成 LoRA 训练数据
Step 1: 用 Qwen-Max API 对专利摘要抽取 IME + M九维九法分类
Step 2: 将结果转换为 LoRA 训练格式（与 Qwen2.5-1.5B 兼容）
"""
import os
import sys
import json
import re
import time
import random
import argparse
import pandas as pd
import dashscope
from dashscope import Generation, TextEmbedding

sys.path.insert(0, os.path.dirname(__file__))
from ime_m_classify_prompt import (
    IME_SYSTEM_PROMPT, M_CLASSIFY_SYSTEM_PROMPT,
    IME_ICL_EXAMPLES, M_CLASSIFY_ICL_EXAMPLES,
    build_m_classify_user_prompt,
)

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
DASHSCOPE_API_KEY = 'sk-4af4ab13c47d4408aabaa93c32273a48'
MODEL = 'qwen-max'
CHROMA_DIR = '/tmp/ime-patent-extractor/yuanyi_rag_chroma'
COLLECTION_NAME = 'yuanyi_knowledge'

dashscope.api_key = DASHSCOPE_API_KEY

# RAG相关
import chromadb

def get_rag_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def query_rag_context(collection, query, top_k=3):
    """查询RAG获取相关元易知识"""
    resp = TextEmbedding.call(
        model='text-embedding-v3',
        input=[query],
        dimension=1024,
    )
    if resp.status_code != 200:
        return ""
    query_emb = resp.output['embeddings'][0]['embedding']
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    contexts = []
    for doc in results['documents'][0]:
        contexts.append(doc)
    return "\n\n".join(contexts)


# ═══════════════════════════════════════════
# 多样化instruction模板
# ═══════════════════════════════════════════
IME_INSTRUCTIONS = [
    '从以下专利摘要中提取IME三元组及Method的元易九维九法分类。要求：只提取原文中明确写出的内容，不要推断。Effect必须包含具体技术特征和作用。Method需要分类到元易九维九法体系。\n输出格式：\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '阅读专利文本，提取技术创新信息并分类。重要：只提取原文中已有的内容。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '请从专利摘要中识别：Innovation(被创新的技术主体)、Method(具体创新方法)、Method对应的元易维度和法则、Effect(原文明确的技术效果)。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '提取专利技术创新信息，并对Method进行元易九维九法分类。严格要求：只从原文中提取，不要添加。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '请提取该专利的IME三元组和M的元易分类。所有信息必须来自原文。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '从专利摘要中找出：创新对象、实现方法（含九维九法分类）、以及原文中写明的效果。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '对以下专利进行技术信息抽取和元易分类。所有字段内容必须来自原文，不能自行补充。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '阅读专利摘要并回答：创新主体、技术手段（含维度法则分类）、技术效果分别是什么？\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '从专利文本中提取：被创新的技术对象、实现方式（分类到元易维度和法则）、原文所述的效果。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',

    '分析专利的技术创新要素，按IME三元组提取并标注Method的元易维度和法则。禁止编造信息。\nInnovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx',
]

# ICL示例（合并IME+M分类）
COMBINED_ICL_EXAMPLES = [
    {
        "input": "本发明公开了一种高效散热的永磁同步电机转子结构。所述转子铁芯沿轴向开设有多条通风槽，通风槽内壁设有散热翅片，转子铁芯外圆周面涂覆有高导热绝缘层。通过通风槽和散热翅片的协同作用，显著提高了转子的散热效率，有效降低了绕组温升，使电机能够在更高功率密度下长期稳定运行。",
        "output": "Innovation: 转子铁芯散热结构\nMethod: 在转子铁芯沿轴向开设多条通风槽并设置散热翅片，外圆周面涂覆高导热绝缘层\n维度: 结构维\n法则: 分解与去除\nEffect: 提高转子散热效率降低绕组温升，使电机能在更高功率密度下稳定运行"
    },
    {
        "input": "本发明涉及一种低齿槽转矩的永磁电机转子。在转子铁芯的每个磁极之间设置辅助槽，辅助槽的形状为V形，深度为转子铁芯外径的5%-8%。通过优化辅助槽的位置和尺寸参数，有效降低了齿槽转矩，减小了转矩波动，使电机运行更加平稳。",
        "output": "Innovation: 永磁电机转子铁芯\nMethod: 在磁极之间设置V形辅助槽，优化槽的位置和尺寸参数\n维度: 结构维\n法则: 局部优化\nEffect: 降低齿槽转矩减小转矩波动，使电机运行更平稳"
    },
    {
        "input": "本发明提供一种高安全性的锂离子电池隔膜。所述隔膜包括聚乙烯基层和涂覆在基层两侧的陶瓷涂层，陶瓷涂层由氧化铝纳米颗粒和聚偏氟乙烯粘结剂组成。陶瓷涂层提高了隔膜的耐热性能，在高温下能够维持结构稳定不收缩，有效防止了电池内部短路。",
        "output": "Innovation: 锂离子电池隔膜\nMethod: 在聚乙烯基层两侧涂覆由氧化铝纳米颗粒和聚偏氟乙烯粘结剂组成的陶瓷涂层\n维度: 材料维\n法则: 组合与集成\nEffect: 提高隔膜耐热性防止高温收缩导致内部短路"
    },
    {
        "input": "本发明公开了一种采用碳纤维绑环的永磁电机转子。转子永磁体外部设置碳纤维绑环，绑环采用多层缠绕工艺，预紧力逐层递增。碳纤维绑环替代了传统的不锈钢护套，在保证转子结构强度的同时，显著降低了转子重量和转动惯量。",
        "output": "Innovation: 永磁电机转子固定结构\nMethod: 用碳纤维绑环替代不锈钢护套，采用多层缠绕工艺预紧力逐层递增\n维度: 材料维\n法则: 替代\nEffect: 降低转子重量和转动惯量，提高动态响应性能"
    },
    {
        "input": "本发明涉及一种协作机器人的碰撞检测方法。通过在机器人各关节安装六维力传感器实时检测外部力，结合改进的动量观测器算法进行碰撞检测。当检测到异常力信号时，控制系统在5ms内触发安全停机。",
        "output": "Innovation: 协作机器人碰撞检测系统\nMethod: 在各关节安装六维力传感器结合改进的动量观测器算法进行碰撞检测\n维度: 人机关系维\n法则: 智慧化\nEffect: 提高人机协作安全性，快速响应异常碰撞"
    },
]


def call_llm(messages, temperature=0.1, max_tokens=512):
    """调用dashscope Qwen API"""
    for attempt in range(3):
        try:
            resp = Generation.call(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format='message',
            )
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            elif resp.status_code == 429:
                time.sleep(5)
                continue
            else:
                print(f"  [API错误] {resp.status_code}: {resp.message}")
                return ""
        except Exception as e:
            print(f"  [异常] {e}")
            time.sleep(3)
    return ""


def extract_ime_and_classify(patent_text, rag_context="", instruction_idx=None):
    """一步完成IME抽取+M分类（合并调用，节省token）"""
    if instruction_idx is None:
        instruction_idx = random.randint(0, len(IME_INSTRUCTIONS) - 1)
    instruction = IME_INSTRUCTIONS[instruction_idx]

    # 构建system prompt
    system_parts = ["""你是一位专利技术分析专家，精通元易创新方法的九维九法分类体系。

【IME字段定义】
Innovation: 被改造的技术部件或系统（要具体，不能写"一种电机"）
Method: 具体做了什么改动（不能只写"优化"）
Effect: 具体技术效果+作用（不能只写"提高效率"，要写明改了什么指标）

【元易九维】空间维、环境维、结构维、功能维、机理维、材料维、动力体系维、时序维、人机关系维
【元易九法】分解与去除、组合与集成、局部优化、替代、动态化、自服务、友好化、柔性化、智慧化

【严格规则】
1. 所有IME内容必须来自原文，禁止编造
2. Effect必须包含具体技术特征
3. 维度判断：Method改变了系统的什么方面？
4. 法则判断：Method的核心动作是什么？（拆解？组合？优化？替代？动态？智能？）
5. 如果Method包含多个子动作，取最主要的维度和法则"""]

    if rag_context:
        system_parts.append(f"\n【参考知识】\n{rag_context[:1500]}")

    system_prompt = "\n".join(system_parts)

    messages = [{"role": "system", "content": system_prompt}]

    # ICL示例（随机选2-3个）
    n_icl = random.randint(2, min(3, len(COMBINED_ICL_EXAMPLES)))
    icl_samples = random.sample(COMBINED_ICL_EXAMPLES, n_icl)
    for ex in icl_samples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # 实际输入
    messages.append({"role": "user", "content": f"{instruction}\n\n专利摘要：\n{patent_text}"})

    result = call_llm(messages, temperature=0.1, max_tokens=400)
    return result


def parse_combined_output(raw_text):
    """解析合并输出结果"""
    result = {
        "Innovation": "",
        "Method": "",
        "维度": "",
        "法则": "",
        "Effect": "",
        "raw": raw_text,
    }
    for field in ["Innovation", "Method", "维度", "法则", "Effect"]:
        patterns = [
            rf'{field}\s*[:：]\s*(.+?)(?:\n|$)',
        ]
        m = re.search(patterns[0], raw_text, re.IGNORECASE)
        if m:
            result[field] = m.group(1).strip()
    return result


def generate_training_data(patents_csv, output_dir, max_patents=500, batch_check_point=50):
    """批量生成训练数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载专利数据
    df = pd.read_csv(patents_csv)
    if max_patents > 0 and max_patents < len(df):
        df = df.sample(n=max_patents, random_state=42)
    print(f"加载 {len(df)} 条专利")

    # 加载RAG
    try:
        rag_collection = get_rag_collection()
        print(f"RAG知识库已加载: {rag_collection.count()} 条")
    except Exception as e:
        print(f"RAG加载失败: {e}，不使用RAG")
        rag_collection = None

    results = []
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')

    # 恢复checkpoint
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
            start_idx = ckpt.get('next_idx', 0)
            results = ckpt.get('results', [])
        print(f"从checkpoint恢复: 已处理 {start_idx} 条")

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        abstract = str(row.get('abstract', row.get('摘要', ''))).strip()
        if not abstract or len(abstract) < 30:
            continue

        print(f"\n[{idx + 1}/{len(df)}] ", end="")

        # RAG检索
        rag_context = ""
        if rag_collection:
            # 用专利摘要前100字做检索query
            rag_context = query_rag_context(rag_collection, abstract[:100], top_k=2)

        # 抽取IME+M分类
        instruction_idx = idx % len(IME_INSTRUCTIONS)
        raw_output = extract_ime_and_classify(abstract, rag_context, instruction_idx)
        parsed = parse_combined_output(raw_output)

        # 检查质量
        has_all_fields = all([parsed['Innovation'], parsed['Method'], parsed['Effect']])
        has_classification = parsed['维度'] and parsed['法则']

        if has_all_fields and has_classification:
            results.append({
                "patent_idx": idx,
                "abstract": abstract,
                "domain": str(row.get('domain', row.get('领域', 'unknown'))),
                "Innovation": parsed['Innovation'],
                "Method": parsed['Method'],
                "维度": parsed['维度'],
                "法则": parsed['法则'],
                "Effect": parsed['Effect'],
                "raw_output": raw_output,
            })
            print(f"✓ I={parsed['Innovation'][:15]}... M={parsed['维度']}+{parsed['法则']} E={parsed['Effect'][:15]}...")
        else:
            print(f"✗ 质量不足: {raw_output[:80]}...")

        # Checkpoint
        if (idx + 1) % batch_check_point == 0:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({'next_idx': idx + 1, 'results': results}, f, ensure_ascii=False)
            print(f"  [Checkpoint] 已保存 {len(results)} 条有效数据")

        time.sleep(1)  # 避免限流

    # 最终保存
    out_path = os.path.join(output_dir, 'ime_m_training_data.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"训练数据生成完成！")
    print(f"有效数据: {len(results)}/{len(df)}")
    print(f"保存路径: {out_path}")
    print(f"{'=' * 60}")

    return results


def convert_to_lora_format(training_data, output_dir):
    """转换为LoRA训练格式"""
    os.makedirs(output_dir, exist_ok=True)

    lora_data = []
    for item in training_data:
        # 用多样化的instruction
        instruction = random.choice(IME_INSTRUCTIONS)
        output_text = (
            f"Innovation: {item['Innovation']}\n"
            f"Method: {item['Method']}\n"
            f"维度: {item['维度']}\n"
            f"法则: {item['法则']}\n"
            f"Effect: {item['Effect']}"
        )

        lora_data.append({
            "instruction": instruction,
            "input": item['abstract'],
            "output": output_text,
        })

    # 数据增强：每条数据用3种不同instruction变体
    augmented = []
    for item in lora_data:
        augmented.append(item)
        for _ in range(2):
            new_instruction = random.choice(IME_INSTRUCTIONS)
            augmented.append({
                "instruction": new_instruction,
                "input": item['input'],
                "output": item['output'],
            })

    random.shuffle(augmented)

    # 分train/val
    split_idx = int(len(augmented) * 0.95)
    train_data = augmented[:split_idx]
    val_data = augmented[split_idx:]

    train_path = os.path.join(output_dir, 'train.json')
    val_path = os.path.join(output_dir, 'val.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"\nLoRA训练数据已生成:")
    print(f"  原始数据: {len(lora_data)}")
    print(f"  增强后: {len(augmented)} (3x)")
    print(f"  训练集: {len(train_data)} → {train_path}")
    print(f"  验证集: {len(val_data)} → {val_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='专利CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='/tmp/ime-patent-extractor/lora_data')
    parser.add_argument('--max', type=int, default=200, help='最大处理专利数')
    parser.add_argument('--skip_generation', action='store_true', help='跳过API调用，只转换已有数据')
    args = parser.parse_args()

    if args.skip_generation:
        # 从checkpoint加载已有数据并转换
        ckpt_path = os.path.join(args.output_dir, 'checkpoint.json')
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                results = json.load(f).get('results', [])
            convert_to_lora_format(results, args.output_dir)
        else:
            print("没有找到checkpoint文件")
    else:
        if not args.csv:
            # 使用默认的专利数据
            default_csv = '/home/jiyong/桌面/ime论文/experiments/processed/all_patents.csv'
            if os.path.exists(default_csv):
                args.csv = default_csv
            else:
                print("请提供专利CSV文件: --csv path/to/patents.csv")
                sys.exit(1)

        results = generate_training_data(args.csv, args.output_dir, max_patents=args.max)
        convert_to_lora_format(results, args.output_dir)
