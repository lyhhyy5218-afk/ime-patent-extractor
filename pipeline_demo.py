#!/usr/bin/env python3
"""
端到端Pipeline: 专利摘要 → IME抽取 + M九维九法分类
先用Qwen-Max API跑少量样例验证质量，后续再批量生成LoRA训练数据。
"""
import os
import sys
import json
import re
import time
import argparse
import dashscope
from dashscope import Generation

DASHSCOPE_API_KEY = 'sk-4af4ab13c47d4408aabaa93c32273a48'
MODEL = 'qwen-max'
dashscope.api_key = DASHSCOPE_API_KEY

sys.path.insert(0, os.path.dirname(__file__))
from ime_m_classify_prompt import (
    IME_SYSTEM_PROMPT, M_CLASSIFY_SYSTEM_PROMPT,
    IME_ICL_EXAMPLES, M_CLASSIFY_ICL_EXAMPLES,
    build_m_classify_user_prompt,
)

# ═══════════════════════════════════════════
# 测试专利
# ═══════════════════════════════════════════
TEST_PATENTS = [
    {
        "id": 1,
        "domain": "电机",
        "text": "本发明公开了一种高效散热的永磁同步电机转子结构。所述转子铁芯沿轴向开设有多条通风槽，通风槽内壁设有散热翅片，转子铁芯外圆周面涂覆有高导热绝缘层。通过通风槽和散热翅片的协同作用，显著提高了转子的散热效率，有效降低了绕组温升，使电机能够在更高功率密度下长期稳定运行。"
    },
    {
        "id": 2,
        "domain": "电机",
        "text": "本发明涉及一种低齿槽转矩的永磁电机转子。在转子铁芯的每个磁极之间设置辅助槽，辅助槽的形状为V形，深度为转子铁芯外径的5%-8%。通过优化辅助槽的位置和尺寸参数，有效降低了齿槽转矩，减小了转矩波动，使电机运行更加平稳，特别适合高精度伺服控制场合。"
    },
    {
        "id": 3,
        "domain": "机器人",
        "text": "本发明公开了一种工业机器人关节模块的柔性传动装置。包括谐波减速器和力矩传感器，力矩传感器集成在谐波减速器的柔轮内侧，采用应变片式结构。通过将力矩传感与减速功能一体化设计，实现了关节模块的紧凑化，减小了关节体积和重量，同时提高了力控精度和响应速度。"
    },
    {
        "id": 4,
        "domain": "锂电池",
        "text": "本发明提供一种高安全性的锂离子电池隔膜。所述隔膜包括聚乙烯基层和涂覆在基层两侧的陶瓷涂层，陶瓷涂层由氧化铝纳米颗粒和聚偏氟乙烯粘结剂组成。陶瓷涂层提高了隔膜的耐热性能，在高温下能够维持结构稳定不收缩，有效防止了电池内部短路，大幅提升了电池的安全性和循环寿命。"
    },
    {
        "id": 5,
        "domain": "机器人",
        "text": "本发明涉及一种协作机器人的碰撞检测与安全防护方法。通过在机器人各关节安装六维力传感器实时检测外部力，结合改进的动量观测器算法进行碰撞检测。当检测到异常力信号时，控制系统在5ms内触发安全停机。该方法有效提高了人机协作的安全性，能够区分有意接触和意外碰撞，减少误停机率。"
    },
    {
        "id": 6,
        "domain": "电机",
        "text": "本发明公开了一种采用碳纤维绑环的永磁电机转子。转子永磁体外部设置碳纤维绑环，绑环采用多层缠绕工艺，预紧力逐层递增。碳纤维绑环替代了传统的不锈钢护套，在保证转子结构强度的同时，显著降低了转子重量和转动惯量，提高了电机的动态响应性能和最高转速。"
    },
    {
        "id": 7,
        "domain": "机器人",
        "text": "本发明涉及一种基于视觉引导的工业机器人抓取系统。通过在机械臂末端安装深度相机获取工件三维信息，结合改进的PointNet++点云分割算法识别工件位姿。系统根据识别结果自动规划抓取路径，并通过力控手爪实现自适应夹持，能够处理无序堆放的多种工件，抓取成功率达99.2%。"
    },
    {
        "id": 8,
        "domain": "锂电池",
        "text": "本发明提供一种具有梯度孔隙率结构的锂电池正极材料。所述正极材料由内到外孔隙率逐渐增大，内层为高密度活性材料提供高容量，外层为大孔隙结构促进电解液渗透和锂离子传输。通过梯度孔隙率设计，在保持高能量密度的同时，显著提高了电池的倍率性能和循环稳定性。"
    },
]


def call_llm(messages, temperature=0.1, max_tokens=512):
    """调用dashscope Qwen API"""
    resp = Generation.call(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        result_format='message',
    )
    if resp.status_code == 200:
        return resp.output.choices[0].message.content
    else:
        print(f"  [API错误] {resp.status_code}: {resp.message}")
        return ""


def extract_ime(patent_text):
    """Step 1: 从专利摘要抽取IME三元组"""
    # 构建带ICL的prompt
    messages = [{"role": "system", "content": IME_SYSTEM_PROMPT}]

    # 加ICL示例（取前2个）
    for ex in IME_ICL_EXAMPLES[:2]:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # 实际输入
    messages.append({"role": "user", "content": patent_text})

    result = call_llm(messages, temperature=0.1, max_tokens=256)
    return result


def classify_method(method_text, innovation_text="", effect_text="", rag_context=""):
    """Step 2: 对Method进行九维九法分类"""
    system_prompt = M_CLASSIFY_SYSTEM_PROMPT.format(rag_context=rag_context)

    messages = [{"role": "system", "content": system_prompt}]

    # 加ICL示例
    for ex in M_CLASSIFY_ICL_EXAMPLES[:3]:
        user_msg = build_m_classify_user_prompt(ex["method"], ex["innovation"], ex["effect"])
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})

    # 实际输入
    user_msg = build_m_classify_user_prompt(method_text, innovation_text, effect_text)
    messages.append({"role": "user", "content": user_msg})

    result = call_llm(messages, temperature=0.1, max_tokens=256)
    return result


def parse_ime(raw_text):
    """解析IME抽取结果"""
    result = {"Innovation": "", "Method": "", "Effect": "", "raw": raw_text}
    for field in ["Innovation", "Method", "Effect"]:
        patterns = [
            rf'{field}\s*[:：]\s*(.+?)(?:\s*\|\s*|\s*$)',
            rf'{field}\s*[\(（][^\)）]*[\)）]\s*[:：]\s*(.+?)(?:\s*\|\s*|\s*$)',
        ]
        for p in patterns:
            m = re.search(p, raw_text, re.IGNORECASE)
            if m:
                result[field] = m.group(1).strip()
                break
    return result


def parse_m_classify(raw_text):
    """解析M分类结果"""
    try:
        # 尝试提取JSON
        json_match = re.search(r'\{[^}]+\}', raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {"raw": raw_text, "parse_error": True}


def run_demo():
    """运行小样例Demo"""
    print("=" * 70)
    print("IME抽取 + M九维九法分类 Demo")
    print("=" * 70)

    results = []

    for patent in TEST_PATENTS:
        print(f"\n{'─' * 60}")
        print(f"[专利{patent['id']}] 领域: {patent['domain']}")
        print(f"摘要: {patent['text'][:80]}...")

        # Step 1: IME抽取
        print("\n  [Step 1] IME抽取...")
        ime_raw = extract_ime(patent['text'])
        ime = parse_ime(ime_raw)
        print(f"    Innovation: {ime['Innovation']}")
        print(f"    Method:     {ime['Method']}")
        print(f"    Effect:     {ime['Effect']}")

        time.sleep(1)  # 避免限流

        # Step 2: M分类
        if ime['Method']:
            print("\n  [Step 2] M九维九法分类...")
            m_raw = classify_method(
                ime['Method'],
                ime['Innovation'],
                ime['Effect'],
            )
            m_result = parse_m_classify(m_raw)
            if 'parse_error' in m_result:
                print(f"    原始输出: {m_raw[:200]}")
            else:
                print(f"    维度: {m_result.get('维度', '?')}")
                print(f"    法则: {m_result.get('法则', '?')}")
                print(f"    置信度: {m_result.get('置信度', '?')}")
                print(f"    理由: {m_result.get('理由', '?')}")
        else:
            m_result = {"error": "IME抽取失败，Method为空"}

        results.append({
            "patent_id": patent['id'],
            "domain": patent['domain'],
            "abstract": patent['text'],
            "ime": ime,
            "m_classify": m_result,
        })

        time.sleep(1)

    # 保存结果
    out_path = '/tmp/ime-patent-extractor/demo_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Demo完成，结果已保存: {out_path}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    run_demo()
