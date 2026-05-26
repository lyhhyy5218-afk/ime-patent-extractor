# IME Patent Extractor

从专利摘要中抽取 IME（Innovation-Method-Effect）三元组的小模型，基于 Qwen2.5-1.5B LoRA 微调。

## 背景

本项目是论文《基于IME三元组的专利技术创新信息抽取研究》的配套模型。通过知识蒸馏方法，将大模型（Qwen-Max）的 IME 抽取能力迁移到 1.5B 参数的小模型上，使其能在本地 GPU 上高效运行，无需调用大模型 API。

### IME 三元组定义

| 字段 | 含义 | 示例 |
|------|------|------|
| **Innovation** | 被创新的技术主体 | 脱模机驱动系统 |
| **Method** | 实现创新的具体技术手段 | 采用直线电机替换旋转电机和液压缸 |
| **Effect** | 原文中明确提到的技术效果 | 提高工作效率，降低维修保养成本 |

## 模型性能

在 142 条专利测试集上的评估结果：

| 指标 | 值 |
|------|-----|
| IME 格式正确率 | 100.0% |
| 效果捕获率 (ECR) | 85.7%（教师模型 Qwen-Max: 79.5%） |
| Effect 非空率 | 100.0% |
| Effect 重复率 | 0.0% |
| ROUGE-L Overall | 0.471 |
| 推理速度 | ~1.1s/条 |

## 使用方法

### 安装依赖

```bash
pip install torch transformers peft
```

### 下载模型

```bash
git lfs install
git clone https://github.com/lyhhyy5218-afk/ime-patent-extractor.git
```

### 推理

```python
from inference import load_model, extract

model, tokenizer = load_model()

text = "本发明公开了一种基于直线电机驱动的脱模机，包括机架、脱模板和驱动装置，所述驱动装置采用直线电机..."
result = extract(text, model, tokenizer)
print(result)
# Innovation: 脱模机驱动装置 | Method: 采用直线电机驱动 | Effect: 提高工作效率
```

### 命令行使用

```bash
# 单条文本
python inference.py --text "专利摘要文本..."

# 批量处理
python inference.py --file patents.txt
```

## 训练细节

- **基座模型**: Qwen/Qwen2.5-1.5B
- **微调方法**: LoRA (r=32, alpha=64)
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **训练数据**: 1411 条专利（经 3x 数据增强后 3381 条）
- **训练轮次**: 3 epochs
- **学习率**: 8e-5 (cosine schedule)
- **反幻觉策略**: 10 种不同的 anti-hallucination instruction 模板
- **推理参数**: max_new_tokens=55, repetition_penalty=1.15

## 硬件要求

- GPU: >= 8GB 显存（推荐 NVIDIA RTX 3060 及以上）
- 内存: >= 16GB
- 磁盘: ~200MB（LoRA adapter）

## 引用

如果您在研究中使用了本模型，请引用：

```bibtex
@article{ime_patent_2026,
  title={基于IME三元组的专利技术创新信息抽取研究},
  author={Jiyong Li},
  year={2026}
}
```

## 许可证

MIT License
