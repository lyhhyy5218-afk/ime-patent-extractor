# IME Patent Extractor with 元易九维九法分类

从专利摘要中抽取 IME（Innovation-Method-Effect）三元组，并自动对 Method 进行元易创新方法的九维九法分类。基于 Qwen2.5-1.5B LoRA 微调。

## 功能

| 功能 | 说明 |
|------|------|
| **IME 三元组抽取** | 从专利摘要中提取 Innovation（创新对象）、Method（技术手段）、Effect（技术效果） |
| **元易九维九法分类** | 自动将 Method 分类到元易创新方法的 9 个维度 × 9 种法则 |
| **本地推理** | 1.5B 小模型，无需 API，单卡即可运行 |

### 输出格式

```
Innovation: 转子铁芯散热结构
Method: 在转子铁芯沿轴向开设多条通风槽并设置散热翅片，外圆周面涂覆高导热绝缘层
维度: 结构维
法则: 分解与去除
Effect: 提高转子散热效率降低绕组温升，使电机能在更高功率密度下稳定运行
```

### 元易九维九法体系

**九维（创新发生在哪个维度）：**

| 维度 | 判断标志 |
|------|----------|
| 空间维 | 改变了方向、位置、形状、体积、面积 |
| 环境维 | 改变了温度、湿度、光照、声音等环境条件 |
| 结构维 | 改变了部件及部件间的关系 |
| 功能维 | 增减了系统功能 |
| 机理维 | 改变了实现功能的科学原理 |
| 材料维 | 改变了材料构成或材料相态 |
| 动力体系维 | 改变了动力源、传动方式、储能方式 |
| 时序维 | 改变了作业顺序、时长、工序 |
| 人机关系维 | 改变了人与对象/环境/设备的关系 |

**九法（用什么法则实现创新）：**

| 法则 | 核心动作 |
|------|----------|
| 分解与去除 | 拆解、去除、分离、抽取 |
| 组合与集成 | 合并、一体化、统筹 |
| 局部优化 | 参数调整、不对称改变 |
| 替代 | 换掉、取代 |
| 动态化 | 静态变动态、增加可调节性 |
| 自服务 | 自我服务/保护/维护 |
| 友好化 | 使对人/环境伤害最小 |
| 柔性化 | 直变曲、硬变软 |
| 智慧化 | 加传感器、算法、控制系统 |

## 示例

### 示例 1：电机散热

**输入：**
> 本发明公开了一种高效散热的永磁同步电机转子结构。所述转子铁芯沿轴向开设有多条通风槽，通风槽内壁设有散热翅片，转子铁芯外圆周面涂覆有高导热绝缘层。通过通风槽和散热翅片的协同作用，显著提高了转子的散热效率，有效降低了绕组温升，使电机能够在更高功率密度下长期稳定运行。

**输出：**
```
Innovation: 转子铁芯散热结构
Method: 在转子铁芯沿轴向开设多条通风槽并设置散热翅片，外圆周面涂覆高导热绝缘层
维度: 结构维
法则: 组合与集成
Effect: 显著提高转子的散热效率，降低绕组温升，使电机能在更高功率密度下长期稳定运行
```

### 示例 2：锂电池隔膜

**输入：**
> 本发明提供一种高安全性的锂离子电池隔膜。所述隔膜包括聚乙烯基层和涂覆在基层两侧的陶瓷涂层，陶瓷涂层由氧化铝纳米颗粒和聚偏氟乙烯粘结剂组成。陶瓷涂层提高了隔膜的耐热性能，在高温下能够维持结构稳定不收缩，有效防止了电池内部短路，大幅提升了电池的安全性和循环寿命。

**输出：**
```
Innovation: 锂离子电池隔膜
Method: 在聚乙烯基层两侧涂覆由氧化铝纳米颗粒和聚偏氟乙烯粘结剂组成的陶瓷涂层
维度: 材料维
法则: 组合与集成
Effect: 提高隔膜耐热性能防止高温收缩导致内部短路，提升电池安全性和循环寿命
```

### 示例 3：机器人碰撞检测

**输入：**
> 本发明涉及一种协作机器人的碰撞检测与安全防护方法。通过在机器人各关节安装六维力传感器实时检测外部力，结合改进的动量观测器算法进行碰撞检测。当检测到异常力信号时，控制系统在5ms内触发安全停机。该方法有效提高了人机协作的安全性，能够区分有意接触和意外碰撞，减少误停机率。

**输出：**
```
Innovation: 协作机器人碰撞检测系统
Method: 在各关节安装六维力传感器结合改进的动量观测器算法进行碰撞检测
维度: 人机关系维
法则: 智慧化
Effect: 提高人机协作安全性，区分有意接触和意外碰撞，减少误停机率
```

### 示例 4：碳纤维替代

**输入：**
> 本发明公开了一种采用碳纤维绑环的永磁电机转子。转子永磁体外部设置碳纤维绑环，绑环采用多层缠绕工艺，预紧力逐层递增。碳纤维绑环替代了传统的不锈钢护套，在保证转子结构强度的同时，显著降低了转子重量和转动惯量，提高了电机的动态响应性能和最高转速。

**输出：**
```
Innovation: 永磁电机转子固定结构
Method: 用碳纤维绑环替代不锈钢护套，采用多层缠绕工艺预紧力逐层递增
维度: 材料维
法则: 替代
Effect: 降低转子重量和转动惯量，提高动态响应性能和最高转速
```

## 使用方法

### 安装依赖

```bash
pip install torch transformers peft
```

### 下载基座模型

```bash
# 从 HuggingFace 下载 Qwen2.5-1.5B
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./base_model
```

### 推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载模型
BASE_MODEL = "./base_model"  # 或 "Qwen/Qwen2.5-1.5B"
LORA_MODEL = "./model"       # 本 repo 的 model 目录

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()

# 推理
instruction = (
    "从以下专利摘要中提取IME三元组及Method的元易九维九法分类。\n"
    "输出格式：\n"
    "Innovation: xxx\nMethod: xxx\n维度: xxx维\n法则: xxx\nEffect: xxx"
)
abstract = "你的专利摘要..."

messages = [
    {"role": "system", "content": "你是一位专利技术分析专家，精通元易创新方法的九维九法分类体系。"},
    {"role": "user", "content": f"{instruction}\n\n专利摘要：\n{abstract}"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

## 训练细节

- **基座模型**: Qwen/Qwen2.5-1.5B
- **微调方法**: LoRA (r=32, alpha=64)
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **训练数据**: 500 条专利（由 Qwen-Max API 生成，经 3x 增强后 1500 条）
- **训练轮次**: 5 epochs
- **学习率**: 2e-4 (cosine schedule)
- **最终 eval_loss**: 0.075
- **推理速度**: ~46 tokens/s (单卡)

### 数据蒸馏流程

1. 构建**元易创新方法 RAG 知识库**（书稿九维定义 + 九法定义 + 维法耦合详解，共 969 条知识块）
2. 使用 Qwen-Max API + RAG 检索 + ICL 示例，批量生成 IME 三元组 + 九维九法分类
3. 10 种多样化 instruction 模板 × 3x 数据增强
4. LoRA 微调蒸馏到 Qwen2.5-1.5B

## 硬件要求

- GPU: >= 8GB 显存（推荐 NVIDIA RTX 3060 及以上）
- 内存: >= 16GB
- 磁盘: ~200MB（LoRA adapter）

## 项目结构

```
├── model/                        # LoRA adapter 权重
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...
├── ime_m_classify_prompt.py      # Prompt 模板和 ICL 示例
├── build_yuanyi_rag.py           # 元易 RAG 知识库构建脚本
├── generate_lora_data.py         # API 批量生成训练数据
├── train_ime_m_lora.py           # LoRA 训练脚本
├── inference.py                  # 推理脚本
└── README.md
```

## 引用

如果您在研究中使用了本模型，请引用：

```bibtex
@article{ime_patent_2026,
  title={基于IME三元组的专利创新景观构建与技术机会识别},
  author={Jiyong Li},
  year={2026}
}
```

## 许可证

MIT License
