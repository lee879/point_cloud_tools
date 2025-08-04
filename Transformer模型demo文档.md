# Transformer模型详细demo文档（初学者友好版）

## 概述

本项目实现了一个基于PyTorch的Transformer模型，包含了现代Transformer架构的核心组件：多头注意力机制（Multi-Head Attention）、RMSNorm归一化、旋转位置编码（RoPE）、LoRA参数高效微调、MoE专家混合等。文档将以教学视角，详细讲解每个模块的原理、作用、参数和使用方法，适合初学者学习和动手实践。

---

## 目录结构

```
modules/
├── transformer.py    # 主要的Transformer块实现
├── attentions.py     # 多头注意力机制（支持LoRA）
├── rmsn.py           # RMSNorm归一化层
├── rope.py           # 旋转位置编码(RoPE)
├── moe.py            # 专家混合(MoE)
└── lora.py           # LoRA适配器
```

---

## 1. TransformerBlock（transformer.py）

### 1.1 作用
TransformerBlock 是Transformer的基本单元，包含：
- 归一化（RMSNorm）
- 多头注意力（MultiAttention）
- 前馈网络（MLP或MoE）
- 残差连接

### 1.2 参数详解
| 参数名         | 作用说明                                   |
|:--------------|:--------------------------------------------|
| in_channels   | 输入特征维度                                |
| out_channels  | 输出特征维度                                |
| head_nums     | 注意力头数                                  |
| dropout_rate  | Dropout比率，防止过拟合                     |
| training      | 是否为训练模式，影响Dropout和mask           |
| mask          | 注意力掩码，控制哪些位置可被关注            |
| use_moe       | 是否启用MoE专家混合                         |
| use_lora      | 是否启用LoRA参数高效微调                    |
| moe_config    | MoE配置字典（专家数、隐藏层、top-k等）       |
| lora_config   | LoRA配置字典（秩、缩放因子、dropout等）      |

### 1.3 前向传播流程（带注释）
```python
# x: [batch, seq_len, in_channels]
def forward(self, x):
    x_norm = self.rmsn(x)                    # 步骤1：归一化
    attn_output = self.attn(x_norm, self.mask)  # 步骤2：多头注意力
    x_norm = self.rmsn(attn_output)          # 步骤3：再次归一化
    if self.use_moe:
        mlp_output = self.moe(x_norm)        # 步骤4a：MoE前馈
    else:
        mlp_output = self.mlp(x_norm)        # 步骤4b：标准MLP
    return x + mlp_output                    # 步骤5：残差连接
```

### 1.4 典型用法
```python
from modules.transformer import TransformerBlock

# 基础用法
block = TransformerBlock(256, 256, 8)

# 启用LoRA
block_lora = TransformerBlock(256, 256, 8, use_lora=True, lora_config={'rank':8, 'alpha':16, 'dropout':0.1})

# 启用MoE
block_moe = TransformerBlock(256, 256, 8, use_moe=True, moe_config={'num_experts':4, 'expert_size':512, 'top_k':2})

# 同时启用LoRA和MoE
block_both = TransformerBlock(256, 256, 8, use_lora=True, use_moe=True,
    lora_config={'rank':8, 'alpha':16}, moe_config={'num_experts':4})

# 前向传播
x = torch.randn(2, 32, 256)
out = block_both(x)
print(out.shape)
```

---

## 2. MultiAttention（attentions.py）

### 2.1 原理简介
多头注意力机制可以让模型在不同子空间并行关注不同信息。其核心流程：
- 输入通过线性层生成Q（查询）、K（键）、V（值）
- 计算注意力分数，聚合信息
- 多个头并行，最后拼接

### 2.2 关键参数
- `input_dim`：输入特征维度
- `head_num`：头数，越多表达力越强
- `use_lora`：是否对注意力权重用LoRA微调
- `lora_config`：LoRA参数

### 2.3 LoRA背景知识
LoRA（Low-Rank Adaptation）是一种高效微调大模型的方法。它只训练很小的低秩矩阵，大幅减少参数量和显存消耗，适合下游任务微调。

### 2.4 典型用法
```python
from modules.attentions import MultiAttention
attn = MultiAttention(256, 8, use_lora=True, lora_config={'rank':4, 'alpha':8})
```

---

## 3. RMSNorm（rmsn.py）

### 3.1 原理简介
RMSNorm是LayerNorm的简化版，只用均方根归一化，不减均值，速度快、参数少。

### 3.2 代码片段
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        ...
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.w
```

---

## 4. RoPE（rope.py）

### 4.1 原理简介
RoPE（旋转位置编码）通过复数旋转将位置信息编码进特征，支持长序列外推。

### 4.2 代码片段
```python
class Rope(nn.Module):
    def __init__(self, head_dim, base=10000):
        ...
    def forward(self, x):
        ... # 见源码
```

---

## 5. MoEFeedForward（moe.py）

### 5.1 MoE是什么？
MoE（Mixture of Experts）是一种稀疏激活的前馈网络。每次只激活部分专家，大幅提升模型容量但计算量只略增。

### 5.2 关键参数
- `num_experts`：专家数量
- `expert_size`：每个专家的隐藏层大小
- `top_k`：每个输入最多激活几个专家

### 5.3 典型用法
```python
from modules.moe import MoEFeedForward
moe = MoEFeedForward(256, 0.1, num_experts=4, expert_size=512, top_k=2)
```

---

## 6. LoRAAdapter/LoRALinear（lora.py）

### 6.1 LoRA原理
LoRA通过低秩分解，只训练很小的A、B矩阵，冻结原始大权重，极大节省参数和显存。

### 6.2 用法
- `LoRALinear`：直接替换nn.Linear
- `LoRAAdapter`：包裹已有nn.Linear

```python
from modules.lora import LoRALinear, LoRAAdapter
layer = nn.Linear(256, 256)
lora_layer = LoRAAdapter(layer, rank=8, alpha=16)
```

---

## 7. 综合示例与建议

### 7.1 综合用法
```python
import torch
from modules.transformer import TransformerBlock

# 配置
lora_config = {'rank': 8, 'alpha': 16, 'dropout': 0.1}
moe_config = {'num_experts': 4, 'expert_size': 512, 'top_k': 2}

# 创建支持LoRA和MoE的TransformerBlock
block = TransformerBlock(256, 256, 8, use_lora=True, lora_config=lora_config, use_moe=True, moe_config=moe_config)

# 输入
x = torch.randn(2, 32, 256)
output = block(x)
print('输出形状:', output.shape)

# 获取MoE专家使用情况
usage = block.get_expert_usage(x)
print('专家使用统计:', usage)
```

### 7.2 学习建议
1. **先理解基础注意力机制**：建议先手写单头注意力，理解Q、K、V的含义。
2. **逐步加深**：多头注意力、位置编码、归一化、残差连接。
3. **动手调试**：多打印中间变量，画图理解注意力分布。
4. **尝试不同配置**：如只用MLP、只用MoE、只用LoRA、全部叠加。
5. **查阅资料**：推荐阅读《Attention is All You Need》、LoRA和MoE相关论文。

---

## 8. 常见问题解答（FAQ）

- **Q: LoRA和MoE能一起用吗？**
  A: 可以，二者互不影响，LoRA节省参数，MoE提升容量。
- **Q: 如何只用标准Transformer？**
  A: 不传use_lora/use_moe参数即可。
- **Q: 如何调大模型容量？**
  A: 增加head_nums、in_channels、num_experts等。
- **Q: 为什么用RMSNorm不用LayerNorm？**
  A: RMSNorm更快、参数更少，适合大模型。

---

## 9. 参考资料
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [RMSNorm论文](https://arxiv.org/abs/1910.07467)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [MoE论文](https://arxiv.org/abs/1701.06538)

---

本项目代码结构清晰，注释丰富，适合初学者逐步学习和实验。欢迎多尝试不同配置，深入理解每个模块的作用！ 