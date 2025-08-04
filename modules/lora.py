import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) 线性层"""
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 原始权重矩阵 (冻结)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(2.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 冻结原始权重
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        
    def forward(self, x):
        # 原始线性变换
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA 低秩适应
        lora_output = F.linear(
            self.dropout(x), 
            self.lora_B @ self.lora_A
        ) * self.scaling
        
        return base_output + lora_output

class LoRAAdapter(nn.Module):
    """LoRA适配器，用于包装现有的线性层"""
    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 获取原始层的维度
        if hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        else:
            weight_shape = original_layer.weight.shape
            out_features, in_features = weight_shape
            
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(2.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 冻结原始层参数
        for param in original_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # 原始层输出
        original_output = self.original_layer(x)
        
        # LoRA 适应
        lora_output = F.linear(
            self.dropout(x), 
            self.lora_B @ self.lora_A
        ) * self.scaling
        
        return original_output + lora_output

