import torch.nn as nn
import torch
from attentions import MultiAttention
from rmsn import RMSNorm
from moe import MoEFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, head_nums, dropout_rate=0.0, 
                 training=True, mask=None, use_moe=False, use_lora=False, 
                 moe_config=None, lora_config=None):
        super(TransformerBlock, self).__init__()

        if training:
            self.mask = mask
            self.dropout_rate = dropout_rate
        else:
            self.mask = None
            self.dropout_rate = 0.0

        self.use_moe = use_moe
        self.use_lora = use_lora

        self.rmsn = RMSNorm(in_channels)
        self.attn = MultiAttention(
            in_channels, head_nums, training, dropout_rate,
            use_lora=use_lora, lora_config=lora_config
        )
        
        # 前馈网络选择：MoE或标准MLP
        if use_moe:
            moe_config = moe_config or {}
            self.moe = MoEFeedForward(
                hidden_units=in_channels,
                dropout_rate=dropout_rate,
                num_experts=moe_config.get('num_experts', 8),
                expert_size=moe_config.get('expert_size', None),
                top_k=moe_config.get('top_k', 2)
            )
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, 2 * in_channels),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2 * in_channels, 2 * in_channels),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(2 * in_channels, out_channels),
            )
            self.moe = None

    def forward(self, x):
        x_norm = self.rmsn(x)
        attn_output = self.attn(x_norm, self.mask)
        x_norm = self.rmsn(attn_output)
        
        # 选择使用MoE或标准MLP
        if self.use_moe and self.moe is not None:
            mlp_output = self.moe(x_norm)
        elif self.mlp is not None:
            mlp_output = self.mlp(x_norm)
        else:
            raise RuntimeError("Both moe and mlp are None!")
        
        return x + mlp_output

    def get_expert_usage(self, x):
        """获取MoE专家使用情况（仅在use_moe=True时有效）"""
        if not self.use_moe or self.moe is None:
            return None
        x_norm = self.rmsn(x)
        attn_output = self.attn(x_norm, self.mask)
        x_norm = self.rmsn(attn_output)
        return self.moe.get_expert_usage(x_norm)

if __name__ == '__main__':
    seq_len = 32
    in_channels = 256
    out_channels = 256
    head_nums = 8
    dropout_rate = 0.0
    training = True
    mask = None
    
    # 测试标准Transformer
    print("=== 测试标准Transformer ===")
    input_data = torch.randn(1, seq_len, in_channels)
    transformer_block = TransformerBlock(
        in_channels, out_channels, head_nums, 
        dropout_rate=dropout_rate, training=training, mask=mask
    )
    output = transformer_block(input_data)
    print(f"标准Transformer输出形状: {output.shape}")
    
    # 测试MoE Transformer
    print("\n=== 测试MoE Transformer ===")
    moe_config = {
        'num_experts': 8,
        'expert_size': 1024,
        'top_k': 2
    }
    transformer_moe = TransformerBlock(
        in_channels, out_channels, head_nums,
        dropout_rate=dropout_rate, training=training, mask=mask,
        use_moe=True, moe_config=moe_config
    )
    output_moe = transformer_moe(input_data)
    expert_usage = transformer_moe.get_expert_usage(input_data)
    print(f"MoE Transformer输出形状: {output_moe.shape}")
    print(f"专家使用情况: {expert_usage}")
    
    # 测试LoRA Transformer
    print("\n=== 测试LoRA Transformer ===")
    lora_config = {
        'rank': 16,
        'alpha': 32,
        'dropout': 0.1
    }
    transformer_lora = TransformerBlock(
        in_channels, out_channels, head_nums,
        dropout_rate=dropout_rate, training=training, mask=mask,
        use_lora=True, lora_config=lora_config
    )
    output_lora = transformer_lora(input_data)
    print(f"LoRA Transformer输出形状: {output_lora.shape}")
    
    # 测试MoE + LoRA组合
    print("\n=== 测试MoE + LoRA组合 ===")
    transformer_combined = TransformerBlock(
        in_channels, out_channels, head_nums,
        dropout_rate=dropout_rate, training=training, mask=mask,
        use_moe=True, use_lora=True, 
        moe_config=moe_config, lora_config=lora_config
    )
    output_combined = transformer_combined(input_data)
    expert_usage_combined = transformer_combined.get_expert_usage(input_data)
    print(f"MoE + LoRA Transformer输出形状: {output_combined.shape}")
    print(f"专家使用情况: {expert_usage_combined}")