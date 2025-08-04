import torch.nn as nn
import torch
from rope import Rope
import torch.nn.functional as F
from lora import LoRAAdapter

class MultiAttention(nn.Module):
    def __init__(self, input_dim, head_num, training=True, dropout_rate=0.0, use_lora=False, lora_config=None):
        super(MultiAttention, self).__init__()
        assert input_dim % head_num == 0
        self.input_dim = input_dim
        self.head_nums = head_num
        self.head_dim = input_dim // head_num

        self.qkv = nn.Linear(input_dim, input_dim * 3) #算子的融合
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.rope = Rope(self.head_dim)

        self.training = training
        self.dropout_rate = dropout_rate

        # LoRA集成
        if use_lora:
            lora_config = lora_config or {}
            rank = lora_config.get('rank', 16)
            alpha = lora_config.get('alpha', 32)
            dropout = lora_config.get('dropout', 0.0)
            self.qkv = LoRAAdapter(self.qkv, rank, alpha, dropout)
            self.out_proj = LoRAAdapter(self.out_proj, rank, alpha, dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 得到qkv矩阵
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 重塑为多头格式: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.head_nums, self.head_dim)
        k = k.view(batch_size, seq_len, self.head_nums, self.head_dim)
        v = v.view(batch_size, seq_len, self.head_nums, self.head_dim)

        # 旋转位置编码
        q = self.rope(q)
        k = self.rope(k)

        #交换维度得到[batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            #使用flash_attention_v2
            if mask is not None:
                #mask是0，1类型的
                if mask.dim() == 2:
                    # 扩展为 [batch_size, 1, 1, seq_len]
                    mask = mask[:, None, None, :]
                elif mask.dim() == 3:  # 2D掩码 (通常是因果掩码)
                    # 扩展为 [batch_size, 1, seq_len, seq_len]
                    mask = mask[:, None, :, :]
            attn = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=self.dropout_rate if self.training else 0.0)
        else:
            # 正常的attn的计算
            scores = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
            if mask is not None:
                #mask是0，1类型的
                if mask.dim() == 2:
                    # 扩展为 [batch_size, 1, 1, seq_len]
                    mask = mask[:, None, None, :]
                elif mask.dim() == 3:  # 2D掩码 (通常是因果掩码)
                    # 扩展为 [batch_size, 1, seq_len, seq_len]
                    mask = mask[:, None, :, :]

                scores = scores.masked_fill(mask == 0, -1e9) #将不需要的执行注意力的地方给一个很小的负数，经过softmax后就是0了
            attn = torch.matmul(F.softmax(scores, dim=-1), v)
        attn_out = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        return self.out_proj(attn_out) + x





















