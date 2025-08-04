import torch.nn as nn
import torch
from attentions import MultiAttention
from rmsn import RMSNorm
from  moe import MoEFeedForward
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, head_nums, dropout_rate=0.0, training=True, mask=None):
        super(TransformerBlock, self).__init__()

        if training:
            self.mask = mask
            self.dropout_rate = dropout_rate
        else:
            self.mask = None
            self.dropout_rate = 0.0

        self.rmsn = RMSNorm(in_channels)
        self.attn = MultiAttention(in_channels, head_nums, training, dropout_rate)
        #支持混合专家，混合专家使用分布式训练开启CP策略
        #self.moe = MoEFeedForward(out_channels, dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * in_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2 * in_channels, 2 * in_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2 * in_channels, out_channels),
        )
    def forward(self, x):
        x_norm = self.rmsn(x)
        attn_output = self.attn(x_norm,self.mask)
        x_norm = self.rmsn(attn_output)
        mlp_output = self.mlp(x_norm)
        return x + mlp_output

if __name__ == '__main__':
    seq_len = 32
    in_channels = 256
    out_channels = 256
    head_nums = 8
    dropout_rate=0.0
    training=True
    mask=None
    input_data = torch.randn(1, seq_len, in_channels)
    transformerblock = TransformerBlock(in_channels, out_channels, head_nums, dropout_rate=dropout_rate, training=training,mask = mask)
    print(transformerblock(input_data).shape)