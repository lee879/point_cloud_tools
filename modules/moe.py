# moe.py
import torch
import torch.nn as nn

class MoEFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate, num_experts=8, expert_size=None, top_k=2):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_size = expert_size or hidden_units * 4

        # 专家网络
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Linear(hidden_units, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def _create_expert(self):
        return nn.Sequential(
            nn.Linear(self.hidden_units, self.expert_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.expert_size, self.hidden_units),
            nn.Dropout(p=self.dropout_rate)
        )

    def forward(self, inputs):
        original_shape = inputs.shape
        inputs = inputs.view(-1, original_shape[-1])

        # 门控计算
        gate_logits = self.gate(inputs)
        weights = self.softmax(gate_logits)
        top_weights, top_indices = torch.topk(weights, k=self.top_k, dim=-1)

        # 专家计算
        outputs = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            # 选出分配给当前专家的样本
            mask = (top_indices == i).any(dim=-1)
            if mask.any():
                expert_output = expert(inputs[mask])
                # 加权求和
                expert_weights = top_weights[mask].gather(
                    1, (top_indices[mask] == i).long().argmax(dim=-1, keepdim=True))
                outputs[mask] += expert_output * expert_weights

        return outputs.view(*original_shape)

    def get_expert_usage(self, inputs):
        """获取专家使用情况统计"""
        original_shape = inputs.shape
        inputs = inputs.view(-1, original_shape[-1])
        
        gate_logits = self.gate(inputs)
        weights = self.softmax(gate_logits)
        top_weights, top_indices = torch.topk(weights, k=self.top_k, dim=-1)
        
        # 统计每个专家的使用次数
        expert_usage = torch.zeros(self.num_experts, device=inputs.device)
        for i in range(self.num_experts):
            mask = (top_indices == i).any(dim=-1)
            expert_usage[i] = mask.sum()
            
        return expert_usage
