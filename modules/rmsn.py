import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.w = torch.nn.Parameter(torch.ones(self.dim))

    def forward(self, x):

        rms = torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        return x * rms * self.w


