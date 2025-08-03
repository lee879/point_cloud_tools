import torch
import torch.nn as nn



class Rope(nn.Module):
    def __init__(self,head_dim: int, base:int = 100000):
        super(Rope, self).__init__()
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("inv_freq",self._compute_inv_freq(),persistent=False)

    def _compute_inv_freq(self):
        k = torch.arange(0,self.head_dim,2,dtype=torch.float32)
        return 1/(self.base ** (k/self.head_dim))
    def _rota_half_emb(self,x):
        x0,x1 = x.chunk(2,dim=-1)
        return torch.cat((-x1,x0),dim=-1)

    def _apply_rope(self,x,freq):
        freq = freq.to(x.device)
        cos = torch.cos(freq)
        sin = torch.sin(freq)
        return x * cos + self._rota_half_emb(x) * sin

    def forward(self, x, start_idx=0):
        seq_dim = x.size(1)
        pos = torch.arange(start_idx,seq_dim + start_idx,dtype=torch.float32).to(x.device)
        freq = torch.einsum("i,j->ij",pos, self.inv_freq)
        freqs = torch.cat((freq,freq),dim = -1)
        freqs = freqs.view(1, seq_dim, 1,self.head_dim)
        return self._apply_rope(x,freqs)








