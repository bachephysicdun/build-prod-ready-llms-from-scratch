import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        # implement the normalization
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Do the math in float32 for safety (by x.float()), but return the result in the 
        # same dtype (by type_as(x)) as the input to keep training efficient 
        # (e.g., if the input was float16 (for efficiency), the output will also be float16.)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight