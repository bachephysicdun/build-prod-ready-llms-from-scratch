import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import SiGLU


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super().__init__()
        # instantiate 3 linear layers
        self.linear1 = nn.Linear(d_ff, hidden_size)
        self.linear2 = nn.Linear(d_ff, hidden_size)
        self.linear3 = nn.Linear(hidden_size, d_ff)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        # implement the expert logic
        x1 = self.linear1(x)    # [batch_size, seq_length, d_ff]
        x2 = self.linear2(x)    # [batch_size, seq_length, d_ff]
        x = SiGLU(x1 * x2)      # [batch_size, seq_length, d_ff]
        x = self.linear3(x)     # [batch_size, seq_length, hidden_size]
        return x


class MoeLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, num_experts, n_experts_per_token):
        super().__init__()

        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        # instantiate the experts and the gate
        self.experts = nn.ModuleList([
            FeedForward(hidden_size=hidden_size, d_ff=d_ff) 
            for _ in range(num_experts)
        ]) 
        self.gate = nn.Linear()

    def forward(self, x):
        # TODO: pass the input x to the gate
        # TODO: use torch.topk to get the topk values and indexes
        # TODO: pass the topk values to F.softmax to get the weights for each expert 
  
        out = torch.zeros_like(x, device=x.device)
        for i, expert in enumerate(self.experts):
            # TODO: find the indexes of the hidden states that should be routed to the current expert
            # TODO: update the out tensor
            pass
        return out