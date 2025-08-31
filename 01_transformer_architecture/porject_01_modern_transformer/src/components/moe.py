import torch
import torch.nn as nn
import torch.nn.functional as F
from components.activations import SiGLU


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super().__init__()
        # instantiate linear layers and SiGLU activation
        self.w3 = nn.Linear(d_ff, hidden_size)
        self.siglu = SiGLU(d_ff, d_ff) # SiGLU already does two separate linear layers w1 and w2 internally

    def forward(self, x) -> torch.Tensor:
        # implement the expert logic
        # siglu expects a single input x and internally applies two linear layers.
        x = self.siglu(x)  # W1 x * Sigmoid(W2 x) -> [batch_size, seq_length, d_ff]
        x = self.w3(x)     # [batch_size, seq_length, hidden_size]
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
        self.gate = nn.Linear(hidden_size, num_experts, bias=False) # a gate network or router determines which tokens are sent to which experts.

    def forward(self, x):
        # pass the input x to the gate
        gout = self.gate(x)    # [batch_size, seq_length, num_experts]
        # use torch.topk to get the topk values and indexes
        topk_values, topk_indexes = torch.topk(gout, self.n_experts_per_token, dim=-1) # each tensor has shape of [batch_size, seq_length, n_experts_per_token]

        # pass the topk values to F.softmax to get the weights for each expert 
        topk_weights = F.softmax(topk_values, dim=-1)
  
        out = torch.zeros_like(x, device=x.device)  # [batch_size, seq_length, hidden_size]
        for i, expert in enumerate(self.experts):
            # find the indexes of the hidden states that should be routed to the current expert
            batch_idx, token_idx, topk_idx = torch.where(topk_indexes == i) # -> [n_selected token routed to exper i], [n_selected], [n_selected]
            
            # update the out tensor
            # topk_weights[batch_idx, token_idx, topk_idx, None] has shape of [n_selected, 1]
            # out[batch_idx, token_idx, :] has shape of [n_selected, hidden_size]
            # expert(x[batch_idx, token_idx, :]) has shape of [n_selected, hidden_size]
            out[batch_idx, token_idx, :] += topk_weights[batch_idx, token_idx, topk_idx, None] * expert(x[batch_idx, token_idx, :])
            
        return out  # [batch_size, seq_length, hidden_size]