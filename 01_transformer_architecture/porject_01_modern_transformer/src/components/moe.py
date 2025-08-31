import torch
import torch.nn as nn
import torch.nn.functional as F
from components.activations import SiGLU


class FeedForward(nn.Module):
    """
    FeedForward network used as an expert in the Mixture of Experts (MoE) layer.
    It consists of a SiGLU activation followed by a linear layer to project back to the hidden size.

    Args:
        hidden_size (int): Size of the input and output hidden representation.
        d_ff (int): dimension of the feedforward layer (expert capacity).
    """    
    def __init__(self, hidden_size: int, d_ff: int) -> None:
        super().__init__()
        # instantiate linear layers and SiGLU activation
        self.w3 = nn.Linear(d_ff, hidden_size)
        self.siglu = SiGLU(in_features=hidden_size, out_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FeedForward forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """        
        # implement the expert logic
        # apply SiGLU activation (W1(x) * sigmoid(W2(x)))
        # siglu expects a single input x and internally applies two linear layers.
        x = self.siglu(x)  # W1 x * Sigmoid(W2 x) -> [batch_size, seq_length, d_ff]
        
        # project back to hidden size
        x = self.w3(x)     # [batch_size, seq_length, hidden_size]
        return x


class MoeLayer(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int, num_experts: int, n_experts_per_token: int) -> None:
        """
        The Mixture of Experts (MoE) layer consists of multiple FeedForward experts. A gating network (or router)
        determines which tokens are sent to which experts by selecting subsets of experts per token (top-k selection) 
        and combines their outputs weighted by the gating scores.
        Args:
            hidden_size (int): dimension of the input and output hidden representation.
            d_ff (int): dimension of the intermediate feedforward layer (expert capacity).
            num_experts (int): total number of experts in the MoE layer.
            n_experts_per_token (int): number of experts each token is routed to (top-k).
        """        
        super().__init__()

        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        # instantiate the experts and the gate
        self.experts = nn.ModuleList([
            FeedForward(hidden_size=hidden_size, d_ff=d_ff) 
            for _ in range(num_experts)
        ]) 
        
        # gating (routing) network: computes logits for each expert
        self.gate = nn.Linear(hidden_size, num_experts, bias=False) # a gate network or router determines which tokens are sent to which experts.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        fowratd pass of the MoE layer
        each token is routed to 'n_experts_per_token' experts bsed on 
        the top-k gating scores computed by the gate network.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """
        
        # compute gating scores
        # pass the input x to the gate
        gout = self.gate(x)    # [batch_size, seq_length, num_experts]
        # use torch.topk to get the topk values and indexes
        topk_values, topk_indexes = torch.topk(gout, self.n_experts_per_token, dim=-1) # each tensor has shape of [batch_size, seq_length, n_experts_per_token]

        # pass the topk values to F.softmax to get the weights for each expert 
        topk_weights = F.softmax(topk_values, dim=-1)
  
        # initialize the output tensor
        out = torch.zeros_like(x, device=x.device)  # [batch_size, seq_length, hidden_size]
        
        # route the tokens to the selected experts and combine their outputs
        for i, expert in enumerate(self.experts):
            # find the indexes of the hidden states that should be routed to the current expert
            batch_idx, token_idx, topk_idx = torch.where(topk_indexes == i) # -> [n_selected token routed to exper i], [n_selected], [n_selected]
            if batch_idx.numel() == 0:
                continue  # skip if no tokens are routed to this expert

            # update the out tensor
            # topk_weights[batch_idx, token_idx, topk_idx, None] has shape of [n_selected, 1]
            # out[batch_idx, token_idx, :] has shape of [n_selected, hidden_size]
            # expert(x[batch_idx, token_idx, :]) has shape of [n_selected, hidden_size]
            out[batch_idx, token_idx, :] += topk_weights[batch_idx, token_idx, topk_idx, None] * expert(x[batch_idx, token_idx, :])
            
        return out  # [batch_size, seq_length, hidden_size]