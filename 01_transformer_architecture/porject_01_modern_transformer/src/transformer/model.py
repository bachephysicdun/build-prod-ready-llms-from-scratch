import torch.nn as nn

from transformer.blocks import TransformerBlock
from components.rope import get_rotation_matrix


class Transformer(nn.Module):
    """
    Transformer (Decoder only) Model with Sliding Window Multihead Attention and Mixture of Experts (MoE) layers.
    The architecture consists of:
    1. Token Embedding layer to convert input token ids to dense vectors.
    2. A stack of Transformer blocks (see TransformerBlock class) each containing:
       - Sliding Window Multihead Self-Attention with RoPE
       - Mixture of Experts (MoE) layer
       - RMSNorm layers and residual connections.
       - residual connections around both attention and MoE layers.
    3. Final projection layer to vocabulary logits.

    Args:
        vocabulary_size (int): size of the (input/output) vocabulary.
        hidden_size (int): dimension of hidden states (must be divisible by num_heads).
        num_heads (int): number of attention heads.
        window_size (int): size of the sliding window (must be odd to have a center).
        d_ff (int): dimension of the intermediate feedforward layer (expert capacity).
        num_experts (int): total number of experts in the MoE layer.
        n_experts_per_token (int): number of experts each token is routed to (top-k).
        n_blocks (int): number of stacked Transformer blocks.
        max_seq_len (int): Maximum sequence length (used for RoPE positional embeddings).
        period (float, Optional): period for RoPE (default: 10,000).
    """    
    def __init__(
            self,
            vocabulary_size: int,
            hidden_size: int, 
            num_heads: int, 
            window_size: int, 
            d_ff: int, 
            num_experts: int, 
            n_experts_per_token: int, 
            n_blocks: int,
            max_seq_len: int,
            period: float = 10_000.0
        ):

        super().__init__()

        head_dim = hidden_size // num_heads
        self.rotation_matrix = get_rotation_matrix(head_dim, max_seq_len, period) # [max_seq_len, head_dim // 2]

        # instantiate the components
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        
        # create a stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size = hidden_size, 
                num_heads = num_heads, 
                window_size = window_size, 
                d_ff = d_ff, 
                num_experts = num_experts, 
                n_experts_per_token = n_experts_per_token,
                rotation_matrix = self.rotation_matrix
            )
            for _ in range(n_blocks)
        ])

        # final projection layer to vocabulary logits
        self.out = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the Transformer model

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, vocabulary_size)
        """        
        
        # implement for the forward method
        # embed the input tokens
        x = self.embedding(x) # [batch_size, seq_length, hidden_size]
        
        # pass through the transformer blocks
        for block in self.blocks:
            x = block(x)    # [batch_size, seq_length, hidden_size]
        
        # project to vocabulary logits
        return self.out(x)  # [batch_size, seq_length, vocabulary_size]