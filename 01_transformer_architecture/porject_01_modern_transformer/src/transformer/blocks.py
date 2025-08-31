import torch
import torch.nn as nn

from components.norm_layers import RMSNorm
from components.attentions import EfficientSlidingWindowMultiheadAttention
from components.moe import MoeLayer


class TransformerBlock(nn.Module):
    """
    Transformer (Decoder only) Block with Sliding Window Multihead Attention and Mixuter of Experts (MoE) layer.
    This block consists of:
    1. Sliding Window Multihead Self-Attnetion laywer with rotary positional embeddings.
    2. Mixture of Experts (MoE) layer with multiple FeedForward experts and a gating network (router).
    3. RMSNorm layers for normalization before attention and MoE layers.
    4. Residual connections around both attention and MoE layers.

    Args:
        hidden_size (int): dimension of hidden states (must be divisible by num_heads).
        num_heads (int): number of attention heads.
        window_size (int): size of the sliding window (must be odd to have a center).
        d_ff (int): dimension of the intermediate feedforward layer (expert capacity).
        num_experts (int): total number of experts in the MoE layer.
        n_experts_per_token (int): number of experts each token is routed to (top-k).
        rotation_matrix (torch.Tensor): precomputed RoPE rotation matrix of shape [context_size, head_dim // 2]
    """    
    def __init__(
          self, 
          hidden_size: int, 
          num_heads: int, 
          window_size: int, 
          d_ff: int, 
          num_experts: int, 
          n_experts_per_token: int,
          rotation_matrix: torch.Tensor
        ) -> None:
        super().__init__()

        # instantiate the different components
        # sliding window multihead attention with RoPE
        self.attn = EfficientSlidingWindowMultiheadAttention(hidden_size, num_heads, window_size, rotation_matrix)
        
        # Mixture of Experts (MoE) layer 
        self.moe_layer = MoeLayer(hidden_size, d_ff, num_experts, n_experts_per_token)
        
        # RMSNorm layers
        self.norm_layer1 = RMSNorm(hidden_size)
        self.norm_layer2 = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer Block forward pass

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """        
        
        # implement for the forward logic
        # self-attention (with rmsnor normalization) + residual connection
        attn_out = self.attn(self.norm_layer1(x))   # [batch_size, seq_length, hidden_size]
        x = x + attn_out    # [batch_size, seq_length, hidden_size]
        
        # MoE forward (with rmsnor normalization) + residual connection
        moe_out = self.moe_layer(self.norm_layer2(x))  # [batch_size, seq_length, hidden_size]
        x = x + moe_out  # [batch_size, seq_length, hidden_size]
        
        return x