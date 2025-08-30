import torch.nn as nn

from components.norm_layers import RMSNorm
from components.attentions import EfficientSlidingWindowMultiheadAttention
from components.moe import MoeLayer


class TransformerBlock(nn.Module):
    def __init__(
          self, 
          hidden_size, 
          num_heads, 
          window_size, 
          d_ff, 
          num_experts, 
          n_experts_per_token,
          rotation_matrix
        ) -> None:
        super().__init__()

        # instantiate the different components
        self.attn = EfficientSlidingWindowMultiheadAttention(hidden_size, num_heads, window_size)
        self.moe_layer = MoeLayer(hidden_size, d_ff, num_experts, n_experts_per_token)
        self.norm_layer1 = RMSNorm(hidden_size)
        self.norm_layer2 = RMSNorm(hidden_size)
        self.rotation_matrix = rotation_matrix

    def forward(self, x):
        # implement for the forward logic
        attn_out = self.attn(self.norm_layer1(x))   # [batch_size, seq_length, hidden_size]
        x = x + attn_out    # [batch_size, seq_length, hidden_size]
        moe_out = self.moe_layer(self.norm_layer2(x))  # [batch_size, seq_length, hidden_size]
        x = x + moe_out  # [batch_size, seq_length, hidden_size]
        return x