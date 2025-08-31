import torch
import torch.nn as nn
import torch.nn.functional as F
from components.rope import RoPE


class EfficientSlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int, rotation_matrix: torch.Tensor) -> None:
        """
        Efficient Sliding Window Multihead Self-Attention with RoPE

        This implementation uses 'torch.einsum' for efficient computation of attention scores and context vectors.
        It applies Rotary Position Embeddings (RoPE) to the queries and keys to encode positional information.
        A tensor unfolding technique is used to create sliding windows. The attention mechanism is restricted to a 
        sliding window, which reduces computational complexity while still capturing local dependencies.

        Args:
            hidden_size (int): dimension of hidden states (must be divisible by num_heads).
            num_heads (int): numnber of attention heads.
            window_size (int): size of the sliding window (must be odd to have a center).
            rotation_matrix (torch.Tensor): precomputed rotation matrix for RoPE of shape [context_size, head_dim // 2]
        """        
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        assert window_size % 2 == 1, f"window_size ({window_size}) must be odd to have a center"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        # linear projection for queries, keys and values
        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

        # create a position embedding attribute with RoPE
        # rotation_matrix = get_rotation_matrix(dim=self.head_dim, context_size=hidden_size, period=10_000)
        self.rope = RoPE(rotation_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of Efficient Sliding Window Multihead Attention with RoPE

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """        
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # create the queries, keys and values
        qkv = self.qkv_linear(x) # [batch_size, seq_length, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim) # [batch_size, seq_length, num_heads, 3 * head_dim]
        qkv = qkv.permute(0, 2, 1, 3) # [batch_size, num_heads, seq_length, 3 * head_dim]
        queries, keys, values = qkv.chunk(3, dim=-1)  #[batch_size, num_heads, seq_length, head_dim]

        # rotate the queries and keys using RoPE
        queries, keys = self.rope(queries, keys)  # same dimension: batch_size, num_heads, seq_length, head_dim]

        # pad the keys and values
        keys_padded = F.pad(input=keys, pad=(0, 0, padding, padding), mode="constant", value=0) # [batch_size, num_heads, seq_length + 2*padding, head_dim]
        values_padded = F.pad(input=values, pad=(0, 0, padding, padding), mode="constant", value=0)

        # Create sliding windows for keys and values
        keys_windows = keys_padded.unfold(dimension=2, size=self.window_size, step=1)  # [batch_size, num_heads, seq_length + 2*padding - window_size + 1, head_dim, window_size]
        values_windows = values_padded.unfold(dimension=2, size=self.window_size, step=1)  # [batch_size, num_heads, seq_length + 2*padding - window_size + 1, head_dim, window_size]

        # Compute attention scores
        # einsum string explanation:
        # b: batch_size, n: num_heads, s: seq_length, w: window_size, h: head_dim
        # queries: [batch_size, num_heads, seq_length, head_dim] ('bnsh')
        # keys_windows: [batch_size, num_heads, seq_length, window_size, head_dim] ('bnswh')
        # The einsum sums over h, which collapses the head_dim axis. That’s why the H dimension disappears.
        # The S dimension comes from queries (original seq_length) — that is why the result has length seq_length, not the (seq_length + 2*padding - window_size + 1).
        # einsum computes a dot product along h for each query position with its corresponding window of keys
        scores = torch.einsum('bnsh,bnshw->bnsw', queries, keys_windows)  # [batch_size, num_heads, seq_length, window_size]
        scores = scores / (self.head_dim ** 0.5)
        attentions = F.softmax(scores, dim=-1) # [batch_size, num_heads, seq_length, window_size]
        
        # multiply attentions to values_windows
        hidden_state = torch.einsum('bnsw,bnshw->bnsh', attentions, values_windows)  # [batch_size, num_heads, seq_length, head_dim]

        # Merge heads and combine the last two dimensions
        hidden_state = hidden_state.permute(0, 2, 1, 3).reshape((batch_size, seq_length, self.hidden_size))

        # perform the final linear transformation
        return self.out(hidden_state)
   



class SlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int) -> None:
        """
        Sliding Window Multihead Self-Attention
        This is a reference implementation of sliding window attention using an explicit for loop 
        over sequence positions. It is less effecient than the 'EfficientSlidingWindowMultiheadAttention' 
        implementation but is easier to understand. It restricts the attention mechanism to a sliding window, 
        which reduces computational complexity while still capturing local dependencies.

        Args:
            hidden_size (int): dimension of hidden states (must be divisible by num_heads).
            num_heads (int): number of attention heads.
            window_size (int): size of the sliding window (must be odd to have a center).
        """        
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        assert window_size % 2 == 1, f"window_size ({window_size}) must be odd to have a center"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of Sliding Window Multihead Attention

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """        
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Compute Q, K, V
        qkv = self.qkv_linear(x)    # [batch_size, seq_length, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Pad sequence for windowed attention
        keys = F.pad(input=keys, pad=(0, 0, padding, padding), mode="constant", value=0)    # [batch_size, num_heads, seq_length + 2*padding, head_dim]
        values = F.pad(input=values, pad=(0, 0, padding, padding), mode="constant", value=0)    # [batch_size, num_heads, seq_length + 2*padding, head_dim]

        # Initialize context tensors
        context = torch.zeros_like(queries, device=x.device)    # [batch_size, num_heads, seq_length, head_dim]

        # Compute attention for each sliding window
        for i in range(seq_length):
            # Determine the start and end of the window
            start = i
            end = i + self.window_size
            
            # Compute scores
            scores = torch.matmul(
                queries[:, :, i:i+1, :],    # [batch_size, num_heads, 1, head_dim]
                keys[:, :, start:end, :].transpose(-2, -1)  # [batch_size, num_heads, head_dim, window_size]
            )
            scores = scores / (self.head_dim ** 0.5)    # [batch_size, num_heads, 1, window_size]
            attention = F.softmax(scores, dim=-1)   # [batch_size, num_heads, 1, window_size]
            
            # Apply attention to values and add to context
            context[:, :, i:i+1, :] += torch.matmul(
                attention, # [batch_size, num_heads, 1, window_size]
                values[:, :, start:end, :]  # [batch_size, num_heads, window_size, head_dim]
            )   # [batch_size, num_heads, 1, head_dim]

        # Reshape context to (batch_size, seq_length, num_heads * head_dim)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)

        # Final linear layer
        output = self.out(context)  # [batch_size, seq_length, hidden_size]
        return output