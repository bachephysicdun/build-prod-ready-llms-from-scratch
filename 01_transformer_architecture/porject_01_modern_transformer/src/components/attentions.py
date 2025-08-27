import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientSlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

        # TODO: create a position embedding attribute with RoPE

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # create the queries, keys and values
        qkv = self.qkv_linear(x) # [batch_size, seq_length, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim) # [batch_size, seq_length, num_heads, 3 * head_dim]
        qkv = qkv.permute(0, 2, 1, 3) # [batch_size, num_heads, seq_length, 3 * head_dim]
        queries, keys, values = qkv.chunk(3, dim=-1)    #[batch_size, num_heads, seq_length, head_dim]

        # TODO: rotate the queries and keys using RoPE

        # pad the keys and values
        keys_padded = F.pad(input=keys, pad=(0, 0, padding, padding), mode="constant", value=0) # [batch_size, num_heads, seq_length + 2 x padding, head_dim]
        values_padded = F.pad(input=values, pad=(0, 0, padding, padding), mode="constant", value=0)

        # Create sliding windows for keys and values
        keys_windows = keys_padded.unfold(dimension=2, size=self.window_size, step=1)   # [batch_size, num_heads, seq_length, window_size, head_dim]
        values_windows = values_padded.unfold(dimension=2, size=self.window_size, step=1)  

        # Compute attention scores
        scores = torch.einsum('bnswh,bnsh->bn')
        # TODO: multiply attentions to values_windows
        # TODO: Merge heads and combine the last two dimensions
        # TODO: perform the final linear transformation
        output = None
        return output
   
    
class SlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Compute Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Pad sequence for windowed attention
        keys = F.pad(input=keys, pad=(0, 0, padding, padding), mode="constant", value=0)
        values = F.pad(input=values, pad=(0, 0, padding, padding), mode="constant", value=0)

        # Initialize context tensors
        context = torch.zeros_like(queries, device=x.device)

        # Compute attention for each sliding window
        for i in range(seq_length):
            # Determine the start and end of the window
            start = i
            end = i + self.window_size
            
            # Compute scores
            scores = torch.matmul(queries[:, :, i:i+1, :], keys[:, :, start:end, :].transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)
            
            # Apply attention to values and add to context
            context[:, :, i:i+1, :] += torch.matmul(attention, values[:, :, start:end, :])

        # Reshape context to (batch_size, seq_length, num_heads * head_dim)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)

        # Final linear layer
        output = self.out(context)
        return output