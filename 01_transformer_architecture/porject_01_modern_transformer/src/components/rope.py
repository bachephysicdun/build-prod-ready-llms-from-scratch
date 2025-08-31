import torch
import torch.nn as nn
from typing import Tuple


def get_rotation_matrix(dim: int, context_size: int, period: float) -> torch.Tensor:
    """
    Function to compute the Rotary Position Embedding (RoPE) rotation matrix.
    (In practice, this matrix is precomputed and registered as a buffer in the RoPE class.)

    Args:
        dim (int): dimension (i.e., head_dim) of the vectors to be partitioned into dim // 2 segments of pairs.
        context_size (int): maximum length of the context
        period (float): Predio of the RoPE, typically set to a large value greater than context_size, e.g., 10,000.

    Returns:
        torch.Tensor: rotation matrix of shape [context_size, dim // 2]
    """    
    # compute a tensor of frequencies
    freqs = 1.0 / (period ** (torch.arange(0, dim, 2).float() / dim))  # [dim // 2]
    
    # compute a tensor of token indexes
    token_indexes = torch.arange(context_size)
    
    # compute the matrix thetas
    # thetas = torch.einsum('i,j->ij', token_indexes, freqs)  # [context_size, dim // 2]    
    thetas = torch.outer(token_indexes, freqs)  # [context_size, dim // 2]
    
    # create the rotation matrix
    rotation_matrix = torch.polar(torch.ones_like(thetas), thetas)  # [context_size, dim // 2]
    
    return rotation_matrix


class RoPE(nn.Module):
    def __init__(self, rotation_matrix: torch.Tensor) -> None:
        """
        Rotary Position Embedding (RoPE) Layer.
        This module applies rotary position embeddings to the input queries and keys tensors used 
        in the self-attention mechanism. RoPE encodes positional information by rotating the query and key vectors
        in a way that depends on their position in the sequence. This allows the model to capture relative positional relationships.

        Args:
            rotation_matrix (torch.Tensor): precomputed rotation matrix of shape [context_size, head_dim // 2]
            This is fixed, non-learnable, so we register it as a buffer and will be used for position dependent rotation.
        """        
        super().__init__()
        # self.rotation_matrix = rotation_matrix  # [context_size, head_dim // 2]
        # the RoPE rotation matrix is fixed (non-learnable), so we register it as a buffer
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)  # [context_size, head_dim // 2]

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to the input queries and keys.

        Args:
            queries (torch.Tensor): query tensor of shape [batch_size, num_heads, seq_length, head_dim]
            keys (torch.Tensor): query tensor of shape [batch_size, num_heads, seq_length, head_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - new_queries (torch.Tensor): rotated query tensor of shape [batch_size, num_heads, seq_length, head_dim]
                - new_keys (torch.Tensor): rotated key tensor of shape [batch_size, num_heads, seq_length, head_dim]
        """        
        batch_size, num_heads, seq_length, head_dim = queries.size()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # reshape to [batch_size, num_heads, seq_length, head_dim // 2 , 2]
        queries = queries.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)
        keys = keys.reshape(batch_size, num_heads, seq_length, head_dim // 2, 2)

        # transform into a complex tensor
        queries_complex = torch.view_as_complex(queries) # [batch_size, num_heads, seq_length, head_dim // 2]
        keys_complex = torch.view_as_complex(keys)  # [batch_size, num_heads, seq_length, head_dim // 2]

        # rotate the queries and keys
        queries_rotated = queries_complex * self.rotation_matrix[:seq_length, :] # [batch_size, num_heads, seq_length, head_dim // 2]
        keys_rotated = keys_complex * self.rotation_matrix[:seq_length, :] # [batch_size, num_heads, seq_length, head_dim // 2]

        # convert to read and reshape back to [batch_size, num_heads, seq_length, head_dim]
        new_queries = torch.view_as_real(queries_rotated).reshape(batch_size, num_heads, seq_length, head_dim)
        new_keys = torch.view_as_real(keys_rotated).reshape(batch_size, num_heads, seq_length, head_dim)

        return new_queries, new_keys











def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)