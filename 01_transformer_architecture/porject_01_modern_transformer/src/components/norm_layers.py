import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    RMSNorm is a simpler alternative to LaywerNorm that normalized hidden states using 
    root mean square of activations, witout centering them by their mean. It is computationally
    more efficient and performs well in large scale language models.

    Args:
        hidden_size (int): Dimension of hidden states.
        eps (float): A small value to avoid division by zero.
    """    
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # learnable weight parameter for scaling the normalized output
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply RMS normalization to the input tensor

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """        
        # implement the normalization
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm forward pass

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, hidden_size)
        """        
        # Do the math in float32 for safety (by x.float()), but return the result in the 
        # same dtype (by type_as(x)) as the input to keep training efficient 
        # (e.g., if the input was float16 (for efficiency), the output will also be float16.)
        output = self._norm(x.float()).type_as(x)
        
        # apply learbale scalling (per dimension)
        return output * self.weight