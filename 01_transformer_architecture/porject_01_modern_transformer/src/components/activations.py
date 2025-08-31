import torch
import torch.nn as nn


class SiGLU(nn.Module):
    """
    Sigmoid Gated Linear Unit (SiGLU) Activation Function
    SiGLU(x) = W * x * sigma(W_g * x) where W and W_g are learnable weight matrices,  
    bias terms are omitted (for simplicity) and sigma is the sigmoid function.
    Gated activation functions tend to boost the learning quality but worsen the stability of 
    the training, especially with the Mixture of Experts (MoE) layers.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """    
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # create 2 linear layers
        self.w1 = nn.Linear(in_features, out_features, bias=False)
        self.w2 = nn.Linear(in_features, out_features, bias=False)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiGLU forward pass

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, in_features)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, seq_length, out_features)
        """        
        # implement SiGLU W * x * sigma (W_g * x)
        return self.w1(x) * torch.sigmoid(self.w2(x))