import torch
import torch.nn as nn


class SiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # create 2 linear layers
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
  
    def forward(self, x):
        # implement SiGLU W * x * sigma (W_g * x)
        return self.linear1(x) * torch.sigmoid(self.linear2(x))