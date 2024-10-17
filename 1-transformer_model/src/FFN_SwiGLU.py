# -*- coding  : utf-8 -*-
# Author: honggao.zhang

import torch
import torch.nn as nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class FFNSwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.fc3 = nn.Linear(input_dim, hidden_dim, bias=False) 
        
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x) * self.fc3(x)))
    
layer = FFNSwiGLU(128, 256)
x = torch.randn(1, 128)
out = layer(x)
print(out.shape) # torch.Size([1, 128])
