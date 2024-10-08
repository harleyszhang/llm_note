import torch.nn as nn
import torch

import torch
import torch.nn as nn

class FlexibleLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        初始化 FlexibleLayerNorm 模块。

        参数: 
            normalized_shape (int or list or tuple): 用于归一化的形状。
                - 对于 NLP: 通常为最后一个维度，如 hidden_size。
                - 对于 CV: 可以是 channel 维度或 channel + spatial 维度。
            eps (float): 一个小常数，防止除零。
            elementwise_affine (bool): 是否学习可缩放和平移参数。
        """
        super(FlexibleLayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        """前向传播。
        参数: 
            x (Tensor): 输入张量。
        返回: 
            Tensor: 归一化后的张量。
        """
        # 确定归一化的维度
        if len(self.normalized_shape) == 1:
            # 例如，NLP 中的最后一个维度
            normalized_axes = (-1,)
        else:
            # 例如，CV 中的 channel 或 channel + spatial 维度
            normalized_axes = tuple(range(-len(self.normalized_shape), 0))

        # 计算均值和方差
        mean = x.mean(dim=normalized_axes, keepdim=True)
        var = x.var(dim=normalized_axes, keepdim=True, unbiased=False)

        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习参数
        if self.elementwise_affine:
            # 通过广播将 gamma 和 beta 扩展到输入张量的形状
            x_normalized = x_normalized * self.gamma + self.beta

        return x_normalized

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    

class LayerNorm(nn.Module):
    """nlp 领域"""
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # '-1' means last dimension. 
        var = x.var(-1, keepdim=True)
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        
        return out

# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)

# 1，Activate nn.LayerNorm module
layer_norm1 = nn.LayerNorm(embedding_dim)
pytorch_ln_out = layer_norm1(embedding)

# 2，Activate my nn.LayerNorm module
layer_norm2 = LayerNorm(embedding_dim)
my_ln_out = layer_norm2(embedding)

# 比较结果
print(torch.allclose(pytorch_ln_out, my_ln_out, rtol=0.1,atol=0.01))  # 输出 True