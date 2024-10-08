import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        """
        :param dim: 输入的维度
        :param eps: 防止除以0的稳定项
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数
    
    def forward(self, x):
        # x 的形状为 [batch_size, seq_len, dim]
        
        # 计算均方根 (RMS)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化，并应用缩放参数
        return x / rms * self.scale

# 测试 RMSNorm
batch_size, seq_len, dim = 2, 4, 8
x = torch.randn(batch_size, seq_len, dim)

rmsnorm = RMSNorm(dim)
output = rmsnorm(x)

# nn.RMSNorm 如果传入的是单个整数，则会将其视为一个单元素列表，
# 模块会对最后一个维度进行归一化，并且该维度的大小应该符合这个指定值。
rmsnorm_pytorch = nn.RMSNorm(dim)
output_pytorch = rmsnorm_pytorch(x)
    
print("输入:\n", x)
print("输出:\n", output)
print("输入和输出的形状: ", x.shape, output.shape)

if torch.allclose(output, output_pytorch, atol=1e-6):
    print("结果验证通过: 自己实的 RMSNorm 和 pytorch nn.RMSNorm 结果一致！")
else:
    print("结果验证失败: 自己实的 RMSNorm 和 pytorch nn.RMSNorm 结果不一致。")