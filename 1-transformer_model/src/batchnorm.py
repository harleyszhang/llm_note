import torch
from torch import nn
import time

# 定义修正后的 FlexibleBatchNorm 类
class FlexibleBatchNorm(nn.Module):
    def __init__(self, num_features, dim, eps=1e-5, momentum=0.1, affine=True):
        """
        初始化 FlexibleBatchNorm 模块。

        参数：
            num_features (int): 归一化的特征数（例如，channels）。
            dim (int): 输入张量的维度（例如，2、3、4）。
            eps (float): 一个小常数，防止除零。
            momentum (float): 运行均值和方差的更新动量。
            affine (bool): 是否学习可缩放和平移参数。
        """
        super(FlexibleBatchNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.dim == 2:
            # 2D 张量: (batch_size, channels), [N, C]
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            shape = (1, -1)
        elif self.dim == 3:
            # 3D 张量: (batch_size, channels, length), [N, C, L]
            mean = x.mean(dim=(0, 2))
            var = x.var(dim=(0, 2), unbiased=False)
            shape = (1, -1, 1)
        elif self.dim == 4:
            # 4D 张量: (batch_size, channels, height, width), [N, C, H, W]
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            shape = (1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {self.dim}")

        if self.training:
            # 更新运行均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            running_mean = mean
            running_var = var
        else:
            # 使用运行均值和方差
            running_mean = self.running_mean
            running_var = self.running_var

        # 归一化
        x_normalized = (x - running_mean.view(*shape)) / torch.sqrt(running_var.view(*shape) + self.eps)

        # 应用可学习参数
        if self.affine:
            # 根据归一化维度调整 gamma 和 beta 的形状
            gamma = self.gamma.view(*shape)
            beta = self.beta.view(*shape)
            x_normalized = x_normalized * gamma + beta

        return x_normalized

    def extra_repr(self):
        return f'num_features={self.running_mean.size(0)}, dim={self.dim}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}'

# 定义一致性测试函数
def test_batchnorm_consistency():
    # 定义参数
    batch_size = 8
    features = 16
    sequence_length = 10
    height = 28
    width = 28

    # 2D BatchNorm
    x_fc = torch.randn(batch_size, features)
    batch_norm_fc_custom = FlexibleBatchNorm(num_features=features, dim=2, affine=True)
    batch_norm_fc_pytorch = nn.BatchNorm1d(num_features=features, affine=True)

    # 同步参数
    batch_norm_fc_pytorch.weight.data = batch_norm_fc_custom.gamma.data.clone()
    batch_norm_fc_pytorch.bias.data = batch_norm_fc_custom.beta.data.clone()

    # 设置为训练模式
    batch_norm_fc_custom.train()
    batch_norm_fc_pytorch.train()

    # 前向传播
    y_fc_custom = batch_norm_fc_custom(x_fc)
    y_fc_pytorch = batch_norm_fc_pytorch(x_fc)

    # 比较结果
    assert torch.allclose(y_fc_custom, y_fc_pytorch, atol=1e-5), "2D BatchNorm results do not match!"
    print("2D BatchNorm consistency test passed.")

    # 3D BatchNorm [N, C, L]
    # where :math:`N` is the batch size,:math:`C` is the number of features or channels, and :math:`L` is the sequence length
    x_rnn = torch.randn(batch_size, features, sequence_length)
    batch_norm_rnn_custom = FlexibleBatchNorm(num_features=features, dim=3, affine=True)
    batch_norm_rnn_pytorch = nn.BatchNorm1d(num_features=features, affine=True)  # 使用 BatchNorm1d 归一化 features

    # 同步参数
    batch_norm_rnn_pytorch.weight.data = batch_norm_rnn_custom.gamma.data.clone()
    batch_norm_rnn_pytorch.bias.data = batch_norm_rnn_custom.beta.data.clone()

    # 设置为训练模式
    batch_norm_rnn_custom.train()
    batch_norm_rnn_pytorch.train()

    # 前向传播
    y_rnn_custom = batch_norm_rnn_custom(x_rnn)
    # 使用 BatchNorm1d 处理 3D 张量，需调整维度
    y_rnn_pytorch = batch_norm_rnn_pytorch(x_rnn)

    # 比较结果
    assert torch.allclose(y_rnn_custom, y_rnn_pytorch, atol=1e-5), "3D BatchNorm results do not match!"
    print("3D BatchNorm consistency test passed.")

    # 4D BatchNorm
    channels = 3
    x_conv = torch.randn(batch_size, channels, height, width)
    batch_norm_conv_custom = FlexibleBatchNorm(num_features=channels, dim=4, affine=True)
    batch_norm_conv_pytorch = nn.BatchNorm2d(num_features=channels, affine=True)

    # 同步参数
    batch_norm_conv_pytorch.weight.data = batch_norm_conv_custom.gamma.data.clone()
    batch_norm_conv_pytorch.bias.data = batch_norm_conv_custom.beta.data.clone()

    # 设置为训练模式
    batch_norm_conv_custom.train()
    batch_norm_conv_pytorch.train()

    # 前向传播
    y_conv_custom = batch_norm_conv_custom(x_conv)
    y_conv_pytorch = batch_norm_conv_pytorch(x_conv)

    # 比较结果
    assert torch.allclose(y_conv_custom, y_conv_pytorch, atol=1e-5), "4D BatchNorm results do not match!"
    print("4D BatchNorm consistency test passed.")

# 性能基准测试函数
def benchmark_batchnorm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 定义不同维度的输入张量
    shapes = [
        (128, 256),                # 2D BatchNorm1d, (N, C)
        (128, 64, 256),            # 3D BatchNorm1d, (N, C, L)
        (128, 64, 32, 32),         # 4D BatchNorm2d, (N, C, H, W)
    ]

    # 运行次数
    runs = 1000

    for shape in shapes:
        dim = len(shape)

        # 正确设置 num_features
        if dim == 2:
            num_features = shape[1]
        elif dim == 3:
            num_features = shape[1]
        elif dim == 4:
            num_features = shape[1]
        else:
            continue

        x = torch.randn(shape, device=device)

        # 定义自定义 BatchNorm
        batch_norm_custom = FlexibleBatchNorm(num_features=num_features, dim=dim).to(device)
        batch_norm_custom.train()

        # 定义 PyTorch BatchNorm
        if dim == 2:
            batch_norm_pytorch = nn.BatchNorm1d(num_features=num_features).to(device)
        elif dim == 3:
            batch_norm_pytorch = nn.BatchNorm1d(num_features=num_features).to(device)  # 3D 使用 BatchNorm1d 归一化 features
        elif dim == 4:
            batch_norm_pytorch = nn.BatchNorm2d(num_features=num_features).to(device)
        else:
            continue

        # 同步参数
        if batch_norm_custom.affine:
            batch_norm_pytorch.weight.data = batch_norm_custom.gamma.data.clone()
            batch_norm_pytorch.bias.data = batch_norm_custom.beta.data.clone()

        # 预热
        # with torch.no_grad():
        #     batch_norm_custom(x)
        #     batch_norm_pytorch(x)

        # 测试自定义 BatchNorm
        torch.cpu.synchronize()
        start = time.time()
        for _ in range(runs):
            y_custom = batch_norm_custom(x)
        torch.cpu.synchronize()
        custom_time = (time.time() - start) * 1000 / runs  # 平均时间(ms)

        # 测试 PyTorch BatchNorm
        torch.cpu.synchronize()
        start = time.time()
        for _ in range(runs):
            y_pytorch = batch_norm_pytorch(x)
        torch.cpu.synchronize()
        pytorch_time = (time.time() - start) * 1000 / runs  # 平均时间(ms)

        # 输出结果
        print(f"Shape: {shape}, Dimension: {dim}")
        print(f"Custom BatchNorm: {custom_time:.6f} ms per run")
        print(f"PyTorch BatchNorm: {pytorch_time:.6f} ms per run")
        print("-" * 50)

if __name__ == "__main__":
    print("Running BatchNorm Consistency Tests...")
    test_batchnorm_consistency()
    print("\nRunning BatchNorm Performance Benchmarks...")
    benchmark_batchnorm()