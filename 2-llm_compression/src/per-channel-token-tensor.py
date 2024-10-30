import torch
import torch.nn as nn

fc1 = nn.Linear(512, 512,) # shape is [h, h]
w1 = fc1.weight

weight_max1 = w1.abs().max() # 1, per-tensor compute max, max1 is a Scalar
weight_max2 = w1.abs().max(dim=-1, keepdim=True)[0] # 2, per-channel computex, max2 shape is [h, ]

print(f"weight per-tensor max value is {weight_max1}, weight per-channel max shape is {weight_max2.shape}")

"""
per-tensor max value is 0.0441, per-channel max shape is torch.Size([512, 1])
"""

input = torch.randn([16, 64, 512]) # input shape is [batch_size, seq_len, hidden_dim]
input.view(-1, input.shape[-1])

input_max1 = input.abs().max() # input_max1 is Scalar
input_max2 = input.abs().max(dim = -1, keepdim=True)[0] # input_max1 is [batch_size * seq_len, ]
print(f"activation per-tensor max value is {input_max1}, activation per-channel max shape is {input_max2.shape}")

"""
activation per-tensor max value is 4.675, activation per-channel max shape is torch.Size([16, 64, 1])
"""

#######################################权重和激活值的量化实现#############################################3
@torch.no_grad()
def quantize_weight_per_channel_absmax(weight, n_bits=8):
    """
    按每个 token 的绝对最大值进行激活量化
    参数:
        - weight (torch.Tensor): 权重张量, shape is (hidden_size, hidden_size).
        - n_bits (int): 量化位数, 默认 8
    返回:
        torch.Tensor: 量化后的激活张量
    """
    weight_max = weight.abs().max(dim = 0, keepdim = True)[0] # 逐 token 计算最大值
    q_max = pow(2, n_bits) - 1
    scales = weight_max.clamp(min=1e-5) / q_max
    quantized = (weight / scales).round().clamp(-q_max, q_max).to(torch.int8)

    return quantized, scales

@torch.no_grad()
def quantize_activation_per_token_absmax(act, n_bits=8):
    """
    按每个 token 的绝对最大值进行激活量化
    参数:
        - act (torch.Tensor): 激活张量, 本质上是线性层的输入张量, shape is (batch_size, seq_len, hidden_size).
        - n_bits (int): 量化位数, 默认 8
    返回:
        torch.Tensor: 量化后的激活张量
    """

    original_shape = act.shape
    act_reshaped = act.view(-1, original_shape[-1]) # 重塑激活张量为  (batch_size * seq_len, hidden_size), 方便逐 token 量化

    act_max = act.abs().max(dim = -1, keepdim = True)[0] # 逐 token 计算最大值
    q_max = pow(2, n_bits) - 1
    scales = act_max.clamp(min=1e-5) / q_max

    quantized = (act_reshaped / scales).round().clamp(-q_max, q_max).to(torch.int8)

    dequantized = quantized.float() * scales
    dequantized = dequantized.view(original_shape)

    return dequantized, scales

@torch.no_grad()
def quantize_activation_per_tensor_absmax(tensor, n_bits=8):
    """
    按激活整个张量的绝对最大值进行激活量化
    参数:
        - act (torch.Tensor): 激活张量, 本质上是线性层的输入张量, shape is (batch_size, seq_len, hidden_size).
        - n_bits (int): 量化位数, 默认 8
    返回:
        torch.Tensor: 量化后的激活张量
    """
    
    scale = tensor.abs().max()
    q_max = 2 ** (n_bits - 1) - 1  # 对于int8, q_max=127
    scale = max(scale.item(), 1e-5) / q_max
    quantized_tensor = (tensor / scale).round().clamp(-q_max, q_max).to(torch.int8)
    
    return quantized_tensor, scale


def test():
    # weight = torch.randn([32, 32])
    weight = torch.tensor([[1.0, -2.0, 3.0, -4.0],
                            [5.0, -6.0, 7.0, -8.0]], dtype=torch.float32)
    quant_w, scales = quantize_weight_per_channel_absmax(weight)

    print(quant_w)
    print(scales)
    print(f"\nquant_w and scales shape is {quant_w.shape}, {scales.shape}")
    
if __name__ == "__main__":
    test()