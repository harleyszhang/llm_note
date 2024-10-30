## 量化范围和粒度

### 量化范围

量化范围在论文中已经指出来了，`decode layer` 中的量化 `layer` 如下：

- `self-attention` 中的: `q_liner`、`k_liner`、`v_liner` 以及 $QK^T$ 对应的 `BMM`（批量矩阵乘法），以及注意力输出线性层；
- `mlp` 中的全部线性层以及激活层。

到这里我们会发现，transformer 中没有被量化的只有 `Token Bmbedding` 层、`LayerNorm` 层（llama 中是 RMSNorm 层），我个人推测之所以不量化 Token Bmbedding 层，是因为其参数冗余性较小且不存在权重稀疏现象，这个通过可视化 Token Bmbedding 层的权重值统计分布可以观测得到。

<div align="center">
<img src="../images/smoothquant/smoothquant_in_llm.png" width="55%" alt="smoothquant_compute_process">
</div>

### 量化粒度

论文中提到 `SmoothQuant` 一共提供了 `3` 种量化量化粒度，从细到粗分别是：`per-channel`、`per-token`、`per-tensor` 量化，不同的粒度本质上是指**基于不同的粒度去计算量化缩放系数**。它们的定义描述如下:
1. `per-tensor` 量化: 为整个张量使用一个统一的缩放系数（scale）和零点（zero point）;
2. `per-token` 量化：同一 `token`（如序列中的每个单词或子词）才使用统一的缩放系数和零点，量化后缩放系数有 `[batch_size * seq_len]` 个；
3. `per-channel` 量化：同一 `embeeding` 维度的才使用相同的缩放系数和零点，缩放系数总共 `[hidden_size]` 个；对于 CNN 模型，则是为张量的每个通道（例如，卷积层的每个输出通道或线性层的每个输出特征）使用独立的缩放因子和零点。

权重 weight 支持`per-channel`、`per-tensor` 量化，其中获取权重张量最大值的区别如下所示：

```python
import torch
import torch.nn as nn

fc1 = nn.Linear(512, 512,) # shape is [h, h]
w1 = fc1.weight

max1 = w1.abs().max() # 1, per-tensor compute max, max1 is a Scalar
max2 = w1.abs().max(dim=-1, keepdim=True)[0] # 2, per-channel computex, max2 shape is [h, ]
print(f"per-tensor max value is {max1}, per-channel max shape is {max2.shape}")

"""
per-tensor max value is 0.04419414699077606, per-channel max shape is torch.Size([512, 1])
"""
```

激活值 支持`per-token`、`per-tensor` 量化，其中获取权重张量最大值的区别如下所示：

```python
input = torch.randn([16, 64, 512]) # input shape is [batch_size, seq_len, hidden_dim]
input.view(-1, input.shape[-1])

input_max1 = input.abs().max() # input_max1 is Scalar
input_max2 = input.abs().max(dim = -1, keepdim=True)[0] # input_max1 is [batch_size * seq_len, ]
print(f"activation per-tensor max value is {input_max1}, activation per-channel max shape is {input_max2.shape}")

"""
activation per-tensor max value is 4.675, activation per-channel max shape is torch.Size([16, 64, 1])
"""
```

很明显，激活基于 `per-token` 量化去算最大值，其输出维度就是 `[batch_size, seq_len]`, 即保留 `tokens` 维度。这里假设输入是 2D 张量来分析 per-token 和 per-channel 计算的区别，重点在于理解沿着哪个维度做计算，示例代码如下所示：

```python
>>> act
tensor([[4, 5, 5, 6, 6, 6, 6, 6, 8, 9],
        [2, 2, 2, 2, 2, 7, 8, 3, 4, 2],
        [4, 5, 6, 6, 7, 8, 1, 6, 8, 2]])
>>> max = torch.tensor([[2], [2], [2]])
>>> act /max
tensor([[2.0000, 2.5000, 2.5000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 4.0000,
         4.5000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 3.5000, 4.0000, 1.5000, 2.0000,
         1.0000],
        [2.0000, 2.5000, 3.0000, 3.0000, 3.5000, 4.0000, 0.5000, 3.0000, 4.0000,
         1.0000]])

>>> weight
tensor([[4, 5, 5, 6, 6, 6, 6, 6, 8, 9],
        [2, 2, 2, 2, 2, 7, 8, 3, 4, 2],
        [4, 5, 6, 6, 7, 8, 1, 6, 8, 2]])
>>> scale = torch.tensor([2,2,2,3,4,5,6,7,8,9])
>>> weight / scale
tensor([[2.0000, 2.5000, 2.5000, 2.0000, 1.5000, 1.2000, 1.0000, 0.8571, 1.0000,
         1.0000],
        [1.0000, 1.0000, 1.0000, 0.6667, 0.5000, 1.4000, 1.3333, 0.4286, 0.5000,
         0.2222],
        [2.0000, 2.5000, 3.0000, 2.0000, 1.7500, 1.6000, 0.1667, 0.8571, 1.0000,
         0.2222]])
```

再结合前面的量化公式：

$$X¯_{\text{INT8}} = \left\lfloor \frac{X_{\text{FP16}}}{\Delta} \right\rceil, \quad \Delta = \frac{\max(|X|)}{2^{N-1} - 1} \tag{1}$$

我们就可以实现权重和激活的量化算法，代码如下所示：

```python
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
    dequantized = torch.view(original_shape)

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
```

程序运行后输出结果如下所示:

```bash
tensor([[  51,  -85,  109, -127],
        [  -1,    1,   -1,    1]], dtype=torch.int8)
tensor([[0.0196, 0.0235, 0.0275, 0.0314]])

quant_w and scales shape is torch.Size([2, 4]), torch.Size([1, 4])
```

经测试 `per-channel`、`per-token`、`per-tensor` 量化函数均正确。

到这里，我们只是理解了三种量化方式的原理，以及实现过程，虽然这还是以前的量化知识，但这是理解 SmoothQuant 源码的先验知识。

## SmoothQuant 项目剖析

先快速看下整个项目的文件统计情况:

```bash
├── act_scales # 统计得到的激活值
│   ├── opt-13b.pt
│   └── ...
├── examples
│   ├── export_int8_model.py                 # realQunat -如何导出INT8模型
│   ├── generate_act_scales.py               # common-统计激活值
│   ├── smoothquant_opt_demo.ipynb           # fakeQunat -基于 torch 的伪量化
│   └── smoothquant_opt_real_int8_demo.ipynb # realQuant -torch+CUTLASS 量化的过程
├── figures # SQ量化方案与效果图
│   ├── accuracy.png
│   └── ...
├── LICENSE
├── README.md
├── setup.py
└── smoothquant 
    ├── calibration.py # common - 通过校准集，统计得到 act_scales，其实是权重和激活的最大值用于计算 s
    ├── fake_quant.py  # fakeQunat  - 伪量化操作
    ├── __init__.py
    ├── opt.py         # realQunat  - 量化 opt 模型
    └── smooth.py      # common - 模型平滑
```


## 参考资料

- [[源码] [万字] SmoothQuant量化深入探究](https://zhuanlan.zhihu.com/p/701436876)
- https://github.com/mit-han-lab/smoothquant/