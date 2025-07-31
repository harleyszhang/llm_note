import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

# 定义激活函数字典
ACT2FN = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
}

# 定义 DeepseekV2MLP 模块
class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # 例如 "silu"

    def forward(self, x):
        # 计算激活后的门控信号
        gate_out = self.act_fn(self.gate_proj(x))
        # 对输入进行 up 投影
        up_out = self.up_proj(x)
        # 元素级乘法后 down 投影，输出形状与输入相同
        mlp_out = self.down_proj(gate_out * up_out)
        return mlp_out

# 定义一个简单的配置类
class DummyConfig:
    def __init__(self, hidden_size=64, intermediate_size=128, hidden_act="silu"):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

# 单元测试类
class TestDeepseekV2MLP(unittest.TestCase):
    def setUp(self):
        # 创建配置对象
        self.config = DummyConfig(hidden_size=64, intermediate_size=128, hidden_act="silu")
        self.model = DeepseekV2MLP(self.config)
        # 模拟一个输入张量，形状为 [batch, seq_len, hidden_size]
        self.input_tensor = torch.randn(8, 10, self.config.hidden_size)

    def test_output_shape(self):
        # 测试输出形状是否与输入相同
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_forward_pass(self):
        # 测试前向传播不报错，并且输出数据类型正确
        output = self.model(self.input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, self.input_tensor.dtype)

    def test_gradients(self):
        # 测试反向传播是否能正常计算梯度
        output = self.model(self.input_tensor)
        loss = output.mean()
        loss.backward()
        # 检查模型参数是否获得梯度
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")

if __name__ == '__main__':
    unittest.main()