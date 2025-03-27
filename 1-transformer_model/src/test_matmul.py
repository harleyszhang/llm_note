import torch
import torch.nn as nn

def exec_out(self, input_x, output_y=None):
    """
    对输入张量执行矩阵乘法，支持批量广播，并将结果存储到 output_y 中。
    若未传入 output_y，则根据输入张量自动创建新的输出张量。

    参数:
        input_x (list[torch.Tensor]): 包含两个张量，分别用于矩阵乘法的左操作数和右操作数。
        output_y (torch.Tensor, 可选): 存储结果的输出张量。

    返回:
        torch.Tensor: 矩阵乘法结果存储于 output_y 中。
    """
    # Step 1: 检查是否均为张量
    if all(torch.is_tensor(t) for t in input_x):
        print("DEBUG: 两个输入均为 tensor")
        # Step 2: 计算批处理广播形状（除去最后两个矩阵维度）
        batch_shape = torch.broadcast_shapes(*(t.shape[:-2] for t in input_x[:2]))
        print(f"DEBUG: 批处理 shape = {batch_shape}")
        
        # Step 3: 获取矩阵乘法相关的两个维度
        m = input_x[0].shape[-2]
        n = input_x[1].shape[-1]
        print(f"DEBUG: m (左张量倒数第二维) = {m}, n (右张量最后一维) = {n}")
        
        # Step 4: 最终输出 shape = 批处理 shape + [m, n]
        shape = list(batch_shape) + [m, n]
        print(f"DEBUG: 最终输出 shape = {shape}")
    else:
        shape = list(input_x[0].shape)
        print("DEBUG: 输入中至少有一个不是 tensor，直接使用 input_x[0] 的 shape")
    
    # Step 5: 如果未提供 output_y，则创建一个空张量用于存储结果
    if output_y is None:
        output_y = input_x[0].new_empty(shape)
        print(f"DEBUG: 创建新的 output_y, shape = {output_y.shape}")
    else:
        print(f"DEBUG: 使用已有的 output_y, shape = {output_y.shape}")
    
    # Step 6: 执行矩阵乘法，并将结果写入 output_y
    torch.matmul(input_x[0], input_x[1], out=output_y)
    print("DEBUG: 执行 torch.matmul 完成")
    
    # 返回结果
    return output_y

# ----- 以下为模拟调试过程 -----
if __name__ == '__main__':
    # 构造测试用的输入张量
    tensor_a = torch.randn(2, 3, 4, 5)
    tensor_b = torch.randn(2, 3, 5, 6)
    input_x = [tensor_a, tensor_b]

    # 假设我们在某个类中调用该方法，这里用 None 模拟 self
    # 直接调用函数，观察打印的 debug 信息
    result = exec_out(None, input_x)
    print(f"DEBUG: 最终结果的 shape = {result.shape}")
    nn.RMSNorm()
