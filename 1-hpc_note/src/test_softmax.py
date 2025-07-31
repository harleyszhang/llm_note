import numpy as np

L = 64
Dim = 16
BLK = 4

# Q[L, Dim]
# K[L, Dim]
# V[L, Dim]

MIN_M = -10000


def np_softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x


def normal_softmax(x):
    out = np.array(x)
    for r in range(0, L):
        maxi = 0
        for i in range(0, L):
            maxi = max(maxi, x[r, i])
        e_sum = 0
        for i in range(0, L):
            e_sum += np.exp(x[r, i] - maxi)
        for i in range(0, L):
            out[r, i] = np.exp(x[r, i] - maxi) / e_sum
    return out

a = np.random.uniform(-1, 1, [L, L]).astype("float32")

np_res = np_softmax(a)
# print(np_res)

normal_res = normal_softmax(a)
# print(normal_res)
np.testing.assert_allclose(np_res, normal_res, 1e-6, 1e-6)

import torch

# 创建一个张量
tensor = torch.randn(8192, dtype = torch.float16)

# 获取张量中的元素总数
num_elements = tensor.numel()

# 获取每个元素的字节大小
element_size = tensor.element_size()

# 计算总存储空间大小（字节）
total_size_in_bytes = num_elements * element_size

print(f"张量的元素总数: {num_elements}")
print(f"每个元素的字节大小: {element_size} bytes")
print(f"张量的存储空间大小: {total_size_in_bytes} bytes") # 单位 B
print(f"张量的存储空间大小: {total_size_in_bytes/1024} KB")
