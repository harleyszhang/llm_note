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
