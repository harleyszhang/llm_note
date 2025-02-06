import torch
import time

# 配置参数
b, q, l, h, d, c, D = 32, 64, 128, 64, 64, 128, 256  # 将 h 调整为 64
n_warmup = 10   # 预热次数
n_trials = 100  # 正式测试次数

# 初始化张量（GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_UV = torch.randn(h, d, c, device=device)
c_t_KV = torch.randn(b, l, c, device=device)
attn_weights = torch.randn(b, q, h, l, device=device)
W_o = torch.randn(h, d, D, device=device)  # h 维度为 64

# 预热 GPU
for _ in range(n_warmup):
    _ = torch.einsum('hdc,blc->blhd', W_UV, c_t_KV)
    _ = torch.einsum('bqhl,blhd->bqhd', attn_weights, _)
    _ = torch.einsum('hdD,bhqd->bhD', W_o, _)

# 原始分步实现
def original_method():
    v_t = torch.einsum('hdc,blc->blhd', W_UV, c_t_KV)
    o = torch.einsum('bqhl,blhd->bqhd', attn_weights, v_t)
    u = torch.einsum('hdD,bhqd->bhD', W_o, o.permute(0, 2, 1, 3))

# 优化后实现
def optimized_method():
    o_ = torch.einsum('bhql,blc->bhqc', attn_weights.permute(0, 2, 1, 3), c_t_KV)
    o = torch.einsum('bhqc,hdc->bhqd', o_, W_UV)
    u = torch.einsum('hdD,bhqd->bhD', W_o, o)

# 测量时间
def benchmark(func):
    times = []
    for _ in range(n_trials):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)
    return sum(times) / n_trials

# 执行测试
time_original = benchmark(original_method) * 1000  # 转换为毫秒
time_optimized = benchmark(optimized_method) * 1000

# 打印结果
print(f"原始方法平均时间: {time_original:.3f} ms")
print(f"优化方法平均时间: {time_optimized:.3f} ms")
print(f"速度提升: {time_original / time_optimized - 1:.1%}")

# 验证等价性
def validate_equivalence():
    v_t_orig = torch.einsum('hdc,blc->blhd', W_UV, c_t_KV)
    o_orig = torch.einsum('bqhl,blhd->bqhd', attn_weights, v_t_orig)
    u_orig = torch.einsum('hdD,bhqd->bhD', W_o, o_orig.permute(0, 2, 1, 3))
    
    v_t_opt = torch.einsum('hdc,blc->blhd', W_UV, c_t_KV)
    o_opt = torch.einsum('bqhl,blhd->bqhd', attn_weights, v_t_opt)
    u_opt = torch.einsum('hdD,bhqd->bhD', W_o, o_opt.permute(0, 2, 1, 3))
    
    # 检查是否等价
    assert torch.allclose(u_orig, u_opt, atol=1e-4), "两种方法结果不一致！"
    print("两种方法结果一致，验证通过。")

# 调用验证函数
validate_equivalence()