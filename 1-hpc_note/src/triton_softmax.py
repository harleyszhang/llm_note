import torch,time

import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def triton_softmax(X_ptr, Y_ptr, M, N, BLOCK_SIZE):
    pid = tl.program_id(0)                        # 获取当前块的 ID
    block_start = pid * BLOCK_SIZE                # 计算当前块的起始索引
    offsets = tl.arange(0, BLOCK_SIZE)            # 生成当前块的线程偏移量
    idx = block_start + offsets                   # 计算每个线程负责的索引
    mask = idx < M                                # 创建掩码，防止越界
    
    # 加载行数据
    x_row = tl.load(X_ptr + idx*N, mask = mask)   # 假设行连续存储
    x_max = tl.max(x_row)
    x_shifted = x_row - x_max
    exp_x = tl.exp(x_shifted)
    sum_x = tl.sum(exp_x)
    
    softmax_ret = exp_x / sum_x
    tl.store(Y_ptr + idx * N, softmax_ret, mask=mask)

def softmax_triton(X):
    M, N = X.shape
    Y = torch.empty_like(X[:,])
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)
    triton_softmax[grid](X, Y, M, N, BLOCK_SIZE=1024)
    return Y

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    """
    x_max = x.max(dim=1)[0] # read  MN elements ; write M  elements
    safe_x = x - x_max[:, None] # read MN + M elements ; write MN elements
    numerator = torch.exp(safe_x) # read  MN elements ; write MN elements
    denominator = numerator.sum(dim=1) # read  MN elements ; write M  elements
    ret = numerator / denominator[:, None]  # read MN + M elements ; write MN elements
    
    return ret

if __name__ == "__main__":
    M, N = 80000, 1000  # M rows, N columns
    X = torch.randn(M, N, device='cuda', dtype=torch.float32)

    # GPU 预热
    for _ in range(10):
        Y_triton = triton_softmax(X)
        Y_naive = naive_softmax(X)

    # Triton softmax 时间
    start_time = time.time()
    Y_triton = triton_softmax(X)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time

    # Naive softmax 时间
    start_time = time.time()
    Y_naive = naive_softmax(X)
    torch.cuda.synchronize()
    naive_time = time.time() - start_time

    # PyTorch softmax 时间
    start_time = time.time()
    Y_pytorch = torch.softmax(X, dim=1)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    # 验证结果
    if torch.allclose(Y_triton, Y_pytorch, atol=1e-4):
        print("Triton optimized softmax 与 PyTorch softmax 结果一致！")
    else:
        print("Triton optimized softmax 与 PyTorch softmax 结果不一致！")

    if torch.allclose(Y_naive, Y_pytorch, atol=1e-4):
        print("naive_softmax 与 PyTorch softmax 结果一致！")
    else:
        print("naive_softmax 与 PyTorch softmax 结果不一致！")

    # 输出时间
    print(f"Triton optimized softmax 时间: {triton_time * 1000:.2f} ms")
    print(f"naive_softmax 时间: {naive_time * 1000:.2f} ms")
    print(f"PyTorch softmax 时间: {pytorch_time * 1000:.2f} ms")