- [1，基础知识](#1基础知识)
- [2. 向量相加](#2-向量相加)
- [3. 矩阵乘法](#3-矩阵乘法)

`Triton` 是一种用于并行编程的语言和编译器，它旨在提供一个基于 Python 的编程环境，帮助高效编写自定义的深度神经网络（DNN）计算核，并在现代 GPU 硬件上以最大吞吐量运行。

Triton 的编程在语法和语义上与 `Numpy` 和 `PyTorch` 非常相似。然而，作为低级语言，开发者需要注意很多细节，特别是在内存加载和存储方面，这对于在低级设备上实现高速计算非常关键。

### 1，基础知识

- `tl.load`：用于于从由指针定义的内存位置加载数据。
- `tl.store`：用于将张量的数据写入由指针定义的内存位置。
- `tl.program_id(axis)`：返回当前程序实例在指定轴上的 `ID`。axis 是一个常量，指定你想要查询的轴。
- `tl.arange`：在半开区间 `[start, end)` 内返回连续值，用于生成从 $0$ 开始的偏移量。

元数据就是描述数据本身的数据，元类就是类的类，相应的元编程就是描述代码本身的代码，元编程就是关于创建操作源代码(比如修改、生成或包装原来的代码)的函数和类。主要技术是使用装饰器、元类、描述符类。

META 是一个常用的变量名，通常用于表示“元数据”（metadata）。在 Triton 中，META 通常是一个字典，用于传递配置参数给内核（kernel）。它可以包含多个键值对，每个键对应一个特定的配置参数。例如：
```python
META = {
    'BLOCK_SIZE': 128,  # 每个块的大小
    'ANOTHER_PARAM': 42,  # 其他参数
    # 其他配置参数...
}
```

`META['BLOCK_SIZE']` 表示每个块（block）的大小，这个值很重要，因为它直接影响到内核的并行性和性能。

在 Triton 中，**块（block）是内核并行执行的基本单位**（与 cuda 编程把 thread 当作并行执行的基本单位不同）。每个块负责处理任务的一个子集，通过合理划分块大小，可以充分利用 GPU 的并行计算能力。BLOCK_SIZE 通常是由用户定义的，而不是由 Triton 自动从硬件中推断出来的。这是因为最佳的块大小取决于具体的应用场景、数据特性以及目标硬件的架构（如 GPU 的核数、共享内存大小等）。

### 2. 向量相加

一维向量相加是学习 Triton 编程模型中入门实例，代码如下所示，看不懂不要紧，有个大概映像和知道逻辑流程即可，下面会一步步分析。

```python
import torch
import triton
import triton.language as tl
import time

# 1，内核定义
@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                        # 获取当前块的 ID
    block_start = pid * BLOCK_SIZE                # 计算当前块的起始索引
    offsets = tl.arange(0, BLOCK_SIZE)            # 生成当前块的线程偏移量
    idx = block_start + offsets                   # 计算每个线程负责的索引
    mask = idx < N                                # 创建掩码，防止越界

    x = tl.load(X_ptr + idx, mask=mask)           # 加载 X 的值
    y = tl.load(Y_ptr + idx, mask=mask)           # 加载 Y 的值
    z = x + y                                     # 执行加法

    tl.store(Z_ptr + idx, z, mask=mask)           # 存储结果

# 2，内核调用
def vector_add_triton(X, Y):
    assert X.shape == Y.shape, "输入张量形状必须相同"
    N = X.numel()
    Z = torch.empty_like(X)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    vector_add_kernel[grid](X, Y, Z, N, BLOCK_SIZE=1024)

    return Z

# 3，主函数
if __name__ == "__main__":
    N = 10_000_000
    X = torch.randn(N, device='cuda', dtype=torch.float32)
    Y = torch.randn(N, device='cuda', dtype=torch.float32)

    # GPU 预热
    for _ in range(10):
        Z_triton = vector_add_triton(X, Y)

    # Triton 向量加法时间
    start_time = time.time()
    Z_triton = vector_add_triton(X, Y)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time

    # PyTorch 向量加法时间
    start_time = time.time()
    Z_pytorch = X + Y
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    # 验证结果
    if torch.allclose(Z_triton, Z_pytorch):
        print("Triton 向量加法成功！")
    else:
        print("Triton 向量加法失败！")

    # 输出时间
    print(f"Triton 向量加法时间: {triton_time * 1000:.2f} ms")
    print(f"PyTorch 向量加法时间: {pytorch_time * 1000:.2f} ms")
```

总的来说，代码分为三个部分，依次是：
1. 内核定义
2. 内核调用
3. main 主函数

1，先看内核调用函数 `vector_add_triton`。

Triton 内核的调用类似于 CUDA 的内核调用，但具有更高的抽象和简化的语法。

`grid` 定义了内核（kernel）执行的网格大小，即有多少个块（`blocks`）将被启动来执行内核，同时每个块包含 'BLOCK_SIZE' 个线程（`threads`）。
```python
grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
```

- `lambda META: (...)`：使用匿名函数（lambda）来**动态计算网格的大小**，而不需要在定义 `grid` 时硬编码具体的值。
- `triton.cdiv`：执行向上取整的整数除法（ceiling division）。具体来说，triton.cdiv(a, b) 计算 (a + b - 1) // b，确保结果向上取整。
- `N`：通常代表任务的总规模或元素数量。在向量加法的例子中，N 是向量的长度。
- `META['BLOCK_SIZE']`：META 元数据，根据 BLOCK_SIZE 变量来灵活创建变量 META['BLOCK_SIZE']。

理解了 `grid` 定义就能轻松理解内核调用了。
```python
# XYZ 参数分别代表输入和输出张量的数据地址，N 代表元素数量，BLOCK_SIZE 代表块大小
vector_add_kernel[grid](X, Y, Z, N, BLOCK_SIZE=1024) # 调用 Triton 内核，传递参数。
```

2，再看内核定义函数 `vector_add_kernel`，其函数原型如下所示。

```python
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):

    # 1，定义每个线程在全局数据中的具体索引
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets # 有时直接写 block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # 2，加载数据，并执行内核算法（向量加法）
    x = tl.load(X_ptr + idx, mask=mask)
    y = tl.load(Y_ptr + idx, mask=mask)
    
    # 3，执行向量加法(内核算法)
    z = x + y                                     
    
    # 4，存储结果
    tl.store(Z_ptr + idx, z, mask=mask)  
```

内核中的 `pid`、`block_start`、`offsets` 和 `idx` 这四个概念，只有正确理解了这四个（三个）概念，才能写出正确、高效的内核代码。

1. `pid`（Program ID）: 当前块（Block）的唯一标识符，代表块在整个网格（Grid）中的位置（第几个块）。`pid = tl.program_id(0)`。
2. `block_start`: 当前块在全局数据中的起始位置索引，用于确保每个块处理的数据范围不重叠且覆盖整个数据集。`block_start = pid * BLOCK_SIZE`。
3. `offsets`: 表示当前块内每个线程相对于块起始位置的偏移量，帮助每个线程计算其在全局数据中的具体索引。`offsets = tl.arange(0, BLOCK_SIZE)`。
4. `idx`: **表示每个线程在全局数据中的具体索引，用于加载和存储数据，确保每个线程处理唯一的数据元素**。`idx = block_start + offsets`。
Triton 内核调用流程：
另外，`mask = idx < N` 的作用是创建掩码，防止线程访问超出数据范围的元素

为了更好的理解上述 4 个变量的关系和值意义，可通过一个实例。假设我们有一个向量长度为 N = 10，BLOCK_SIZE = 4，则内核执行如下：

| Block ID (pid) | block_start | offsets | idx (global index) |
| --- | --- | --- | --- |
| 0 | 0*4=0 | [0,1,2,3] | [0,1,2,3] |
| 1 | 1*4=4 | [0,1,2,3] | [4,5,6,7] |
| 2 | 2*4=8 | [0,1,2,3] | [8,9,10,11] (mask applied for N=10) |

知道如何计算 `idx` 和 `mask` 值，就会知道如何加载和存储数据并执行相应算法操作。
```python
 # 加载数据
x = tl.load(X_ptr + idx, mask=mask)           # 加载 X 的值
y = tl.load(Y_ptr + idx, mask=mask)           # 加载 Y 的值
# 执行向量加法(内核算法)
z = x + y                                     # 执行加法
# 存储结果
tl.store(Z_ptr + idx, z, mask=mask)           # 存储结果到 Z
```

3，最后就是 main 函数了，主要是：初始化张量、GPU 预热、执行 Triton 向量加法并记录时间、执行 PyTorch 向量加法并记录时间和验证结果并输出时间。这里的代码没什么好讲的，都是 pytorch 代码，记住下这个流程即可。

### 3. 矩阵乘法