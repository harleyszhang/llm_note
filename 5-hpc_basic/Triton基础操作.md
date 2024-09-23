- [1，基础知识](#1基础知识)
  - [张量维度判断](#张量维度判断)
  - [Triton 基础函数](#triton-基础函数)
  - [网格、块和内核](#网格块和内核)
  - [cuda 执行模型](#cuda-执行模型)
- [2. 向量相加](#2-向量相加)
- [3. 理解内核运行的机制](#3-理解内核运行的机制)
- [3. 融合 Softmax](#3-融合-softmax)

`Triton` 是一种用于并行编程的语言和编译器，它旨在提供一个基于 Python 的编程环境，帮助高效编写自定义的深度神经网络（DNN）计算核，并在现代 GPU 硬件上以最大吞吐量运行。

Triton 的编程在语法和语义上与 `Numpy` 和 `PyTorch` 非常相似。然而，作为低级语言，开发者需要注意很多细节，特别是在内存加载和存储方面，这对于在低级设备上实现高速计算非常关键。

### 1，基础知识

#### 张量维度判断

在一个 $M$ 行 $N$ 列的二维数组中，$M$ 是第 0 维，即行数；$N$ 是第 1 维，即列数。那么怎么肉眼判断更复杂的张量数据维度呢，举例：
```python
import torch

# 示例张量
tensor = torch.tensor([[[0.6238, -0.9315, 0.2173, 0.1954, -1.1565],
                        [0.4559, 0.1531, 0.4178, 1.0225, 0.5923],
                        [0.0499, 0.4024, -1.2547, -0.5042, -0.0231],
                        [-1.1253, 0.3145, 0.8796, 0.4516, -0.0915]],

                       [[1.5794, -0.6367, -0.2559, 0.1237, -0.1951],
                        [0.1012, 0.0357, -0.5699, 1.0983, -0.2084],
                        [-0.7019, 0.5872, 0.7736, 0.7423, -0.7894],
                        [-0.3248, -0.5316, 1.2029, 0.2852, -0.4565]],

                       [[-0.0073, 1.4143, -0.1859, -0.7211, -0.8652],
                        [-0.3173, -0.4816, 0.1174, -0.1554, 0.9385],
                        [0.1283, -0.6547, 0.3687, -0.1948, 0.7754],
                        [-0.2185, -1.0437, 1.5963, -0.3284, -0.3654]]])
```
**判断规则**：方括号 `[` 的嵌套层数代表张量的维度。最外层括号的元素数量是第 0 维的大小，往内推。
以上述张量为例分析：
```python
tensor([[[ 0.6238, -0.9315,  0.2173,  0.1954, -1.1565], ... ]])
```
- 最外层 [ 里有 3 个子列表 -> 第 0 维大小为 3。
- 第二层 [ 里有 4 个子列表 -> 第 1 维大小为 4。
- 第三层 [ 里有 5 个元素 -> 第 2 维大小为 5。

因此，这个张量是 3 维张量，形状为 `[3, 4, 5]`。

#### Triton 基础函数

常用的 Triton 基础函数及其作用如下：
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

#### 网格、块和内核

**不同的 grid 则可以执行不同的程序（即 kernel）**。`grid` 定义了内核（kernel）执行的网格大小，即有多少个块（`blocks`）将被启动来执行一个内核，同时每个块包含 'BLOCK_SIZE' 个线程（`threads`），一个 block 中的 thread 能存取同一块共享的内存。

与 cuda 编程把 thread 当作并行执行的基本单位不同，在 Triton 中，**块 block 才是内核并行执行的基本单位**，每个块负责处理任务的一个子集，通过合理划分块大小，可以充分利用 GPU 的并行计算能力。

#### cuda 执行模型

在执行 CUDA 程序的时候，每个 SP（stream processor） 对应一个 thread，每个 SM（stream multiprocessor）对应一个 Block。

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
> Triton 内核的调用类似于 CUDA 的内核调用，但具有更高的抽象和简化的语法。

网格 grid 定义：
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

### 3. 理解内核运行的机制

**内核在 triton 中是并行执行的，每个内核负责处理的数据范围不一样，一般通过 `idx`（张量）决定**。内核执行的个数跟块数有关，多内核并行实际就是多块并行，即多个块可以在不同的多处理器（SMs）上同时运行，每个块内的线程也在其所属的 SM 上并行执行。

为了更好的理解有多少个块（内核）并行执行，各自执行的数据范围是多少，可以通过在向量相加的内核中，添加打印信息，修改后的内核代码如下所示，其他代码跟上一节一样。
```python
os.environ["TRITON_INTERPRET"] = "1"

N = 50 # 为了减少打印信息量，数据元素数量和BLOCK_SIZE分别调小到 50 和 32
BLOCK_SIZE=32

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
    
    # 程序数目 = 块的数目（grid 大小） = 内核并行运行的次数
    assert tl.num_programs(axis=0) == triton.cdiv(N,BLOCK_SIZE) 
    print(f"内核将执行 {tl.num_programs(axis=0)} 次（块数）。")
    print("pid: ", pid)
    print("block_start: ", block_start)
    print("offsets: ", offsets)
    print("idx: ", idx)
```

程序运行后，输出信息如下所示，
>内核将执行 [2] 次（块数）。
pid:  [0]
block_start:  [0]
offsets:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31]
idx:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31]
内核将执行 [2] 次（块数）。
pid:  [1]
block_start:  [32]
offsets:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31]
idx:  [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55
 56 57 58 59 60 61 62 63]
Triton 向量加法成功！
Triton 向量加法时间: 12.23 ms
PyTorch 向量加法时间: 95.85 ms

从输出信息可以看出，内核执行次数 = grid，并行执行的内核其处理的数据范围也是不一样的，第一个内核处理数据范围是 [0,1,2..,31]，第二个是 [32,33,..,63]。

注意，虽然 `TRITON_INTERPRET=1` 设置成解释模式可以打印线程索引等信息，但某些高级的 Triton 操作（如归约、复杂的内存访问模式）在解释模式下可能存在限制或未完全实现，这将导致内核运行报错。

### 3. 融合 Softmax

本节的例子是实现一个融合的 “Softmax” 操作，对于某类矩阵，其速度明显高于“PyTorch” 的原生操作：那些矩阵的每一行都能存储在 GPU 的 `SRAM` 中。这个例子是让我们学习**内核融合对于带宽受限操作的优势**。

softmax 将向量变换为值为正且和为 1 的概率分布，其公式为：
> pytorch 中 softmax 实现的 c++ 代码在 [Softmax.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SoftMax.cpp)

$$\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)} \\
\text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)$$
其中：
- $x_i$ 是输入向量中的第 $i$ 个元素。
- $n$ 是输入向量的长度。
- 输出的每个值都是在 0 到 1 之间，并且所有输出值的总和为 1，表示概率分布。

原生 softmax 算子如下：

```python
import torch

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

naive_softmax 函数实现了行级（row-wise）的 softmax 计算，通过以下步骤实现。

1. `x_max = x.max(dim=1)[0]`： 为了数值稳定性，减去每一行的最大值，避免在计算 exp 时出现溢出（overflow）。x.max(dim=1) 返回每一行的最大值和对应的索引，[0] 表示只要第一部分，即取最大值那部分。
2. `z = x - x_max[:, None]`: x 减去最大值实现数值稳定性，[:, None] 切片将 x_max 从形状 (M,) 扩展为 (M, 1)，然后广播减法。
3. `numerator = torch.exp(z)`：计算 exp(z) 作为分子部分 (numerator)。
4. `denominator = numerator.sum(dim=1)`：计算每一行的和作为分母部分 (denominator)。
5. `ret = numerator / denominator[:, None]`：分子部分除以分母部分，得到softmax 值 (ret)。

注意，代码中注释提到的数据访问量，计算 `y = naive_softmax(x)` $x\in R^{M\times N}$.总读取：5MN + 2M 个元素，总写入：8MN + 2M 个元素，总数据（内存）访问量（MAC） = 7MN + 4M。

上述实现明显不够搞笑，MAC 过大，显然不够高效；因此我们更希望使用一个自定义的“融合”内核，仅对 X 进行一次读取，并在芯片上完成所有所需的计算。这样只需要读取和写回字节数，理论上可以达到大约 $4 = (8MN + 4M) / 2MN$ 倍的加速效果（即，）。虽然 “torch.jit.script” 标志旨在自动实现这种“内核融合”，但它依然存在一些不足（后面分析）。

1，参考前面的向量相加的例子，实现的 softmax 内核及内核调用函数如下所示:

```python
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
```

这里实现的是逐行的 softmax 操作，所以让每个线程负责处理一整行的 softmax 计算，即处理连续的数据块，而不是让每个线程将处理一个单独的元素。这里的内核配置是一维的，和上一节向量相加的内核配置类似，但这样的内核不是最优性能的。

2，**优化后的 “Softmax” 内核**的运行方式如下：每个程序会按程序数量为步幅，加载输入矩阵 X 的一组行，对其进行归一化处理后，将结果写回输出矩阵 Y。

需要注意的是，“Triton” 有一个重要限制：每个块的元素数量必须是 2 的幂次方。因此，如果要处理任意形状的输入矩阵，我们需要在内部对每一行进行“填充”，并确保内存操作的正确性。

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

1，为什么列偏移用 tl.arange(0, BLOCK_SIZE) 而不是 tl.arange(0, n_cols)?

Softmax 内核中，每个程序（线程块）负责处理输入矩阵的一行，而 BLOCK_SIZE 决定了每行数据中每个程序一次性处理的列数。之所以使用 BLOCK_SIZE 而不是 N，是因为实际案例中矩阵列数千奇百怪，使用 BLOCK_SIZE（ 2 的幂例如 32、64、128 等），可以确保内存访问的对齐（GPU 的内存访问通常对齐到特定的边界（如 32 字节）），减少内存访问的开销，提高带宽利用率。

2，为什么需要 for 循环？

- 处理超过并行程序数的数据：GPU 上可用的并行程序（或线程块）数量是有限的，通常远小于数据的总行数 (n_rows)。
- 可扩展性：使用 for 循环可以让内核适应不同规模的数据集，而不需要根据数据大小动态调整网格大小。
- 优化资源利用：当一个程序在处理一行数据时，另一个程序可以同时处理下一行的数据，从而隐藏内存访问的延迟，提高整体吞吐量。

内核调用函数定义如下：
```python
# 1. 获取 GPU 硬件属性
device = torch.cuda.current_device() # GPU 设备名称
properties = driver.active.utils.get_device_properties(device) # Triton 的工具函数 get_device_properties 获取设备的详细属性
NUM_SM = properties["multiprocessor_count"] # SM 数量
NUM_REGS = properties["max_num_regs"] # 可用寄存器的最大数量
SIZE_SMEM = properties["max_shared_mem"] # 共享内存大小
WARP_SIZE = properties["warpSize"] # 线程束大小，一般为 32
target = triton.runtime.driver.active.get_current_target() # get_current_target() 获取当前 GPU 的架构信息，用于优化内核
kernels = {} # 用于缓存不同块大小（BLOCK_SIZE）的内核，避免重复编译。


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # 将列数 n_cols 调整为最接近的下一个2的幂。这有助于优化内存访问和并行计算。
    BLOCK_SIZE = triton.next_power_of_2(n_cols) 

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2 # 根据共享内存大小决定流水线阶段数。更多的阶段可以提高内核吞吐量，但会增加复杂性。

    # Allocate output
    y = torch.empty_like(x) # 创建与输入相同形状的输出张量 y

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        
        # 根据硬件属性和内核需求计算线程占用率（occupancy），确定同时运行的内核数量（num_programs）。
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows) # 确保程序数量（块数）不超过行数。

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y
```

`“NUM_REGS”` 表示通用寄存器的数量。在 `“CDNA”` 架构中，其等于所有可用寄存器`“NUM_GPRS”` 的一半。但也并不总是这样，在大多数情况下，所有寄存器都可以用作通用寄存器。

`ISA` 部分（“CDNA3” 的 3.6.4 节）
“VGPR”（矢量通用寄存器）分配来自两个池：通用 VGPR 和累积 VGPR。累积 VGPR 用于矩阵 “VALU” 指令，也可以直接从内存加载。一个波（`wave`）最多可以拥有 512 个 VGPR，总数为 512，其中每种类型最多 256 个。当一个波使用少于 512 个 VGPR 时，每种类型的数量是灵活的——不需要两种类型的数量相等。

**`“MAX_NUM_THREADS”` 表示每个多处理器（SM）中驻留线程的最大数量**，将其除以 `“WARP_SIZE”`，得到的是在一个 CU（多处理器）上**并行执行的最大波（wave）数**。