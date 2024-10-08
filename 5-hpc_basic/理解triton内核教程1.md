- [1. Triton 基础函数](#1-triton-基础函数)
- [2. 向量相加](#2-向量相加)
  - [2.1 `BLOCK_SIZE`、`gird` 和 `program_id` 意义](#21-block_sizegird-和-program_id-意义)
- [3. 理解内核运行的机制](#3-理解内核运行的机制)
- [3. Softmax 算子](#3-softmax-算子)
  - [原生 softmax 算子](#原生-softmax-算子)
  - [简单 softmax 内核](#简单-softmax-内核)
  - [优化版 softmax 内核](#优化版-softmax-内核)
- [参考资料](#参考资料)


### 1. Triton 基础函数

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

总的来说，triton 代码分为三个部分，依次是：
1. 内核定义
2. 内核调用
3. main 主函数

1，先看内核调用函数 `vector_add_triton`。

Triton 内核的调用类似于 CUDA 的内核调用，但具有更高的抽象和简化的语法。关键的是网格 grid 定义：
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

只有正确理解内核中的 `pid`、`block_start`、`offsets` 和 `idx` 这四个概念，才能写出正确、高效的内核代码。

1. `pid`（Program ID）: 当前块（Block）的唯一标识符，代表块在整个网格（Grid）中的位置（第几个块）。一维的 `grid` 的 `pid = tl.program_id(0)`。
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

#### 2.1 `BLOCK_SIZE`、`gird` 和 `program_id` 意义

`kernel` 实际**要被重复执行很多次的**, 每次执行处理输入的一部分，直到所有输入处理完。但 kernel 里面没有上述过程 `for` 循环，原因是这些不同数据部分的处理实际是**并行执行的**。`program_id` 则是虚拟的 `for`“循环”里面的 `index` (第几次循环)，axis=0 , 是说这个"循环"只有一层，axis=1 则说明"循环"有两层，以此类推。而 `grid` 的意义就是用来说明**虚拟“循环”有多少层，每层执行多少次**。最后，`BLOCK_SIZE` 则是用来说明每次“循环”（每次内核执行）加载的内存/元素的数量。

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

### 3. Softmax 算子

本节的例子是学习实现一个融合的 “Softmax” 算子：对于某类矩阵，其速度是显著快于 PyTorch 的原生操作，因为这些矩阵的行能够适应 GPU 的 SRAM。

这个例子可以学习到：

- 核函数融合对带宽受限操作的好处。
- Triton 中的归约运算符（Reduction operator）。

#### 原生 softmax 算子

Softmax 函数是一种常用于机器学习，特别是多分类问题中的激活函数。它的作用是将一个任意实数向量转换为一个概率分布，并确保输出的概率和为 1。给定输入 $x\in R^{M\times N}$，执行逐行 “Softmax”，其公式为：
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
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    """
    x_max = x.max(dim=1)[0] # read  MN elements ; write M  elements
    z = x - x_max[:, None] # read MN + M elements ; write MN elements
    numerator = torch.exp(z) # read  MN elements ; write MN elements
    denominator = numerator.sum(dim=1) # read  MN elements ; write M  elements
    ret = numerator / denominator[:, None]  # read MN + M elements ; write MN elements
    
    return ret
```

naive_softmax 函数实现了行级（row-wise）的 softmax 计算，通过以下步骤实现。

1. `x_max = x.max(dim=1)[0]`： 为了数值稳定性，减去每一行的最大值，避免在计算 exp 时出现溢出（overflow）。x.max(dim=1) 返回每一行的最大值和对应的索引，[0] 表示只要第一部分，即取最大值那部分。
2. `z = x - x_max[:, None]`: x 减去最大值实现数值稳定性，[:, None] 切片将 x_max 从形状 (M,) 扩展为 (M, 1)，然后广播减法。
3. `numerator = torch.exp(z)`：计算 exp(z) 作为分子部分 (numerator)。
4. `denominator = numerator.sum(dim=1)`：计算每一行的和作为分母部分 (denominator)。
5. `ret = numerator / denominator[:, None]`：分子部分除以分母部分，得到softmax 值 (ret)。

注意，代码中注释提到的数据访问量，计算 `y = naive_softmax(x)`，总读取：5MN + 2M 个元素，总写入：3MN + 2M 个元素，总数据（内存）访问量（MAC） = 8MN + 4M。

#### 简单 softmax 内核

上述 native 实现 MAC 过大，明显不够高效，因此需要考虑使用一个自定义的“融合”内核，只读取一次 $X$ 并在芯片上完成所有计算。这样只需要一次读取和写回 $X$ 的字节数，理论上可以达到大约 $4 = (8MN + 4M) / 2MN$ 倍的加速效果。虽然 “torch.jit.script” 标志旨在自动实现这种“内核融合”，但它依然存在一些不足（后面分析）。

那么问题来了，和前面处理一维向量不同，二维矩阵数据如何读取和加载呢？办法是让  triton 程序在**给定步幅大小**的情况下迭代每一行。需要注意的是，“Triton” 有一个重要限制：每个块的元素数量必须是 2 的幂次方。因此，如果要处理任意形状的输入矩阵，我们需要在内部对每一行进行“填充”，并确保内存操作的正确性。

参考前面的向量相加的例子，实现的 softmax 内核及内核调用函数如下所示:

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, 
                output_row_stride, n_cols, BLOCK_SIZE:: tl.constexpr):
    
    row_idx = tl.program_id(0) # 一个块处理一行元素，idx 表示第几行，每行之间的处理是并行的
    row_start_ptr = input_ptr + row_idx * input_row_stride # # 步幅表示我们需要增加指针多少才能前进 1 行
    col_offsets = tl.arange( 0 , BLOCK_SIZE) # 块大小是大于 n_cols 的下一个 2 的幂，因此我们可以将每一行放在一个块中
    input_ptrs = row_start_ptr + col_offsets 

    row = tl.load(input_ptrs, mask=col_offsets < n_cols）# using a mask since BLOCK_SIZE may be > than n_cols

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 将结果行数据写入到指定地址范围中
    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # grid = lambda meta: (triton.cdiv(n_rows*n_cols, meta['BLOCK_SIZE']),)

    # 增加每行分配的 warp 数量（num_warps）
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_wraps = 16

    softmax_kernel[n_rows](x, 
        y, 
        x.stride(0), 
        y.stride(0), 
        n_cols, 
        num_warps=num_warps,
        BLOCK_SIZE = BLOCK_SIZE)

    return y
```

这里实现的是逐行的 softmax 操作，所以让**每个块负责处理一整行的 softmax 计算**。

#### 优化版 softmax 内核

**优化版 “Softmax” 内核**的运行机制是：每个计算 `kernel` 会加载输入矩阵 $X$ 中的一组行，组大小就是 `grid_size`。行数据进行归一化处理后，将结果写入输出矩阵 Y。

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, 
                output_row_stride, n_rows, n_cols, BLOCK_SIZE:: tl.constexpr):
    
    row_start = tl.program_id(0) # 一个块处理一组行元素，row_start 表示第几组，每行之间的处理是并行的
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step,num_stages=num_stages):
        row_start_ptr = row_idx + row_idx * input_row_stride # 行步幅表示我们需要增加指针多少才能前进 1 行
        col_offsets = tl.arange( 0 , BLOCK_SIZE) # 块大小是大于 n_cols 的下一个 2 的幂，因此我们可以将每一行放在一个块中
        input_ptrs = row_start_ptr + col_offsets 

        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')）# using a mask since BLOCK_SIZE may be > than n_cols

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # 将结果行数据写入到指定地址范围中
        out_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = out_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
```

1，为什么列偏移用 tl.arange(0, BLOCK_SIZE) 而不是 tl.arange(0, n_cols)?

Softmax 内核中，每个程序（线程块）负责处理输入矩阵的一行，而 BLOCK_SIZE 决定了每行数据中每个程序一次性处理的列数。之所以使用 BLOCK_SIZE 而不是 N，是因为实际案例中矩阵列数千奇百怪，使用 `BLOCK_SIZE= triton.next_power_of_2(n_cols)`（ 2 的幂例如 32、64、128 等），可以确保内存访问的对齐（GPU 的内存访问通常对齐到特定的边界（如 32 字节）），减少内存访问的开销，提高带宽利用率。

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
    num_warps = 8 # 预设为 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2 # 根据共享内存大小决定流水线阶段数。更多的阶段可以提高内核吞吐量，但会增加复杂性。

    # Allocate output
    y = torch.empty_like(x) # 创建与输入相同形状的输出张量 y

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        # 通过 softmax_kernel.warmup 和 kernel._init_handles() 获取内核对寄存器和共享内存的需求
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs # 内核所需寄存器数量
        size_smem = kernel.metadata.shared # 内核所需共享内存大小
        if is_hip():
        # `“NUM_REGS”` 表示通用寄存器的数量。在 `“CDNA”` 架构中，其等于所有可用寄存器`“NUM_GPRS”` 的一半。但也并不总是这样，在大多数情况下，所有寄存器都可以用作通用寄存器。

        # `ISA` 部分（“CDNA3” 的 3.6.4 节）
        # “VGPR”（矢量通用寄存器）分配来自两个池：通用 VGPR 和累积 VGPR。累积 VGPR 用于矩阵 “VALU” 指令，也可以直接从内存加载。一个波（`wave`）最多可以拥有 512 个 VGPR，总数为 512，其中每种类型最多 256 个。当一个波使用少于 512 个 VGPR 时，每种类型的数量是灵活的——不需要两种类型的数量相等。
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
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y
```

调用内核的函数代码主要就是在计算 `num_programs`（程序数量），也是前面说的 grid（blocks数量）。这里没有使用 num_programs = elements / block_size（公式简化了下没有用向上取整），是因为直接这样设置**会忽略了 GPU 的寄存器和共享内存限制，而导致部分 SM 上程序数量不足**。而设置 num_programs = occupancy × SM 数量，是为了动态调整程序数量以适应不同硬件资源和内核需求，以最大化资源利用率和隐藏延迟。

值的注意的是，occupancy 的计算有不同 API 和不同计算方式。上述代码是基于寄存器和共享内存的资源限制，**计算每个 SM 上最多可同时运行的程序数量（SM 中驻留的块数目）**。

对于 CUDA/NVIDIA 架构，**基于寄存器的限制**，一个线程需要使用 n_regs 寄存器，则所有程序数量使用 num_warps * WARP_SIZE * n_regs 个寄存器，而一个 SM 支持最多 NUM_REGS 个寄存器，则一个 SM 支持最多 `NUM_REGS // (n_regs * WARP_SIZE * num_warps` 个程序实例。
```python
# 这里 occupancy 表示每个 SM 上可以同时运行的程序数量
occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
```

再考虑共享内存限制，每个内核程序需要 size_smem 共享内存，则一个 SM 支持最多 `SIZE_SMEM // size_smem` 个程序实例（块）。

```python
# 确保共享内存的使用不会超出每个 SM 的最大容量。
occupancy = min(occupancy, SIZE_SMEM // size_smem)
```

softmax 函数中关键的代码是内核预编译与缓存部分，关键变量解释：
- NUM_SM：GPU 上的 SM（Streaming Multiprocessor，流多处理器）数量。
- NUM_REGS: 每个 SM 的最大寄存器数量。
- NUM_GPRS: 在 HIP（AMD GPU）架构下的寄存器数量。
- SIZE_SMEM: 每个 SM 的最大共享内存大小（单位：字节）。
- WARP_SIZE: 每个 warp（一个 warp 包含 32 个线程）的线程数，通常为 32。
- n_regs: 内核每个线程需要使用的寄存器数量。
- size_smem: 内核每个程序需要使用的共享内存大小。
- num_warps: 每个程序（程序指的是一个内核实例）使用的 warps 数量（grid/32）。

### 参考资料

- [Understanding the Triton Tutorials Part 1](https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c)
