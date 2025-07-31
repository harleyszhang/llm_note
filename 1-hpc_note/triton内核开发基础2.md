- [1. 原生 softmax 算子](#1-原生-softmax-算子)
- [2. 一般 softmax 内核](#2-一般-softmax-内核)
- [3. 优化版 softmax 内核](#3-优化版-softmax-内核)
- [参考资料](#参考资料)

本节的例子是学习实现一个融合的 “Softmax” 算子：对于某类矩阵，其速度是显著快于 PyTorch 的原生操作，因为这些矩阵的行能够适应 GPU 的 SRAM。

这个例子可以学习到：

- 核函数融合对带宽受限操作的好处。
- Triton 中的归约运算符（Reduction operator）。

### 1. 原生 softmax 算子

Softmax 函数是一种常用于机器学习，特别是多分类问题中的激活函数。它的作用是将一个任意实数向量转换为一个概率分布，并确保输出的概率和为 1。给定输入 $x\in R^{M\times N}$，执行逐行 “Softmax”（对于 `2D` 张量，逐行计算对应维度 `dim = 1`），其公式为：
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

### 2. 一般 softmax 内核

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

### 3. 优化版 softmax 内核

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

`softmax` 函数中关键的代码是内核预编译与缓存部分，关键变量解释：
- `NUM_SM`：GPU 上的 SM（Streaming Multiprocessor，流多处理器）数量。
- `NUM_REGS`: 每个 SM 的最大寄存器数量。
- `NUM_GPRS`: 在 HIP（AMD GPU）架构下的寄存器数量。
- `SIZE_SMEM`: 每个 SM 的最大共享内存大小（单位：字节）。
- `WARP_SIZE`: 每个 warp（一个 warp 包含 32 个线程）的线程数，通常为 32。
- `n_regs`: 内核每个线程需要使用的寄存器数量。
- `size_smem`: 内核每个程序需要使用的共享内存大小。
- `num_warps`: 每个程序（程序指的是一个内核实例）使用的 warps 数量（grid/32）。

### 参考资料

- [Understanding the Triton Tutorials Part 1](https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c)
