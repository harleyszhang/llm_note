- [1. Triton 基础函数](#1-triton-基础函数)
- [2. 向量相加](#2-向量相加)
  - [2.1 内核调用函数](#21-内核调用函数)
  - [2.2 内核定义函数](#22-内核定义函数)
  - [2.3 main 主函数](#23-main-主函数)
  - [2.4 `BLOCK_SIZE`、`gird` 和 `program_id` 意义](#24-block_sizegird-和-program_id-意义)
- [3. 理解内核运行的机制](#3-理解内核运行的机制)
  - [3.1 pdb/gdb 调试](#31-pdbgdb-调试)
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

一维向量相加是学习 Triton 编程模型中入门实例，代码如下所示，现在看不懂不要紧，有个大概映像和知道 kernel 执行的逻辑流程即可，后面会一步步分析。

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

总的来说，`triton` 内核代码分为三个部分，依次是：
1. 内核定义
2. 内核调用
3. main 主函数

#### 2.1 内核调用函数

向量相加代码中的内核调用函数是 `vector_add_triton`。

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

#### 2.2 内核定义函数

内核定义函数 `vector_add_kernel` 原型如下所示：

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
5. `mask = idx < N`: 作用是创建掩码，防止线程访问超出数据范围的元素

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

#### 2.3 main 主函数

最后就是 main 函数了，主要是：初始化张量、GPU 预热、执行 Triton 向量加法并记录时间、执行 PyTorch 向量加法并记录时间和验证结果并输出时间。这里的代码没什么好讲的，都是 pytorch 代码，记住下这个流程即可。

#### 2.4 `BLOCK_SIZE`、`gird` 和 `program_id` 意义

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

程序运行后，输出信息如下所示:
```bash
内核将执行 [2] 次（块数）。
pid:  [0]
block_start:  [0]
offsets:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
idx:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
内核将执行 [2] 次（块数）。
pid:  [1]
block_start:  [32]
offsets:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
idx:  [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63]
Triton 向量加法成功！
Triton 向量加法时间: 12.23 ms
PyTorch 向量加法时间: 95.85 ms
```

从输出信息可以看出，内核执行次数 = grid，并行执行的内核其处理的数据范围也是不一样的，第一个内核处理数据范围是 [0,1,2..,31]，第二个是 [32,33,..,63]。

注意，虽然 `TRITON_INTERPRET=1` 设置成解释模式可以打印线程索引等信息，但某些高级的 Triton 操作（如归约、复杂的内存访问模式）在解释模式下可能存在限制或未完全实现，这将导致内核运行报错。

#### 3.1 pdb/gdb 调试

1，pdb 是 python 内置的代码调试工具。在代码中插入 `import pdb; pdb.set_trace()`，程序执行到此会自动进入调试模式。常用命令：
- `p` 打印变量值
- `l` 查看当前代码行
- `n` 跳转到下一行代码，并停留在当前的作用域内（不进入函数）。
- `s` 进入当前执行代码行的函数内部。
- `c` 继续执行代码，直到遇到下一个断点。
- `q` 退出调试模式。

2，gdb 是用于调试 c/c++ 的工具。对于 triton 程序，它的启动命令是 `gbd -args xxx.py`。常用命令：
- `b` 设置断点。
- `r` 运行程序。
- `p` 打印变量值
- `n` 跳转到下一行代码，并停留在当前的作用域内（不进入函数）。
- `s` 进入当前执行代码行的函数内部。
- `c` 继续执行代码，直到遇到下一个断点。
- `q` 退出 gdb。

先看 debug ops 的用法
```python
@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                        # 获取当前块的 ID
    tl.static_print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    if pid == 1:
        tl.device_print("pid: ", pid)
    ...
```

在看 pdb 调试
```python
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
    offsets = tl.arange(0, BLOCK_SIZE)            # 生成当前块的线程偏移量
    import pdb; pdb.set_trace()
    idx = block_start + offsets                   # 计算每个线程负责的索引
```

### 参考资料

- [Understanding the Triton Tutorials Part 1](https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c)
