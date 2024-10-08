- [triton 定义](#triton-定义)
- [triton 和 cuda 特性的对比](#triton-和-cuda-特性的对比)
- [triton 编译过程](#triton-编译过程)
- [张量维度判断](#张量维度判断)
- [矩阵元素指针算术](#矩阵元素指针算术)
- [网格、块和内核](#网格块和内核)
- [cuda 执行模型](#cuda-执行模型)
- [参考资料](#参考资料)


### triton 定义

`Triton` 是一种用于并行编程的语言和编译器，它旨在提供一个基于 Python 的编程环境，帮助高效编写自定义的深度神经网络（DNN）计算核，并在现代 GPU 硬件上以最大吞吐量运行。

Triton 的编程在语法和语义上与 `Numpy` 和 `PyTorch` 非常相似。然而，作为低级语言，开发者需要注意很多细节，特别是在内存加载和存储方面，这对于在低级设备上实现高速计算非常关键。

Triton 有以下特点：

- Intermediate Lauguge：基于Python的DSL
- tiled Neural Network Compute：面向GPU体系特点，自动分析和实施神经网络计算的分块
- Compiler：编译器

**即 Triton 是语言，也是编译器**。

### triton 和 cuda 特性的对比

下述表格将 Triton 与 CUDA 的关键特性进行了对比，Triton 的编程模型在设计上的一大优势在于无需手动划分块，即 Block-wise 编程，Block 上面的归用户处理，Block 内部的归 Triton compiler 自动化处理。同时Triton 在内存、TensorCore 以及向量化等方面都是自动进行的，可以简化一些 CUDA 编程中的手动优化过程，提供更多的自动化特性以提高开发效率和性能。

|  | CUDA | Triton |
| --- | --- | --- |
| Memory | Global/Shared/Local | Automatic |
| Parallelism | Threads/Blocks/Warps | Mostly Blocks |
| Tensor Core | Manual | Automatic |
| Vectorization | .8/.16/.32/.64/.128 | Automatic |
| Async SIMT | Support | Limited |
| Device Function | Support | Not Walable |

### triton 编译过程

triton 生成 triton IR 再到 LLVM IR 再到 PTX。

### 张量维度判断

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

### 矩阵元素指针算术

为了访问矩阵中的特定元素，使用指针算术计算其线性地址。对于矩阵 $A$  形状为 $(M, K)$，元素 $A[m, k]$ 的线性地址计算如下：

$$\text{Address}(A[m, k]) = A_{\text{ptr}} + m \times K + k$$

其中：
- $A_{\text{ptr}}$ 是矩阵 $A$ 的起始指针（即  A[0,0]  的地址）。
- $m \times K$  是跳过前 $m$ 行的元素数量。
- $k$ 是当前行中跳过的元素数量。

**子块地址计算方法**：

对于矩阵 $A$ 形状为 $(M, K)$，元素顺序为：
\[
A = \begin{bmatrix}
A[0,0] & A[0,1] & \dots & A[0,K-1] \\
A[1,0] & A[1,1] & \dots & A[1,K-1] \\
\vdots & \vdots & \ddots & \vdots \\
A[M-1,0] & A[M-1,1] & \dots & A[M-1,K-1]
\end{bmatrix}
\]
内存中的存储顺序：
\[
A[0,0], A[0,1], \dots, A[0,K-1], A[1,0], A[1,1], \dots, A[1,K-1], \dots, A[M-1,K-1]
\]

在分块矩阵乘法中，矩阵被划分为多个子块。每个子块由其在行和列方向上的起始索引定义。矩阵子块加载、存储和计算的伪代码：
```bash
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```

- a_block 地址范围：从 m 到 m + BLOCK_SIZE_M，列索引范围 k: k + BLOCK_SIZE_K。对应的子块矩阵元素地址是二维的，假设矩阵存储是行连续的，则地址范围为 (A_ptr + m * K + k, A_ptr + (m + BLOCK_SIZE_M) * K + k + BLOCK_SIZE_K) -1。
- b_block 地址范围：B_ptr + k * N + n, B_ptr + (k + BLOCK_SIZE_K) * N + (n + BLOCK_SIZE_N) - 1。

如果是 triton 中实现上述地址的计算，对应代码为:

```python
# 1，行块和列块 id，即第几个块
pid_m = tl.program_id(axis=0) # 这里的 pid_m 就是上面的 m 变量
pid_n = tl.program_id(axis=0)
# 2，行和列索引范围
# pid_m * BLOCK_SIZE_M 是块在行方向的起始行索引，加上 tl.arange(0, BLOCK_SIZE_M)[:, None] 生成的行偏移量。
offsets_m = pid_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
# pid_n * BLOCK_SIZE_N 是块在列方向的起始列索引，加上 tl.arange(0, BLOCK_SIZE_N)[None, :] 生成的列偏移量。
offsets_n = pid_n + tl.arange(0, BLOCK_SIZE_N)[None,:]

acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, K, BLOCK_SIZE_K):
    offsets_ak = k + tl.arange(0, BLOCK_SIZE_K)[None,:]
    offsets_bk = k + tl.arange(0, BLOCK_SIZE_K)[:, None]

    # offsets_m * K：跳过前 offsets_m 行，每行有 K 个元素。
    a_idx = A_ptr + offsets_m * K + offsets_ak 
    b_idx = B_ptr + offsets_bk * N + offsets_n

    a_block = tl.load(a_idx, mask=(offsets_m < M) & (offsets_ak < K), other=0.0)
    b_block = tl.load(b_idx, mask=(offsets_bk < K) & (offsets_n < N), other=0.0)
    acc = tl.dot(a, b, acc=acc)

# offs_m * N：跳过前 offs_m 行，每行有 N 个元素。offsets_n：当前块负责的列偏移量。
c_idx = C_ptr + offsets_m * N + offsets_n
tl.store(c_ptr + c_idx, acc, mask = (offsets_m < M) & (offsets < N), other=0.0)
```

`META['BLOCK_SIZE']` 表示每个块（block）的大小，这个值很重要，因为它直接影响到内核的并行性和性能。

### 网格、块和内核

**不同的 grid 则可以执行不同的程序（即 kernel）**。`grid` 定义了内核（kernel）执行的网格大小，即有多少个块（`blocks`）将被启动来执行一个内核，同时每个块包含 'BLOCK_SIZE' 个线程（`threads`），一个 block 中的 thread 能存取同一块共享的内存。

与 cuda 编程把 thread 当作并行执行的基本单位不同，在 Triton 中，**块 block 才是内核并行执行的基本单位**，每个块负责处理任务的一个子集，通过合理划分块大小，可以充分利用 GPU 的并行计算能力。

### cuda 执行模型

在执行 CUDA 程序的时候，每个 SP（stream processor） 对应一个 thread，每个 SM（stream multiprocessor）对应一个 Block。

### 参考资料

- [OpenAI Triton分享：Triton概述](https://zhuanlan.zhihu.com/p/750277836)
- [谈谈对OpenAI Triton的一些理解](https://zhuanlan.zhihu.com/p/613244988)
- [SOTA Deep Learning Tutorials](https://www.youtube.com/@sotadeeplearningtutorials9598)