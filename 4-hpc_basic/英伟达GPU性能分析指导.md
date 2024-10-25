- [一，概览](#一概览)
- [二，GPU 架构基础](#二gpu-架构基础)
- [三，GPU 执行模型](#三gpu-执行模型)
- [四，理解性能（数学带宽 vs 内存带宽）](#四理解性能数学带宽-vs-内存带宽)
  - [4.1，数学（算力 FLOPS）带宽 vs 内存带宽](#41数学算力-flops带宽-vs-内存带宽)
  - [4.2，矩阵乘法的算术强度计算及优化](#42矩阵乘法的算术强度计算及优化)
  - [4.3，如何分析模型推理的性能](#43如何分析模型推理的性能)
- [五，DNN 操作类别](#五dnn-操作类别)
  - [5.1，逐元素操作](#51逐元素操作)
  - [5.2，减少操作](#52减少操作)
  - [5.3，点积操作（Dot-Product Operations）](#53点积操作dot-product-operations)
- [六，总结](#六总结)
- [参考资料](#参考资料)

## 一，概览

本文主要介绍了 GPU 结构、操作的执行方式以及深度学习操作的常见限制，目的是为了更好的理解 GPU 的基本运行原理以及神经网络或特定网络层如何高效利用给定 `GPU`。

本文主要内容可分为以下部分：

1. **GPU 架构基础**：了解 GPU 基本结构有助于神经网络在给定 GPU 上的高效运行。
2. **GPU 执行模型**：理解神经网络层/算子是如何被分割和并行执行的。
3. **理解性能**：如何通过算术强度（arithmetic intensity）来评估 GPU 性能限制。
4. **DNN 算子类别**：深度学习算子的宽松分类及其通常适用的性能限制。

## 二，GPU 架构基础

GPU 是一种**高度并行的处理器架构**，由处理元件和内存层次结构组成。在较高层面上，NVIDIA® GPU 由多个流式多处理器（Streaming Multiprocessors，`SMs`）、片上 L2 缓存和高带宽 `DRAM` 组成。`SMs` 执行算术和其他指令，数据和代码通过 L2 缓存从 DRAM 中访问。举个例子，NVIDIA A100 GPU 包含 108 个 SMs，一个 40MB 的 L2 缓存，以及 80 GB HBM2 内存并提供高达 2039 GB/s 的带宽。

![simple-gpu-arch](../images/gpu_perf/simple-gpu-arch.svg)
图1. GPU 架构的简化视图

每个 `SM` 都有自己的指令调度器和各种指令执行管道。乘加运算是现代神经网络中最常见的运算，是全连接层和卷积层的构建块，这两个层都都可以看作是一组向量点乘的集合。 

下表显示了 NVIDIA 最新 GPU 架构上，不同数据类型的单个 SM 每个时钟周期的乘加操作次数。每个乘加操作都包含两个运算，因此将表中的吞吐量乘以 2，即可获得每个时钟的 FLOP 计数。要获得 GPU 的 `FLOPS` 速率，需要将其乘以 SM 数量和 SM 时钟速率。 例如，具有 108 个 SM 和 1.41 GHz 时钟频率的 A100 GPU 的峰值密集吞吐量为 156 TF32 TFLOPS 和 312 FP16 TFLOPS（应用程序实现的吞吐量取决于本文档中讨论的许多因素）。

![multi-add-op](../images/gpu_perf/multi-add-op.svg)
图 2. SM 每个时钟的乘加运算

如图2所示，`FP16` 操作可以在 Tensor Cores 或 NVIDIA CUDA® 核心中执行。此外，NVIDIA Turing™ 架构可以在 Tensor Cores 或 CUDA 核心中执行 `INT8` 操作。`Tensor Cores` 是在NVIDIA Volta™ GPU 架构中引入的，用于加速机器学习和科学应用中的矩阵乘法和累加操作。这些指令对小矩阵块（例如 4x4 块）进行操作。值得注意的是，Tensor Cores 可以以比输入更高的精度计算和累加乘积。例如，在使用 FP16 输入进行训练时，Tensor Cores 可以在不丢失精度的情况下计算乘积，并以 FP32 累加。**当数学运算无法用矩阵块表示时，它们将在其他 CUDA 核心中执行**。例如，两个半精度张量的逐元素加法将由 CUDA 核心执行，而不是 Tensor Cores。

## 三，GPU 执行模型

为了利用并行资源，GPU 会同时执行许多线程。

有两个概念对于理解线程数与 GPU 性能的关系至关重要：

1. GPU 使用两级线程层次结构执行函数。 给定函数的线程被分组为大小相等的线程块，并启动一组线程块来执行该函数。
2. GPU 通过切换到其他线程的执行来隐藏相关指令延迟。 因此，有效利用 GPU 所需的线程数的重要性远远高于核心数或指令流水线数。

2 级线程层次结构是因为 GPU 内部有很多 SM，**每个 SM 又具有用于执行多个线程的流水线**，并通过共享内存和同步来实现线程间的通信。GPU 运行时，线程块（thread block）是放在 SM 上执行，这使得线程块中的所有线程都能够高效地通信和同步。因为使用单个线程块启动函数将只有一个 SM 被分配工作，因此要充分利用具有多个 SM 的 GPU，就需要启动多个线程块。又因为一个 SM 可以同时执行多个线程块，通常希望线程块的数量是 SM 数量的几倍。这样做的原因是为了最大限度地减少“尾部”效应，即在函数执行结束时，只有少数活动线程块保留，从而导致该时间段内 GPU 利用率不足，如图 3 所示。

![utilize-8sm-gpu](../images/gpu_perf/utilize-8sm-gpu.svg)

图3. 当一次占用 1 个块/SM 的 12 个线程块启动执行时，8-SM GPU 的利用率。 此处，块分 2 波执行，第一波利用 100% 的 GPU，而第二波仅利用 50%。

我们使用术语 “wave” 来指代一组并发运行的线程块。最有效的做法是启动可以在多个线程块中执行的函数，这样在“尾部”波浪中花费的时间较少，从而最小化“尾部”效应，并减少需要处理它的必要性。对于高端GPU，通常只有少于300个线程块的启动需要检查“尾部”效应。

## 四，理解性能（数学带宽 vs 内存带宽）

一个函数在给定处理器上的性能受到以下三个因素之一的限制：**内存带宽** memory bandwidth、**数学带宽** math bandwidth 和**延迟** Latency。

- **CPU/GPU 内存带宽**： 是指主板的 CPU（显卡的 GPU）与内存之间的数据传输速率，决定了将数据从内存 (vRAM) 移动到计算核心的速度。通常以“字节/秒”或“GB/s”为单位。
> 这里的数学带宽其实就是 GPU 算力。

### 4.1，数学（算力 FLOPS）带宽 vs 内存带宽  

考虑一个简化的模型，其中一个函数从内存中读取输入，执行数学运算，然后将输出写入内存。

假设 $\text{T\_mem}$ 表示访问内存所花费的时间，$\text{T\_math}$ 表示执行数学运算所花费的时间。如果进一步假设不同线程的内存和数学部分可以重叠，那么函数的总时间为 $\text{(T\_mem, T\_math)}$ 。其中较长的时间决定了性能的限制：如果数学时间较长，则称函数受数学限制；如果内存时间较长，则称函数受内存限制；即所谓的 memory bound vs math/compute bound。

**内存或数学运算花费多少时间取决于算法及其实现，以及处理器的两种带宽**。 内存时间等于内存中访问的字节数除以处理器的内存带宽。数学时间等于运算次数除以处理器的数学带宽。因此，在给定的处理器上，如果一个特定的算法满足以下条件，那么它是受数学限制的：

$T\_math > T\_mem$ 可表示为 $\#ops / BW\_math > \# bytes / BW\_mem$。

这里的内存中访问的字节数 $\#bytes$ 其实是每个神经网络 `layer` 的内存访问代价 `MAC`，数学运算次数 $\#ops$ 其实就是 `FLOPs`。

上述不等式通过简单的代数变换，可以重新排列为:

$$\frac{\text{\#ops}}{\#bytes} > \frac{\text{BW\_math}}{\text{BW\_mem}}$$

- 左边是**算法实现操作数与访问字节数的比值**，被称为算法的**算术强度**（`arithmetic intensity` (`AI`) ，算数强度和计算强度意义等价）。
- 右边是**处理器的数学带宽与内存带宽的比值**，有时被称为**操作：字节比率**（`ops:byte ratio`）。

算术强度通俗理解就是计算量除以访存量后的值，表示此模型/网络层**在计算过程中，每 `Byte` 内存交换到底用于进行多少次浮点运算**，单位是 FLOPs/Bytes。可以看到，**模型计算强度越大，其内存使用效率越高**。因此，对于给定的 GPU：
- 如果算法的算术强度高于 GPU 的 `ops:byte ratio`，那么该算法受算力限制的，也称 `math bound`，即**性能受算力 `FLOPS` 限制**（算力受限/计算密集型算子）。
- 相反，如果算法的算术强度低于 GPU 的 `ops:byte ratio`，则该算法受内存限制，也称 `memory bound`，即**性能受内存带宽限制**（内存受限/访存密集型算子）。

![Figure 4: Roofline Model](../images/gpu_perf/roof_line_model.png)

总结：**应该尽可能让算法/网络层的算术强度高于 GPU 的 `ops:byte ratio`，这样才能充分利用 `gpu` 的算力**。

此外，算术强度和 ops:byte ratio 的分析假设工作负载足够大，能够饱和给定处理器的数学和内存流水线。但是，如果工作负载不足够大，或者没有足够的并行性，处理器将被低效利用，性能将受到延迟的限制。例如，考虑只启动一个线程，它将访问 16 字节并执行 16000 次数学运算。虽然算术强度为 1000 FLOPS/B，根据 V100 GPU 的情况，执行应该受数学限制，但是只创建一个线程严重低效地利用了 GPU，几乎使其所有的数学流水线和执行资源处于空闲状态。此外，算术强度的计算假设输入和输出仅从内存中访问一次。算法实现中多次读取输入元素并不少见，这将有效降低算术强度。因此，**算术强度是一个一阶近似值**；如果需要更准确的分析，还应使用性能分析器的信息。

### 4.2，矩阵乘法的算术强度计算及优化

全连接层/线性层的操作本质上执行的是矩阵乘法，因此分析矩阵乘法的算术强度其实就是分析线性层的算术强度。

- 输入矩阵 $A: (M, K)$
- 权重矩阵 $B: (K, N)$ 
- 输出矩阵 $C: (M, N)$，

即 $C = A \times B$，对应的则是输入维度为 $K$，输出维度为 $N$，$batch\_size = M$ 的全连接层，数据类型为 `FP16`。该矩阵乘法操作/线性层的**算术强度**为:

$$\frac{2MKN}{2(MK + KN + MN)}$$

![tiled_matrix_multiplication](../images/gpu_perf/tiled_matrix_multiplication.webp)

具体示例在下表 1 中。对于这些例子，我们将比较算法的算术强度与 NVIDIA Volta V100 PCle GPU 的 ops:byte ratio。V100 PCle GPU  的峰值算力为 112 FP16 Tensor TFLOPS，**片外内存带宽**约为 900 GB/s，芯片上 L2 缓存的带宽为 3.1 TB/s，因此其 `ops:byte ratio` 在 40 到 124.4 之间，取决于操作数据的来源（片内或片外存储器）。

假设 GPU 在 FP16 输入上进行 Tensor Core 操作，并使用 FP32 累积，如果数据从 GPU 的内存加载，则 `FLOPS：B`（**ops:byte ratio**）为 `124.4 = 112 / 0.9`。下表显示了一些常见网络层的算术强度。

![memory_math_bound](../images/gpu_perf/memory_math_bound.png)

上述表格第一行的计算过程如下:

$$
\text{arithmetic intensity} = \frac{\text{FLOPs}}{\text{MAC}} = \frac{\#op}{\#bytes} = \frac{2MKN}{2(MK + KN + MN)} = \frac{2 \cdot 512 \cdot 1024 \cdot 4096}{2 \cdot (512\cdot1024 + 1024\cdot4096 + 512\cdot 4096 )}\approx 315
$$

即该线性层（矩阵乘法）的算术强度为 $315$，大于 V100 PCle 的 $124.4$。因此，在 V100 PCle 上，**该矩阵乘法受到算术限制，即 GPU 将被充分利用**。

另外，从表格可以看出，大部分 layer 的算术强度都较低，比如第二行对应于批量大小为 $1$ 的线性层。在这种情况下，线性层变为受内存限制而不是算术限制，这就是为什么**深度学习模型通常尽可能不使用批量大小为 $1$ 进行训练或推理的原因，因为这种情况下 GPU 无法被充分利用**。

当然，这是一种简化的分析，毕竟我们只考虑了算法中使用的算法操作，在实践中，函数还包含了对算法中未明确表示的操作的指令，如**内存访问指令、地址计算指令、控制流指令**等等。

### 4.3，如何分析模型推理的性能

1. count arithmetic intensity ：$ops/bytes$
2. count `ops:byte ratio`: $BW\_math/BW\_mem$
3. 比较 (1) 和 (2)

![analyzing_performance](../images/gpu_perf/analyzing_performance.webp)

## 五，DNN 操作类别 

虽然现代神经网络是由多个层构建的，但根据计算的性质，它们的操作分为三个主要类别。

### 5.1，逐元素操作

逐元素操作（Element-wise operations）可以是一元或二元操作；关键在于该类别中的层对张量中的**每个元素都单独进行数学操作**，独立于其他元素。

例如，`ReLU` 层对输入张量中的每个 $x$ 返回 $max(0, x)$。类似地，两个张量的逐元素相加会独立计算每个输出的和值，不受其他和值的影响。大多数**非线性操作**（sigmoid、tanh 等）、缩放、偏置、加法等都属于逐元素操作层，**这些层往往都受到内存限制**，因为它们每访问一个字节执行的操作很少。
> 有关激活函数的更多细节，可以在[《优化内存限制层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)的激活 Activations 部分找到。

### 5.2，减少操作

减少操作（Reduction operations）是对输入张量值的范围进行计算并生成结果值的操作。

例如，池化（`pooling`）层在输入张量的某些邻域上计算值。批量归一化（`Batch normalization`）层先计算张量的平均值和标准差，再在每个输出元素的运算中使用它们。除了池化和归一化层外，`SoftMax` 也属于减少（reduction）类别。**典型的减少操作层的算术强度比较低，因此受到内存限制**。
> 有关池化层的更多详细信息，请参考[池化（Pooling）](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)部分。

### 5.3，点积操作（Dot-Product Operations）

该类别的操作可以表示为两个张量元素的点积，通常是权重（学习参数）张量和激活张量的点积计算。

全连接（fully-connected）层就是点积操作，其可以单独作为一个 layer，也可作为 rnn 或者 attention 层中的部分。全连接层可以自然地表示为**矩阵-向量**和**矩阵-矩阵**乘法。卷积也可以表示为一组点积的集合——一个向量是给定滤波器的参数集，另一个是应用该滤波器的“展开”激活区域。由于滤波器在多个位置应用，卷积也可以看作是矩阵-向量或矩阵-矩阵乘法操作（请参考[卷积算法](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#conv-algo)）。
> fully-connected/liner layer 全连接层/线性层。

如果相应的矩阵足够大，点积类别中的操作可能会受到数学限制。然而，对于较小的尺寸，这些操作最终会受到内存限制。例如，应用于单个向量（大小为 `1` 的小批量张量）的全连接层会受到内存限制。矩阵-矩阵乘法的性能将在[《NVIDIA 矩阵乘法背景用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)中进行更详细的讨论。关于将某种类型的层建模为矩阵乘法的信息可以在相应的指南中找到：

- [《NVIDIA优化线性/全连接层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html)
- [《NVIDIA优化卷积层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)
- [《NVIDIA优化循环层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/index.html)
  
## 六，总结

可以通过以下步骤粗略的估计指定 GPU 上特定运算函数的性能限制：

1. 查找 `GPU` 上的 `SM`数量（**算力**），并确定 GPU 的 `ops:byte ratio`。
2. 计算算法的算术强度。
3. 通过估计线程块的数量和大小，确定是否有足够的并行性来饱和 GPU。如果线程块的数量至少大约是 SM 数量的 4 倍，并且每个线程块由几百个线程组成，那么可能有足够的并行性。
4. 对于特定层类型，英伟达相应的指南文档提供了有关并行化的更多直观信息（参考[《NVIDIA 优化线性/全连接层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html)、[《NVIDIA 优化卷积层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)和[《NVIDIA 优化循环层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/index.html)；[《NVIDIA 优化内存限制层用户指南》](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)也可能有所帮助，尽管这些层通常受到内存限制）。

另外，最可能的性能限制因素是：

- 如果没有足够的并行性，则受到延迟的限制。
- 如果有足够的并行性，并且算法的算术强度高于 GPU 的字节比率，则受到数学带宽的限制。
- 如果有足够的并行性，并且算法的**算术强度低于 GPU 的字节比率，则受到内存带宽的限制**。

## 参考资料

1. [How to design a high-performance neural network on a GPU](https://medium.com/deep-dives-into-computer-science/how-to-design-a-high-performance-neural-network-on-a-gpu-2f7ada309724)
2. [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#dot-prod-op)
