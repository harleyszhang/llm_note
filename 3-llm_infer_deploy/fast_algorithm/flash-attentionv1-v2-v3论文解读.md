- [1. FlashAttention-v1](#1-flashattention-v1)
  - [Original Softmax](#original-softmax)
  - [Online Softmax](#online-softmax)
  - [Tiling](#tiling)
  - [Roofline](#roofline)
  - [SRAM](#sram)
  - [FlashAttention](#flashattention)
- [2. FlashAttention-v2](#2-flashattention-v2)
- [3. FlashAttention-v3](#3-flashattention-v3)
- [4. FlashDecoding \& FlashDecoding++](#4-flashdecoding--flashdecoding)
- [参考资料](#参考资料)


## 1. FlashAttention-v1

### Original Softmax

### Online Softmax

### Tiling

Parallel online normalizer calculation.

### Roofline

How to understand Roofline Model?

### SRAM

**FlashAttention 论文中说的 `SRAM` 是指哪种 GPU 内存类型？**

1，可以从 cuda 编程和算法角度理解 SRAM 是 L1 Cache (数据缓冲)。

FlashAttention 核心是分块计算注意力，可以简单理解为就是将输入张量划分成很多块，每个数据块放到 sm 里面去计算（cuda/triton 编程的核心就是在于如何将数据分块），sm 里面 L1 cache/共享内存的大小基本就决定了 这个数据块的上限空间大小，所以论文里面说的 SRAM 大小其实值的是 L1 Cache 大小，L2 Cache 是所有 SM 能共同访问的，明显不是论文里指的 SRAM。

2，可以从 GPU 内存层次角度直接看出 SRAM 是 L1 Cache (数据缓冲)。

论文 2.1 节明确都说了 A100 的 SRAM 大小是 192 KB，“As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [44, 45].” 

而英伟达官网给出的 A100 白皮书也明确说了 A100 的 L1 cache 大小是 192KB（ 组合共享内存和 L1 数据缓存），所以论文的 SRAM 肯定指的是 L1 cache 了。

另外，这个论文学习，除非有 cuda 编程经验和 transformer 模型扎实的算法基础就好容易理解点，不然真的太难懂啦，很多人其实刚开始就看了个寂寞，而且这个论文也可以说 attention 和 cuda 优化的集大成者啦。

### FlashAttention

FlashAttention-v1 其实并没有提出新的算法和网络结构上的优化，但是其在算法上综合了过往的两个创新点：**online softmax** 和 **重计算**，并将其应用于 Attention 结构，给出了详尽的数学计算、证明和 IO 复杂度分析（论文长达 34 页大头都是公式），可以说是过往 transformer 模型在 gpu 上优化的**集大成者**，而且最重要的是提供了非常易用的前向传播和反向传播的代码库，这使得其广为引用和应用于工业界。
> 可见，优秀的代码功底、扎实的理论基础、底层硬件和框架的熟悉对于科研工作非常重要，即使你没有提出新的算法，但是你的工作依然可以广为传播和应用。

本文主要分析其在模型推理阶段的优化，因此**重计算**方法的分析就略过了。

## 2. FlashAttention-v2


## 3. FlashAttention-v3


## 4. FlashDecoding & FlashDecoding++


## 参考资料

- [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [FlashAttention-2:Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://ahmdtaha.medium.com/flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness-2a0aec52ed3d)
- [Chenfan Blog-FlashAttentions](https://jcf94.com/2024/02/24/2024-02-24-flash-attention/)
- [FlashAttentions 原理及代码实现](https://jcf94.com/2024/02/24/2024-02-24-flash-attention/)
- [Self Attention 固定激活值显存分析与优化及PyTorch实现](https://zhuanlan.zhihu.com/p/445016136)
- [榨干 GPU 效能的 Flash Attention 3](https://tomohiroliu22.medium.com/%E6%A6%A8%E4%B9%BEgpu%E6%95%88%E8%83%BD%E7%9A%84flashattention%E7%AC%AC%E4%B8%89%E4%BB%A3-4a8b0a2a812e)
- [图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)
- [FlashAttention 实现算子融合原理的可视化](https://www.bilibili.com/video/BV1Zz4y1q7FX/?vd_source=69e98dbaea70afc6b62d55a86d59e408)