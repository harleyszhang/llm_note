
## FlashAttention-v1

FlashAttention-v1 其实并没有提出新的算法和网络结构上的优化，但是其综合了过往的两个创新点：**online softmax** 和 **重计算**，并将其应用于 Attention 结构，给出了详尽的数学计算、证明和 IO 复杂度分析（论文长达 34 页大头都是公式），最重要的是提供了非常易用的前向传播和反向传播的代码库，这使得其广为引用和应用于工业界。可见，优秀的代码功底、扎实的理论基础、底层硬件和框架的熟悉对于科研工作非常重要，即使你没有提出新的算法，但是你的工作依然可以广为传播和应用。

## 参考资料

- [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [FlashAttention-2:Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608)
- [FlashAttentions 原理及代码实现](https://jcf94.com/2024/02/24/2024-02-24-flash-attention/)
- [Self Attention 固定激活值显存分析与优化及PyTorch实现](https://zhuanlan.zhihu.com/p/445016136)
- [榨干 GPU 效能的 Flash Attention 3](https://tomohiroliu22.medium.com/%E6%A6%A8%E4%B9%BEgpu%E6%95%88%E8%83%BD%E7%9A%84flashattention%E7%AC%AC%E4%B8%89%E4%BB%A3-4a8b0a2a812e)
- [图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)
- [FlashAttention 实现算子融合原理的可视化](https://www.bilibili.com/video/BV1Zz4y1q7FX/?vd_source=69e98dbaea70afc6b62d55a86d59e408)