- 直译中文标题：《DeepSpeed 推理：具有定制推理内核和量化支持的多 GPU 推理》
- 原文地址：[DeepSpeed Inference: Multi-GPU inference with customized inference kernels and quantization support](https://www.deepspeed.ai/2021/03/15/inference-kernel-optimization.html)
- 译者：zhanghonggaozhang，译者对原文有所删改和优化。

## 背景知识

早期的 DeepSpeed 框架虽然支持训练大规模模型，但是训练好的模型在已有的推理解决方案上面临以下问题：

1. 缺乏对多 GPU 推理的支持以适应大模型推理的显存要求并满足 Latency 要求；
2. 在小批量推理（batch_size）时 GPU 内核性能有限
3. 很难支持量化。

为此，作者开发了 DeepSpeed Inference 系统，主要有以下三个特性：

1. 具有自适应并行性的多 GPU 推理（Multi-GPU Inference with Adaptive Parallelism）。
2. 专为推理优化的 CUDA 内核。
3. 灵活的量化感知训练以及量化推理内核。

## 一，具有自适应并行性的多 GPU 推理

优化延迟 Latency 对于推理系统是最重要，而训练则是优化吞吐量 Throughput。

使用 MP，虽然可以拆分模型且使用多个 GPU 进行并行计算来减少延迟（Latency），但是它会减小计算的粒度并增加通信，这可能会影响吞吐量（Throughput）。

## 二，用于提高 Transformer 模块计算效率的定制推理内核

为了实现高计算效率，DeepSpeed 推理通过运算符融合为 Transformer blocks 提供**定制化的推理内核**，同时考虑了多 GPU 的模型并行性。该内核融合方案与类似方法的主要区别在于：**不仅融合逐元素操作**（如偏置加法、残差和激活函数），还将广义矩阵乘法（General matrix multiply，GeMM）操作与其他操作进行合并。为了实现这一点，我们设计了一个高效的向量-矩阵或瘦矩阵-矩阵乘法的实现，这允许我们在 GeMM 操作的规约边界处融合更多的操作。

### 2.1，内核融合

我们采取了两个主要策略来融合操作：

1. 在一系列融合操作中保持输入和输出的访问模式不变；
2. 在每个 all-reduce 边界融合操作。

第一个策略确保不同的线程块在传输数据时不会遇到在流多处理器（SM）之间传递数据的情况。这是因为除了使用主内存外，`SM` 之间没有直接的通信，因为内存访问的不确定行为增加了块同步开销。第二个策略的原因是，除非在模型并行的 GPU 之间减少了部分结果，否则无法继续执行。

![推理核融合](../../images/ds_inference/inference-kernel-fusion.png)

图 1：具有 Megatron 式模型并行化 all-reduce 组件的 Transformer Layer。该图用虚线说明了融合在一起的层的部分（线的宽度表示融合深度）

图1 显示了 Transformer 层的不同组成部分以及我们在推理优化中考虑融合的算子组（groups of operations）。我们还考虑了 NVIDIA Megatron-LM 风格的并行方式，将**注意力**（Attn）和**前馈**（FF）块分区到多个 GPU上。因此，我们在 Attn 和 FF 块之后包括了两个 all-reduce 操作，它们减少了并行 GPU 之间的结果。如图 1 所示，我们在 Transformer 层内的四个主要区域进行了操作融合：

1. 输入层标准化，以及 Query、Key、Value 的 `GeMM` 和偏置相加。
2. Transform plus Attention。
3. 中间 FF、层标准化、偏置相加、残差（Residual）和高斯误差线性单元（GELU）。
4. Bias-add plus Residual。

为了融合这些操作：

1. 我们利用共享内存作为中间缓存，用于在 layer-norm 和 GeMM 中使用的缩减操作与逐元素操作之间传输数据。
2. 使用 **warp-level 指令**在线程之间传递数据来减少部分计算。
3. 使用一种新的 `GeMM` 操作调度方式，它允许根据第三次内核融合的需要融合尽可能多的操作。
4. 还通过使用隐式矩阵转换将注意力计算中的 GeMM 合并在一起，以减少内存压力。与使用 cuBLAS GeMM 的非融合计算方式相比，我们分别提高了 1.5 倍、2.9 倍、3 倍和 1.2 倍的性能。

### 2.2，通过自动内核注入从训练到推理的无缝管道

为了在推理模式下运行模型，DeepSpeed 只需要提供模型检查点的位置和所需的并行配置，即 MP/PP 度。DeepSpeed 推理内核可以为许多知名的模型架构启用，如 HuggingFace（Bert和GPT-2）或 Megatron GPT-based 模型，使用预定义的策略映射将原始参数映射到推理内核中的参数。对于其他基于 Transformer 架构的模型，用户可以指定自己的映射策略。

## 三，灵活的量化支持

为了进一步降低大规模模型的推理成本，作者创建了 DeepSpeed 量化工具包，支持灵活的量化感知训练和**用于量化推理的高性能内核**。

为了进一步降低大规模模型的推理成本，我们创建了DeepSpeed量化工具包，支持灵活的量化感知训练和用于量化推理的高性能内核。

对于训练，我们引入了一种新的方法，称为**混合量化**（Mixture of Quantization，MoQ），它受到了混合精度训练的启发，同时无缝应用量化。通过 `MoQ`，我们可以在训练的每个步骤中模拟量化对参数更新的影响，从而控制模型的精度。此外，它支持灵活的量化策略和调度，我们发现通过在训练过程中动态调整量化位数的数量，最终的量化模型在相同的压缩比下提供更高的准确性。为了适应不同的任务，MoQ还可以利用模型的二阶信息来检测其对精度的敏感性，并相应调整量化的调度和目标。

为了最大限度地提高量化模型的性能增益，我们提供了专门针对量化模型的推理内核，通过优化数据传输来降低延迟，但不需要专用的硬件。

## 四，性能测试结果

提高吞吐量，降低推理成本。

图3 显示了对应于三个 Transformer 网络（GPT-2、Turing-NLG 和 GPT-3）的三种模型大小每个 GPU 的推理吞吐量。当使用与基准模型相同的 FP16 精度时，DeepSpeed 推理将每个 GPU 的吞吐量提高了 2 到 4 倍。通过启用量化，我们进一步提升了吞吐量。

![推理吞吐量](../../images/ds_inference/inference-throughput.png)
