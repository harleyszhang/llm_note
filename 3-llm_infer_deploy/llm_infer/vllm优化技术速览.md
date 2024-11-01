- [一 PagedAttention](#一-pagedattention)
  - [1.1 PagedAttention 方案](#11-pagedattention-方案)
  - [1.2 PagedAttention 的内存共享优势](#12-pagedattention-的内存共享优势)
  - [1.3 和 TokenAttention 的区别](#13-和-tokenattention-的区别)
- [二 连续批处理](#二-连续批处理)
  - [2.1 静态批处理概述](#21-静态批处理概述)
  - [2.2 动态批处理方案](#22-动态批处理方案)
- [三 Prefix Caching: RadixAttention](#三-prefix-caching-radixattention)
- [四 服务调度策略](#四-服务调度策略)
- [参考资料](#参考资料)


vLLM 是一个快速且易于使用且大模型推理服务框架，声称有以下快速特性：
- `SOTA` 的 serving 吞吐量
- `PagedAttention` 对 kv cache 的有效管理
- 传入请求的 `continus batching`，而不是 static batching
- 借助 CUDA/HIP 图实现的高速模型运行
- 支持多种量化方案：GPTQ、AWQ、INT4、INT8 和 FP8
- 高性能 CUDA kernel，如 Flashattention
- 支持张量并行、采样并行(parallel sampling)
- 支持分块的预填充处理（Chunked prefill）

## 一 PagedAttention

PagedAttention 技术本质上是 kv cache 管理、存取技术的优化，毕竟之前的 kv cache 技术是存在显存浪费问题的。每个序列的 kv cache 大小依赖于 `seq_len`，由于 llm 都是批处理推理，而 batch 中每个序列长度和输出 tokens 数是不一样的！为了避免 kv cache 申请的内存空间不够的问题，早期 kv cache 统一按照 `max_seq_len` 来申请内存空间的，这明显导致了 decode 阶段的显存资源浪费。

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Initialize caches to store Key, Values at start. (KV Cache Implementation)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
```

### 1.1 PagedAttention 方案

> [vAttention](https://arxiv.org/pdf/2405.04437v1) 论文提出利用 CUDA 底层的虚拟内存和物理内存分配 API 直接分配连续的虚拟内存以及做物理内存映射从而避免 PagedAttention 由于要手动维护 Block Table 和物理内存分配带来的一系列工程麻烦以及高 Overhead 。另外对于新的 Attention 架构，想用上 PagedAttention，需要从 GPU Kernel 的角度去适配 PagedAttention 和重构 Attention kernel 的代码。

受操作系统中虚拟内存和分页机制启发，vLLM 提出了 PagedAttention 注意力算法，以实现 KV Cache 的**动态内存分配**，而不是像之前一样为每个 seq 都分配固定大小的 [max_seq_len, hidden_dim] 连续内存空间用于存储 kv cache。

具体来说，PagedAttention 将每个序列从逻辑上划分为一定数量的 `blocks`（块），每个 block 包含每个 seq 一定数量 tokens 的 key 和 value，并把这些逻辑 blocks 通过 block table 映射到固定大小的 物理 blocks 上，物理 blocks 可能不连续，即 kv 可能不连续分布。一句话总结就是构建 blocks 表， 并将 seq 的 kv tokens 划分成逻辑 blocks 并映射到物理 blocks 上。使用 PagedAttention 的请求的生成过程如下图所示：

<div align="center">
<img src="../../images/vllm_technology/pagedattention_workflow.gif" width="60%" alt="pagedattention_workflow">
</div>

这种方式带来的内存浪费仅出现在序列的最后一个块中，实际中带来了近乎最优的内存使用，浪费不到 4%。这种内存效率的提升大大提高了系统能够同时处理的序列数量，增加了 GPU 的利用率，并显著提升了处理吞吐量。

PagedAttention 这种结构类似于操作系统中的虚拟内存，其中将块视为页，将 tokens 视为字节，将序列视为进程。序列的逻辑连续块通过块表映射到非连续的物理块中。当新的 tokens 被生成时，这些物理块会按需分配。

### 1.2 PagedAttention 的内存共享优势

PagedAttention 还具备高效的内存共享能力。例如，在**并行采样**中，多个输出序列可以从同一个 prompt 生成。在这种情况下，prompt 的计算和内存可以在输出序列之间共享。PagedAttention 通过其块表自然地实现了内存共享，类似于进程共享物理页的方式。不同的序列可以通过将它们的逻辑块映射到相同的物理块来实现共享。为了确保共享的安全，PagedAttention 通过引用计数跟踪物理块，并实现了“写时复制”（Copy-on-Write）机制。

<div align="center">
<img src="../../images/vllm_technology/parallel_sampling.gif" width="60%" alt="并行采样示例">
</div>

PagedAttention 的内存共享显著降低了复杂采样算法（如并行采样和束搜索）的内存开销，可将其内存使用降低最多 55%，吞吐量提升最高可达 2.2 倍。这使得这些采样方法在 LLM 服务中变得更加实用。

PagedAttention 借助块表实现了灵活的内存共享机制。类似于进程间共享物理页面的方式，PagedAttention 中的不同序列可以通过将各自的逻辑块映射到相同的物理块来共享内存资源。为了确保共享的安全性，PagedAttention 跟踪物理块的引用次数，并采用写时复制策略以防止数据冲突。

<div align="center">
<img src="../../images/vllm_technology/multiple_outputs.gif" width="60%" alt="Example generation process for a request that samples multiple outputs.">
</div>

### 1.3 和 TokenAttention 的区别

**不同点**：
1. 虽然两者都是精细化管理、分配 kv 向量的技术，tokenAttention 是粒度为 token 级别的 动态 kv cache 内存管理技术，pagedattention 是 block 粒度。
2. `TokenAttention` 本质上是 `PagedAttention` 的一种特例，PagedAttention 当块大小为 1 时即为 TokenAttention。
3. PagedAttention 会造成一定程度的显存浪费，作者推算是小于 4%，而 tokenAttention 最大的优化点是确保不会浪费显存。


## 二 连续批处理

看图理解连续批处理（Continuous batching）技术原理。

### 2.1 静态批处理概述

理解静态批处理之前，先来回顾下 LLM 的推理过程：LLM 推理分为两个阶段：prefill 和 decode 阶段，严格来讲 decode 阶段才是循环迭代过程，每次循环都只生成一个 token。LLM 推理过程如下图所示：

<img src="../../images/vllm_technology/llm_infer.png" width="40%" alt="llm infer">

llm 推理迭代过程有一些特点:

1. decode 阶段的迭代计算过程是内存受限的，输入 q 只有 1 个 token，无法充分发挥 gpu 并行计算能力。
2. gpu 显存的消耗量随着模型大小和输入 token 序列长度的增加而增加。

上图是一个序列的 llm 推理过程，下面再来看下传统的一个批次的 llm 推理过程，也叫静态批处理，表现出来就是，只有当前批次完全推理完，下一个批次的序列才能进行推理。但是这有个问题是，我们之前一个批次中，序列长度不一，输出 tokens 数也不一样，即迭代结束时间不一样，那这自然会造成 gpu 利用率不高。LLM 静态批处理示意图如下所示：

<img src="../../images/vllm_technology/static_batch.png" width="70%" alt="static_batch infer">

上图显示了一个 batch_size = 4 的静态批处理过程。在第一次迭代（左侧），每个序列从提示 tokens（黄色）生成一个 token（蓝色）。经过几次迭代后（右侧），完成的各序列长度不同，因为每个序列在不同迭代中发出各自的终止 token（红色）。尽管序列 3 在两次迭代后就完成了，但静态批处理意味着之前分配的 GPU 线程资源将一直未被充分利用，直到批次中的最后一个序列完成生成（在此示例中，为六次迭代后完成的序列 2）。

很明显，静态批处理，只有当批次中的不同序列的输入输出 tokens 数完全一致时，静态批次才能实现最佳的GPU 利用率。

### 2.2 动态批处理方案

动态批处理，也叫连续批处理技术，首次提出是在 2022 年发表的论文 [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)中，它大幅提升了 llm 推理服务系统的吞吐量，可有效避免 GPU 资源的浪费。

连续批出理技术的原理是动态调整迭代过程中的批次大小，不再等待批次中的所有序列都完成生成才推理下一个批次，而是根据每次迭代序列的完成情况和当前剩余显存资源来确定当前批次大小，批次中的部分序列完成后，新的序列可以立即插入其位置，这样有效避免了 GPU 空转。

<img src="../../images/vllm_technology/continuous_batching.png" width="70%" alt="static_batch infer">

上图显示了通过连续批处理技术连续完成 7 个序列的推理情况。左图显示了第一次迭代后的批次，右图显示了几次迭代后的批次。每当一个序列发出终止 token 时，我们会将一个新的序列插入其位置（例如序列 S5、S6 和 S7），这样 GPU 无需等待所有序列完成即可开始处理新的序列，从而实现更高的 GPU 利用率。

上述的简要描述忽略了 llm 推理的 prefill（预填充）过程，因为预填充阶段和生成阶段的计算模式是不同的，所以它无法简单的与生成阶段的 tokens 一起批处理。由此，一般连续批处理框架会通过一个超参数 waiting_served_ratio 来管理此问题（实际框架不止一个超参数会有多个超参数和调度策略），该参数表示等待预填充请求与等待终止 token 请求数的比率。假设该值为 1.3，当预填充请求数/等待终止 token 请求数 > 1.3，此时推理框架会暂停批次的 decode 过程，而是去插入相应数量的新请求，并做预填充处理。

## 三 Prefix Caching: RadixAttention


## 四 服务调度策略

等待更新

## 参考资料

- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)