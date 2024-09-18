- [一 decode 阶段推理 latency 估算](#一-decode-阶段推理-latency-估算)
  - [1.1 decode latency 估算](#11-decode-latency-估算)
  - [1.2 批次大小对性能影响的分析](#12-批次大小对性能影响的分析)
    - [1.2.1 批次大小对性能影响的实验](#121-批次大小对性能影响的实验)
- [二 LLM 并发支持估算](#二-llm-并发支持估算)
- [参考资料](#参考资料)

## 一 decode 阶段推理 latency 估算

### 1.1 decode latency 估算

考虑基于 roofline 模型和的 llm decode 阶段的 latency 分析，对于小 `batch` 的模型推理，单个 token 的推理 `latency` 可能受限于 gpu 的内存带宽，即内存读取时间 > 计算时间；对于大 `batch`，单个 token 的推理 `latency` 受限于 gpu 的算力，即内存读取时间 > 计算时间。

注意，上述公式计算得到理论 `Latency` 只是个上限，我们永远不可能达到这个值，而且现实中 GPU 卡的性能也很少能达到其规格所宣称的数字。另外，本章 `Latency` 的计算忽略了**预填充阶段**中计算和**读取 kv cache 的时间、读取 umembedding vector 来计算 logits 的时间**。预填充阶段对应的就是生成第一个 `token` 的过程，这个时候需要计算 `kv cache`，所以第一个 `token` 的 `latency` （首次延时）会比后面的 `token` （decode latency）大很多。

**解码阶段的每轮 decode latency（非首字时延）的估算公式如下所示**：

1，对于小 batch size（比如为 1），计算和通信的 latency 计算公式如下（来源[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)）：
$$
\begin{align}
\text{compute} = \frac{2\cdot P}{N\cdot A_{bm}} \nonumber \\
\text{comms}  = 2\cdot 4\cdot n\cdot 8us \nonumber \\
\end{align}
$$

> 这里 $2P$ 表示权重参数所消耗的内存量，每个参数是两个字节；

2，对于大 batch（比如 512）：
$$
\begin{align}
\text{compute} = B\cdot \frac{2\cdot P}{N\cdot A_{f}} \nonumber \\
\text{comms}  = B\cdot \frac{2\cdot 4\cdot n_{layers}\cdot h}{A_c} \nonumber \\
\end{align}
$$

- $N$ 是 GPU 数目
- $n$ 是 transformer layers 数目
- $h$ embedding 向量的大小，也作 $d_{model}$
- $A_c$ 是 GPU 之间通信带宽，即卡间带宽
- $A_{bm}$ 是 GPU 内存带宽
- $A_f$ 是 GPU 算力
- $P$ 表示模型(`float16`)参数量
- $B$ 是 `batch size`

> 这里的 $2P$ 表示我们需要执行 $2P$ 次运算

举例说明，260B 模型（$n = 80, h=16484$）运行在 16张 A100 卡上。对于小批次，生成每个 token 的时间为 22 毫秒：
$$
\begin{align}
\text{compute} = \frac{2\cdot P}{N\cdot A_{bm}} \nonumber = \frac{2\cdot260e9}{16\cdot1.5e12}\simeq 0.0217 \simeq 22 \text{ms}\\
\text{comms}  = 4\cdot n_{layers}\cdot 8us \nonumber = 4\cdot 80\cdot8us = 2560us\simeq 3\text{ms}\\
\end{align}
$$
对于一个批次大小为 512 的大批次，每个 token 生成的计算总时间为 53 ms，加上卡间通信时间 18 ms，总耗时为 71 ms（即在 71 ms 内生成 512 个 token）：
$$
\begin{align}
\text{compute} = B\frac{2\cdot P}{N\cdot A_{f}} \nonumber = 512\cdot\frac{2\cdot260e9}{16\cdot 312e12}\simeq 0.053 \simeq 53 \text{ms}\\
\text{comms} = B\cdot \frac{2\cdot 4\cdot n_{layers}\cdot h}{A_c} = 512\cdot \frac{8\cdot 80\cdot 16384}{300e9} = \simeq 18\text{ms} \nonumber
\end{align}
$$
对于自回归模型的推理来说就是，**固定 seq_len**， 如果 seq_len * bs < ops:byte ratio * gpu_num，即**小 `batch_size` 范围 的 latency 不明显增加的**。


### 1.2 批次大小对性能影响的分析

通过《英伟达 GPU 性能分析指导》文档可知：算术强度通俗理解就是计算量除以访存量后的值，表示此模型/网络层在计算过程中，每 Byte 内存交换到底用于进行多少次浮点运算，单位是 FLOPs/Byte。即模型计算强度越大，其内存使用效率越高，因此应该尽可能让算法/网络层的算术强度高于 gpu 的 ops:byte ratio，这样才能充分利用 gpu 的算力。

对于 llm 的 `decode` 阶段，模型的算术强度 = $\frac{B\cdot2\cdot P}{2\cdot P} = B(批量大小)$，**即模型的算术强度和批量大小近乎成正比关系**。

假设我们使用 A100 GPU，它每秒可以执行 $312 \times 10^{12}$ 次浮点运算（flops），并且内存带宽为每秒 $1.5 \times 10^{12}$ 字节，即 A100 的操作强度为 $A_f/A_{bw}  = 208$。对于 llm 的 decode 阶段，只要批量大小大于 208，则推理处于计算受限，计算效率更高。

我们也可以通过比较内存带宽时间和计算时间，来判断是内存带宽受限还是计算 flops 受限，**模型内存访问和 decode 计算时间公式如下**:
$$
\begin{align}
\text{mem boundwidth time} = \frac{2\cdot P}{N\cdot A_{bm}} \nonumber \\
\text{flops time}  =B\cdot \frac{2\cdot P}{N\cdot A_{f}} \nonumber
\end{align}
$$

另外，假设在推理系统中我们能实现计算和 GPU 通信并行处理，则可以得到了一个更为复杂的比率：每字节通信的 flops（前面的 A100 的操作强度对应的是每字节内存访问的 flops）。以下使用张量并行的主要 layer 的通信量、计算量和每字节通信 flops 值表：

![flops_per_byte_comms](../images/transformer_latency/flops_per_byte_comms.png)

这是我们 A100 GPU 的每字节通信 flops 值。我们希望上述表格最后一行的值大于硬件的每字节 flops 计算值，这样可以确保系统保持在计算（flops）受限状态（这里先假设内存带宽不是限制因素）。对于 $d_{model} > 1024$ 的模型运行在 A100 上来说，我们的推理是安全高效的！但对于维度为 512 的情况，情况就有些不理想了。

####  1.2.1 批次大小对性能影响的实验

batch_size、内存带宽限制 vs 计算限制对 Latency 会有什么影响呢？

**自回归模型的推理实验**。**固定 seq_len=8/20**， 如果 seq_len * bs < ops:byte ratio * gpu_num，即**小 `batch_size` 范围 的 latency 不明显增加的**。从实验测试结果看，**使用 4/8 个 V100 硬件做模型推理（张量并行），输入长度固定，在 batch_size < 16/32，其 latency 不明显增加**。且有以下实验结果：

![bs_latency2](../images/transformer_latency/bs_latency2.png)
![bs_latency](../images/transformer_latency/bs_latency.png)

**`Latency` 的理论分析**：对于自回归模型的推理，默认推理配置是 `use_cache=True`，**固定 seq_len**，batch_size 较小时，模型的算术强度较低，模型受到内存带宽限制，`Latency` 取决于内存读取和 gpu 通信时间，而 `batch_size` 较小时，kv cache 读写时间和较小，而 gpu 通信时间又是固定的，故 latency 不明显增加；当 batch_size 增加到一定程度，模型的算术强度增加，模型受到算力 `FLOPS` 限制，故此时 `Latency` 与 batch_size 几乎成正比。

另外，基于上述分析和前面 decode 阶段 `mha` 层的计算量估计也可知，当 batch_size 和 output_ids_len 比较大时，**迭代生成 token 的过程中，后面 token 的 Latency 会大于前面的**。

![token latency](../images/transformer_latency/every_token_latency.png)
![token latency](../images/transformer_latency/every_token_latency2.png)

## 二 LLM 并发支持估算

以集群上的单节点 `8` 卡 `V100` 机器运行 `llama-13b` 模型为例，估算极端情况下聊天系统同时服务 10000 人并发所需要的节点数量。这里的**极端情况是指每个请求的输入长度为 512、输出长度为 1536（即上下文长度为 2048）且没有 Latency 要求**。
> LLaMA 系列模型配置文件中 "max_sequence_length": 2048, 即代表预训练的 LLaMA 模型的最大 Context Window 只有 `2048`。

k、v cache 优化中对于每个 `token` 需要存储的字节数为 $4nh^2$

1，**对于 llama-13b 模型而言， 其推理时，每个 token 大约消耗 `1MB` 的显存**（其实是 kv cache 占用的缓冲），对于输入输出上下文长度（512+1536）和为 2048 的请求，其每个请求需要的显存是 2GB。这里对每个请求所需要显存的估算是没有计算推理中间结果所消耗显存（其比较小，可忽略），另外不同框架支持张量并行所需要的额外显存也各不相同，这里暂时也忽略不计。

- 在模型权重为 `float16` 的情况下，支持的理论 batch 上限为 （32*8-24.6）/ 2 = 115.7。
- 在模型权重为 `int8` 的情况下，支持的理论 batch 上限为 （32*8-24.6/2）/ 2 = 121.85。（deepspeed 框架不支持 llama 模型的 int8 量化）

以上是理论值即上限值，float16  权重的实际 batch 数量会小于 115.7，目前的 deepspeed 框架运行模型推理时实测 `batch` 数量只可以达到  $50$ 左右。

10000/50 = 200 (台 8 卡 V100 服务器)。

实际场景中的并发请求具有稀疏性，不可能每个请求都是 `2048` 这么长的上下文长度，因此实际上 200 台 8 卡 V100 服务器能服务的并发请求数目应该远多于 10000，可能是几倍。

2，**对于 llama-65b 模型而言，其推理时，每个 token 大约消耗 `2.5MB` 的显存**，因此，极限情况下每个请求需要的显存是 5GB。
- 在模型权重为 float16 的情况下，支持的理论 batch 上限为 （32*8 - 121.6）/ 5 = 26.88。
- 在模型权重为 int8 的情况下，支持的理论 batch 上限为 （32*8 - 121.6/2）/ 5 = 39.04。（deepspeed 框架不支持 llama 模型的 int8 量化）

另外，如果输入能量化为 int8 数据类型，理论上支持的 batch 数量会翻倍。

## 参考资料
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363)