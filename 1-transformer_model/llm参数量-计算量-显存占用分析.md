- [一 KV cache 原理](#一-kv-cache-原理)
	- [1.1 背景知识](#11-背景知识)
	- [1.2 kv cache 原理](#12-kv-cache-原理)
	- [1.3 prefill 和 decode 阶段的 kv 计算过程的区别](#13-prefill-和-decode-阶段的-kv-计算过程的区别)
- [二 模型参数量](#二-模型参数量)
	- [2.1 CPU 内存使用量](#21-cpu-内存使用量)
- [三 计算量分析](#三-计算量分析)
	- [3.1 MHA(Attention) 层计算量](#31-mhaattention-层计算量)
		- [3.1.1 prefill 阶段](#311-prefill-阶段)
		- [3.1.2 decode 阶段](#312-decode-阶段)
	- [3.2 MLP 层计算量](#32-mlp-层计算量)
	- [3.3 模型总计算量](#33-模型总计算量)
		- [3.3.1 计算量定性和定量结论](#331-计算量定性和定量结论)
- [四 显存占用量分析](#四-显存占用量分析)
	- [4.1 训练过程中显存占用量计算](#41-训练过程中显存占用量计算)
	- [4.2 推理过程中显存占用量计算](#42-推理过程中显存占用量计算)
	- [4.3 显存占用计算的定性分析和定量结论](#43-显存占用计算的定性分析和定量结论)
	- [4.4 LLM 并发支持估算](#44-llm-并发支持估算)
- [结论](#结论)
- [参考资料](#参考资料)

## 一 KV cache 原理

### 1.1 背景知识

`chatgpt` 的火热引爆了 `llm`（大语言模型） 的研究和发展，`llm` 的**大**体现在两个方面：模型参数和训练数据规模，这进而带来了两个挑战：gpu 内存访问和计算效率。

目前的 llm 都是基于 transformer 模型，得益于 `GPT` 模型的成功，主流的模型架构是采用 `decoder-only` 架构的，同时模型的输出是自回归的形式，所以 gpt 这类模型也叫做 `Causal LM`。
> 因果建模模型、自回归模型、生成式 generative 模型所代表的意义几乎一致。

**本文分析的是采用 `decoder-only` 框架的 `llm`（类 `gpt` 的大语言模型）的参数量 `params`、计算量 `FLOPs`、理论所需 `CPU` 内存和 `GPU` 显存**。

这里简单介绍下 `decoder-only` 架构的 llm 结构，其只采用 `Transformer` 模型中的解码器（Decoder）部分，同时 `decoder` 结构去掉了 Encoder-Decoder attention（Decoder 中的第二个 attention），**只保留了 `Masked Self-Attention`**。

与正常的 `Attention` 允许一个位置关注/看见到它两边的 `tokens` 不同，`Masked Attention` 只让模型看到左边的 `tokens`：

![masked Self Attention](../images/transformer_params_flops/4-mask.png)
> 图： self attention vs mask self attention

这里以 `gpt1` 模型为例，其模型结构如下所示:

![decoder-only-model](../images/transformer_params_flops/decoder-only-model.png)
> gpt 模型结构，llama 在细节上会有所区别，但是主要网络层不会变。

```bash
(masked)multi_headed_attention --> layer_normalization --> MLP -->layer_normalization
```

**在计算模型参数量/计算量之前，我们先定义好一些表示符号**：

- $b$：批量大小 `batch_size`。
- $s: 输入序列长度 `seq_len`，即输入 `prompt` 字符串的长度。
- $o$: 输出 `tokens` 数量，用于计算 kv cache 的形状。
- $h$: 隐藏层的维度，也叫 $d_{model}$,即序列中每个 `token` 的 `embedding` 向量的维度。**它定义了输入和输出的特征向量的大小，也是模型内部各个组件（特别是注意力机制和前馈网络）操作的主要向量维度**。
- $V$：词表大小 `vocab_size`。也是每个 token 在做 embedding 前的 one-hot 向量维度。
- $n$：模型中 decoder layers 层数，对应 hf 模型配置文件中的 num_hidden_layers。

这些变量值都可以在模型配置文件中找到，以 `llama-13b` 模型配置文件为例，主要字段解释如下：
![llama-13b-config](../images/transformer_params_flops/llama-13b-config.png)

- `vocab_size`：词汇表中标记的数量，也是嵌入矩阵的第一个维度。
- `hidden_​​size`：模型的隐藏层大小，其实就是 $d_\text{model}$。
- `num_attention_heads`：模型的多头注意力层中使用的**注意力头数量**。
- `num_hidden_layers`：模型中的块数（层数）, number of layers。
- `max_sequence_length`: $2048$, 即代表预训练的 LLaMA 模型的最大 Context Window 只有 $2048$。

注意，很多 `decoder-only` 架构的自回归模型的全连接层的偏置 `bias` 都设置为 False，故这里的计算公式中没有考虑偏置参数。

![LlamaForCausalLM architecture](../images/transformer_params_flops/llama-model-params.png)

### 1.2 kv cache 原理

**背景**：生成式模型的推理过程很有特点，**推理生成 `tokens` 的过程是迭代式的**。简单来说就是，用户给一个输入文本，模型会输出一个回答（长度为 $N$），但该过程中实际执行了 $N$ 次模型前向传播过程。即 `GPT` 类模型一次推理只输出一个 `token`，**当前轮输出的 token 与之前输入 tokens 拼接，并作为下一轮的输入 tokens**，反复多次直到遇到终止符 `EOS` 或生成的 `token` 数目达到设置的 `max_new_token` 才会停止。

可以看出第 $i$ 轮输入数据只比第 $i+1$ 轮输入数据新增了一个 `token`，其他全部相同！因此第 $i+1$ 轮推理时必然包含了第 $i$ 轮的部分计算。`KV Cache` 优化的起因就在这里，**缓存当前轮可重复利用的计算结果**，下一轮计算时直接读取缓存结果，原理很简单，**本质就是用空间换时间**。

另外，**每一层 decode layer 都需要单独缓存 $K$ 和 $V$，因为每层的 `attention` 运算是独立的，即第 $L$ 层的 $K_L$ 和 $V_L$ 是独立的、与其他层不同的**。如果不缓存每一层的 $K$ 和 $V$，在生成下一个 token 时，模型就需要重新计算之前所有 `token` 的 $K$ 和 $V$，这将导致大量冗余计算，通过缓存，避免了重复计算 $K$ 和 $V$，从而加速了生成过程。

单个 `token` 的 `kv` 向量的计算开销，大概占将 `token` 通过整个模型前向计算的 $1/6 = \frac{4nh^2}{24nh^2}$ ，表面上看，使用 kv cache 优化可以节省 1/6 的时间，但实际不是！随着采样的迭代进行，token 数量增加，这个节省的量会随着序列中的 token 数量增加而增加（非常多！），假设输出 tokens 数目为 $o$，即 $o$ 轮迭代 decode 下来，可节省$1/6o$的时间，因此，使用 kv cache 优化是非常有必要的。

### 1.3 prefill 和 decode 阶段的 kv 计算过程的区别

一个典型的自回归模型的生成式推理过程包含了两个阶段：

1. **预填充阶段**（prefill phase）：输入一个 prompt 序列，为每个 transformer 层生成 key cache 和 value cache（KV cache）。这个阶段可以充分利用并行计算的优势。
2. **解码阶段**（decoding phase）：使用并更新 KV cache，一个接一个地生成词（**无并行性**），当前生成的词依赖于之前已经生成的词。该阶段的推理计算分两部分：**更新 KV cache 和计算 decoder layers 的输出**。

这两个阶段的差别在于 $Q$ 的维度不同。在 `prefill` 阶段时，用户输入的所有 token 都需要参与运算，所以此时 $Q$ 的形状为 $[b, s, h]$。在第 `decode` 阶段时，只有新生成的 `token` 作为第二次迭代过程的输入，所以此时 $Q$ 的维度为 $[b, 1, h]$，即**只有新 token 作为 Q**。

**在预填充阶段，多头注意力（`MHA`）模块生成 `KV` 键值对并存储在 KV 缓存中**。设输入到 Transformer 层的输入为 $X_{pre}\in \mathbb{R}^{s\times h}$，其中 $h$ 是隐藏维度，$s$ 是提示词 token 序列的长度。`MHA` 模块的 $4$ 个线性层权重用 $W_q$，$W_k$，$W_v$ 和 $W_o$ 表示。查询、键和值（Q、K、V）的计算过程如下：

$$\text{Query}: Q_{pre}=X_{pre} * W_{q} \\
\text{Key}: K_{pre}=X_{pre} * W_{k} \\
\text{Value}: V_{pre}=X_{pre} * W_{v}$$

生成的 $K_{pre}$ 和 $V_{pre}$ 被存储在 KV 缓存中，每个 transformer layer 都独立的存储 KV 键值对。其余的 `MHA` 计算如下：

$$O_{pre }=\text{softmax}(\frac{Q_{pre } * K_{pre }^{T}}{\sqrt{d_k}}) * V_{pre } * W_{o}+X_{pre }$$
> $d_k = h/n_{heads}$。

MHA 的输出 $O_{pre }\in \mathbb{R}^{s\times h}$ 将传递到 MLP。MLP 的输出作为下一层 Transformer 层的输入。

解码阶段是 LLM 推理的关键部分。在这一阶段，模型使用预填充阶段生成的 **KV 缓存**，同时逐步添加新信息。目标是**逐步迭代的生成新 token**，每个新 token 的生成都会参考之前生成的 token。

**在解码阶段**，MHA 加载先前存储的 KV 缓存 $K_{cache}$ 和 $V_{cache}$。输入为 $X_{dec}\in \mathbb{R}^{1\times h}$。新的键值对被计算并拼接到现有缓存：

$$\text{Query}: Q_{dec}=X_{dec}*W_{q} \\
\text{Key}: K_{cat }=[K_{cache }, X_{dec } * W_{k}] \\
\text{Value}: V_{cat }=[V_{cache }, X_{dec } * W_{v}]$$

新计算的 $X_{dec}\cdot W_{k}$ 和 $X_{dec}\cdot W_{v}$ 紧接着被拼接到 $KV$ 缓存得到新的 $K_{cat}$、$V_{cat}$ 向量。MHA 中的剩下的计算如下进行：

$$O_{dec}=\text{softmax}(\frac{Q_{dec}\cdot K_{cat}^{T}}{\sqrt{d_k}}) * V_{cat } * W_{o}+X_{dec}$$

其中 MHA 的输出 $O_{dec}\in \mathbb{R}^{1\times h}$ 被传递到 MLP。最后一个 Transformer 层的输出被发送到最终的预测层，以预测下一个 token 。

## 二 模型参数量

模型由 $N$ 个相同的 `decoder block` 串联而成，每个 `decoder block` 又由 `1` 个带掩码（`mask`）多头注意力（MHA）层、`1` 个前馈神经网络（FFN）层和 `2` 个层归一化层组成。
> 这里不单独计算每个 self-attention 层的参数量了，毕竟实际代码中，其都是在一个矩阵中。另外 `llama` 模型的 `MLP` 块虽然有 3 个线性层，但其参数量和计算量和 `gpt1` 是一样的。

1，`MHA` 块有 $4$ 个线性层（全连接层/映射层），对应的是 $Q$、$K$、$V$ 和输出映射层的权重矩阵 $W_Q,W_K,W_V,W_o \in \mathbb{R}^{h\times h}$ 及其偏置。$4$ 个线性层权重参数形状都为 $[h,h]$，偏置形状为 $[h]$。**`MHA` 块的参数量 = $4h^2 + 4h$**

2，`MLP/FFN` 块由 $2$ 个线性层组成，一般第一个线性层完成 $h$ 到 $4h$ 的升维，第二个将 $4h$ 降维到 $h$。对应权重矩阵为 $W_1\in \mathbb{R}^{h\times 4h}$, $W_2 \in \mathbb{R}^{4h\times h}$，偏置形状分别为 $4h$ 和 $h$。**`MLP` 块的参数量 = $8h^2 + 5h$**。

3，`LN` 层有两个，分别连接在 `MHA` 和 `MLP` 块的后面，`layer norm` 层有两个可训练参数: $\mu_{\beta}$ 和 $\sigma_{\beta}$（scale factor and offset），参数大小都是 $[h]$。**$2$ 个 `Layer Norm` 层的总参数量 = $4h$**。

4，除了 `decoder block` 有很多参数，`Embedding` 层同样也有参数，`Embedding` 层包括两部分: Token Embedding (`TE`) 和 Positional Embedding (`PE`)。`TE` 层的输入张量形状是 $[b, s, V]$，输出维度是 $[b, s, h]$，对应的 `TE` 层权重矩阵形状为 $V, h$，**即 `TE` 层参数量 = $Vh$**。另外，最后的输出层通常是和 `TE` 层共享权重矩阵的。

位置 Embedding 层的参数量比较小，有时可忽略不计。

综上可知，**参数量和输入序列长度无关。对于有 $n$ 层 `decode block` 块的 `llm` 参数量为 $n(12h^2 + 13h) + Vh$。当 $h$ 较大时，可忽略一次项，`llm` 参数量近似为 $12nh^2$**。

不同版本 `LLaMA` 模型的参数量估算如下：

| 实际参数量 | 隐藏维度 h | 层数 n | heads 数目| 预估参数量 12nh^2 |
| :--------: | :--------: | :----: | ------- | :---------------: |
|    6.7B    |    4096    |   32   | 32      |   6,442,450,944   |
|   13.0B    |    5120    |   40   | 40      |  12,582,912,000   |
|   32.5B    |    6656    |   60   | 52      |  31,897,681,920   |
|   65.2B    |    8192    |   80   | 64      |  64,424,509,440   |

> 该章节主要参考资料 [Transformer Deep Dive: Parameter Counting](https://orenleung.com/transformer-parameter-counting)

另外，推特上有人研究了 `gpt-like` 模型（`opt`）的参数分布，下面是不同大小模型的一些图。可以看出，随着模型变大，`MLP` 和 `Attention` 层参数量占比越来越大，最后分别接近 `66%` 和 `33%`。这个比例可以通过上面的公式推测出来，估算公式:

$$\frac{8nh^2}{12nh^2} = 2/3\cong 66\% \\
\frac{4nh^2}{12nh^2} = 1/3\cong 33\%$$

![gpt-like 模型（`opt`）的参数分布](../images/transformer_params_flops/opt-prams-dist.png)

### 2.1 CPU 内存使用量

1，模型参数内存如何计算？

- 对 `int8` 而言，模型参数内存 = 参数量 *（1字节/参数），单位字节数
- 对 `fp16` 和 `bf16` 而言，模型参数内存 = 参数量 *（2 字节/参数）

`llm` 模型一般都是保存为 `fp16` 或者 `bf16` 格式，**以 `llama13b` 为例，$1\text{B} = 10^9 \text{byte} \simeq 1\text{GB}$，可知 `llam13b` 模型权重参数文件占用的存储空间是 `26GB` 左右**。

2，模型推理需要的总	`cpu` 内存是多少？

推理总内存 ≈ 1.2 × 模型参数内存（20% 是经验，不同框架可能不一样）

## 三 计算量分析

`FLOPs`：floating point operations 指的是浮点运算次数，一般特指乘加运算次数，**理解为计算量**，可以用来衡量算法/模型时间的复杂度。

对于矩阵 $A\in\mathbb{R}^{1\times n}$ 和 $B \in \mathbb{R}^{n\times 1}$ 的矩阵乘法的 FLOPs 为 $2n$；**对于矩阵 $A \in \mathbb{R}^{m\times n}$ 和 $B\in\mathbb{R}^{n\times p}$ 的矩阵乘法的 `FLOPs` 为 $2mnp$**。

`Pytorch` 实现线性层的函数为 `nn.Linear(in_features, out_features, bias=True)`，其中线性层权重的的维度大小是 $[下一层的维数/out_{features}，前一层的维数/in_{features}]$，对应的计算公式为:

$$y = xW^T + \text{bias}$$

线性层（全连接层/映射层）的 `FLOPs` 计算：假设 $I$ 是输入层的维度，$O$ 是输出层的维度，对应全连接层（线性层）的权重参数矩阵维度为 $[I, O]$。

- 不考虑 bias，全连接层的 $FLOPs = (I + I -1) \times O = (2I − 1)O$
- 考虑 bias，全连接层的 $FLOPs = (I + I -1) \times O + O = (2\times I)\times O$

对于 transformer 模型来说，其计算量**主要**来自 `MHA` 层和 `FFN` 层中的矩阵乘法运算。先考虑 `batch_size = 1` 和 输入序列长度为 $s$ 的情况。

### 3.1 MHA(Attention) 层计算量

对于 Attention 层，输入输出矩阵 `QKVO` 大小一模一样，形状都是 $[s,h]$。
#### 3.1.1 prefill 阶段

先分析 `MHA` 块的计算量：

1, **计算 Q、K、V**：对输入矩阵做线性变换，输入 `tokens` 序列的 **`embedding` 向量**的形状为 $[s, h]$，做线性变换的权重矩阵 $W_Q$、$W_K$、$W_V$ $\in \mathbb{R}^{h\times h}$，矩阵乘法的输入和输出形状为: $[s,h] \times [h,h]\to [s,h]$，`FLOPs`: $3* 2sh^2 = 6sh^2$。

2, **Self-Attention 层**，`MHA` 包含 `heads` 数目的 `Self-Attention` 层，这里直接分析所有 `Self-Attention` 层的 `FLOPs`:
- **$QK^T$ 打分计算**：每个头需要计算 Query 和 Key 的点积，所有头的 $QK^T$ 矩阵乘法的输入和输出形状为: $[s,h] \times [h,s]\to [s,s]$，`FLOPs`: $2s^2h$。
- **应用注意力权重**：计算在 $V$ 上的加权 $score\cdot V$，矩阵乘法的输入输出形状: $[s,s] \times [s,h]\to [s,h]$，`FLOPs`: $2s^2h$。

**`Scale Dot Product Attention` 层内部只估算两个矩阵乘法的计算量**，`attention_scale`（$/\sqrt(k)$）、`attn_softmax` ($\text{softmax}$) 的计算量忽略不计，因为这两个小算子都是逐元素操作。

3, **多头拼接和线性映射**：所有注意力头输出拼接后通过线性映射，`concat` 不涉及数学运算，只涉及内存操作。矩阵乘法的输入和输出形状为: $[s,h] \times [h,h]\to [s,h]$，**attention 后的线性映射的 `FLOPs`: $2sh^2$**。

**综上，prefill 阶段 `MHA` 块的 `FLOPs`: $6sh^2 + 4s^2h + 2sh^2 = 8sh^2 + 4s^2h$**

#### 3.1.2 decode 阶段

1，**计算 Q、K、V**：每个 token 的 embedding 向量 $t_e \in \mathbb{R}^{1\times h}$，对应的，3 个矩阵乘法的输入和输出形状为: $[1,h] \times [h,h]\to [1,h]$，`FLOPs`: $3*2h^2 = 6h^2$。

2，**Self-Attention 层**：
- $QK^T$：矩阵乘法的输入输出形状为: $[1, h] \times [h, s+o]\to [1,s+o]$，`FLOPs`: $2h(s+o)$。
- $\text{score}\cdot V$: 矩阵乘法的输入输出形状为: $[1, s+o] \times [s+o, h]\to [1, h]$，`FLOPs`: $2h(s+o)$。

通过上述两个公式，可以看出随着输出 `token` 的增加，计算量也随之线性增加，这也是我们在 llm 推理时观察到的越到后生成 `token` 越慢的原因。
> 在实际代码中，对于每一轮解码的 flops 上述公式有时等效于: $2sh$？

3，输出线性映射层: 矩阵乘法 `matmul` 的输入输出形状为: $[1, h] \times [h, h]\to [1, h]$，`FLOPs`: $2h^2$。

**综上，decode 阶段 `MHA` 层每一轮解码的 `FLOPs`: $6h^2 + 4(s+o)h + 2h^2= 8h^2 + 4(s+o)h$**。

### 3.2 MLP 层计算量

先分析 `prefill` 阶段 `Feed-forward`（MLP/FFN）层的计算量分析。包含两个线性层，以及一个 `relu` 激活层（逐元素操作，flops 很小$=5\cdot 4h$，可忽略）。`MLP` 两个线性层的权重参数矩阵: $W_1 \in \mathbb{R}^{h\times 4h}$、$W_2 \in \mathbb{R}^{4h\times h}$，`MLP` 的输入矩阵: $\in \mathbb{R}^{s\times h}$。

1. 第一个线性层，线性层对应矩阵乘法的输入和输出形状为 $[s,h] \times [h,4h]\to[s,4h]$，`FLOPs` 为 $8sh^2$
2. 第二个线性层，矩阵乘法的输入和输出形状为 矩阵乘法的输入和输出形状为 $[s,4h] \times [4h, h]\to [s,h]$，`FLOPs` 为 $8sh^2$

**因此，`prefill` 阶段 `FFN` 层的 `FLOPs`: $2*8sh^2 = 16sh^2$**。

值得注意的是，除了 `MHA` 层的 `FLOPs` 计算公式区分 `prefill` 和 `decode` 阶段，其他层只需要将 `prefill` 阶段的计算公式中的 $s$ 设置为 $1$。即对于 `decode` 阶段的 `FFN` 块的 `FLOPs = 16h^2`

### 3.3 模型总计算量

除了 `MHA`、`MLP` 块的计算量之外：

- `Embedding` 层只是一个查找表，没有进行显式的乘法运算，因此严格来说，Embedding 层本身不会产生 `FLOPs`，但可以通过其输出维度来推导其他层的 `FLOPs`。
- `LayerNorm` 操作是**逐元素**进行的，因此不存在通用的公式来。`LayerNorm` 层的两个权重都是一个长度为 $h$ 的向量，`FLOPs` 可以预估为: $2h$，但**通常忽略不计**。
- 最后的输出层（线性层）的**将隐藏向量映射为词表大小，得到每个 token 对应的 logits 向量**。线性层的权重矩阵为：$W_{last} \in \mathbb{R}^{h\times V}$，矩阵乘法的输入和输出形状为: $[s, h] \times [h, V] -> [s, V]$。`FLOPs`: $2shV$。

综上分析可知，$n$ 层 `decoder block/layer` 的总计算量大约为: $n(8sh^2 + 4s^2h + 16sh^2) = 24nh^2s + 4nhs^2$。而在输入数据形状为 $[b, s]$ 的情况下，一次训练/推理：

1，`prefill` 阶段总的计算量：

$$b\times (24nh^2s + 4nhs^2) + 2bshV) = 24nh^2*bs + 4nhbs^2 + 2bshV$$

2，`decode` 阶段每轮的计算量：

$$b\times (8nh^2 + 4nh(s+o) + 16nh^2) + 2bhV = 2 4nh^2*b + 4nhb(s+o) + 2bshV$$
> 关于 llm flops 的估算，其实还有一个很简单的方法，就是**直接估算每个 token 的 flops 且只分析 qkv和输出层的矩阵计算，以及 mlp 层的矩阵计算**，这种分析过程更简单，可以直接得到每个 token 的对应的计算量为 $8nh^2 + 16nh^2 = 24nh^2$。

#### 3.3.1 计算量定性和定量结论

**当隐藏维度 $h$ 比较大，且远大于序列长度 $s$ 时，则可以忽略一次项**：
1. `prefill` 阶段的计算量 `FLOPs` 可以近似为 $24nh^2*bs$。
2. `decode` 阶段每轮 `forward` 的计算量为 $24nh^2*b$，模型参数量为 $12nh^2$；
> 每个 token 对应的计算量为 $24nh^2$。

因为，输入的 `tokens` 总数为 $bs$（即上下文总长度），即对于一个 `token` 存在等式: $\frac{24nh^2}{12nh^2} = 2$。所以，我们可以近似认为：**在一次前向传播中，对于每个 `token` 和 每个模型参数，需要进行 $2$ 次浮点数运算，即一次乘法法运算和一次加法运算**。
> 实际会有不到 `2%` 的误差，主要是因为我们忽略了一些小算子的计算量。

一次迭代训练包含了前向传递和后向传递，后向传递的计算量是前向传递的 `2` 倍。因此，前向传递 + 后向传递的系数 $=1 + 2 = 3$ 。**即一次迭代训练中，对于每个 token 和 每个模型参数，需要进行 $6$ 次浮点数运算**。

有了上述训练和推理过程中计算量与参数量关系的结论。接下来，我们就可以估计一次迭代训练 `GPT3-13B` 所需要的计算量。对于 GPT3，每个 token，每个参数进行了 $6$ 次浮点数运算，再乘以参数量和总 `tokens`数就得到了总的计算量。GPT3 的模型参数量为 12850M，训练数据量 300B tokens。

$$6 \times 12850 \times 10^6 \times 300 \times 10^9 = 2.313 \times 10^{22}$$

计算结果和下表所示结果相符合。

![llm_params_flops](../images/transformer_params_flops/llm_params_flops.png)
> 估算训练一个 transformer 模型所需的算力成本的公式可参考文章[Transformer 估算 101](https://mp.weixin.qq.com/s/MFgTUDAOODgMDb59eZC9Cw)。本章主要参考 [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/) 以及 [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)。

这个表总结了常见大型语言模型（LLM）的**参数数量、序列长度、批次大小、隐藏层大小、层数和每次前向推理的浮点操作数总量（FLOPs）**，`FLOPs` 以 T（万亿）为单位。
| Model           | Parameters | Sequence Length | Batch Size | Hidden Size | Number of Layers | FLOPs (prefill)         |
|-----------------|------------|-----------------|------------|-------------|------------------|----------------------------------|
| GPT-3 (175B)    | 175B       | 2048            | 8          | 12288       | 96               | ~7.0 × 10³ T FLOPs                |
| GPT-3 (13B)     | 13B        | 2048            | 8          | 4096        | 40               | ~4.4 × 10² T FLOPs                |
| BERT-Large      | 345M       | 512             | 8          | 1024        | 24               | ~2.4 × 10¹ T FLOPs                |
| T5-11B          | 11B        | 512             | 8          | 1024        | 24               | ~1.4 × 10² T FLOPs                |
| LLaMA-13B       | 13B        | 2048            | 8          | 5120        | 40               | ~4.4 × 10² T FLOPs                |
| PaLM-540B       | 540B       | 2048            | 8          | 16384       | 96               | ~6.7 × 10⁴ T FLOPs                |
| ChatGPT (GPT-4) | 175B       | 2048            | 8          | 12288       | 96               | ~7.0 × 10³ T FLOPs                |

## 四 显存占用量分析

### 4.1 训练过程中显存占用量计算

**中间激活**：前向传播计算过程中，前一层的输出就是后一层的输入，**相邻两层的中间结果也是需要 gpu 显存来保存的**，中间结果变量也叫激活内存，值相对很小。

**在模型训练过程中，设备内存中除了需要模型权重之外，还需要存储中间变量（激活）、梯度和化器状态动量**，后者显存占用量与 `batch size` 成正比。

$$训练总内存 = 模型内存 + 优化器内存 + 中间激活内存 + 梯度内存$$

在模型训练过程中，**存储前向传播的所有中间变量（激活）结果**，称为 `memory_activations`，用以在反向传播过程中计算梯度时使用。而模型中梯度的数量通常等于中间变量的数量，所以 `memory_activations = memory_gradients`。

假设 `memory_modal` 是指存储模型所有参数所需的内存、`memory_optimizer` 是优化器状态变量所需内存。综上，模型训练过程中，显存占用量的理论计算公式为：

```bash
total_memory = memory_modal + 2 * memory_activations + memory_optimizer
```

值得注意的是，**对于 LLM 训练而言，现代 GPU 通常受限于内存瓶颈，而不是算力**。因此，**激活重计算** (`activation recomputation`，或称为激活检查点 (`activation checkpointing`) ) 就成为一种非常流行的**以计算换内存**的方法。

**激活重计算**主要的做法是**重新计算某些层的激活而不是把它们存在 GPU 内存中，从而减少内存的使用量**，内存的减少量取决于我们选择清除哪些层的激活。

### 4.2 推理过程中显存占用量计算

深度学习模型推理任务中，占用 GPU 显存的主要包括三个部分：**模型权重、输入输出以及中间激活结果**。（该结论来源[论文](https://www.usenix.org/conference/osdi20/presentation/gujarati)）因此，LLM 显存占用可分为 3 部分：

1，存储模型权重参数所需的显存计算公式（`params` 是模型参数量，参数类型为 `fp16`）：

$$\text{memory\_model} = \text{params} * 2 = [n(12h^2 + 4h) + Vh] * 2$$

2，中间激活显存占用（额外开销）

和模型训练需要存储前向传播过程中的中间变量结果不同，**模型推理过程中并不需要存储中间变量**，因此推理过程中涉及到的**中间结果**内存会很小（中间结果用完就会释放掉），一般指**相邻两层的中间结果**或者算子内部的中间结果，这里我们只考虑主要算子中最大的中间结果部分即可。

这里我们假设其占用的显存为 `memory_intermediate`，`heads` 数量用符号 $n_\text{head}$ 表示，假设输入数据的形状为 $[b,s]$。
- 每个 self-attention 头需要计算 Query 和 Key 的点积，每个头的 $QK^T$ 矩阵乘法的输入输出形状为 $[b, head\_num, s, h//n\_head] \times [b, head\_num, h//n\_head, s] \rightarrow [b, head\_num, s, s]$，所以占用显存大小为 $2bs^2n_{head}$；
- `mlp` 块中，第一个线性层的输出结果形状为 $[b, s, 4h]$，所以占用显存大小为 $8bsh$。

计算 `MHA`和 `MLP` 的 `memory_intermediate` 的伪代码如下:

```bash
memory_intermediate of attention(qk^t output) = 2 * batch_size * n_head * square_of(sequence_length)
memory_intermediate of mlp(fc1 layer1 output) = 2 * batch_size * s * 4h
```

又因为一般 $h \gg s$，所以 `memory_intermediate of mlp` 远大于 `memory_intermediate of attention`。所以:

$$\text{memory\_intermediate} = 8bsh$$

值得注意的是，根据经验，在模型实际前向传播过程中产生的这些额外开销（中间激活）通常控制在总模型参数内存的 20% 以内（只有 80% 的有效利用率）。

3，`kv cache` 显存占用

`LLM` 推理优化中 `kv cache` 是常见的方法，本质是用空间换时间。假设输入序列的长度为 $s$ ，输出序列的长度为 $o$，decoder layers 数目为 $n$，以 `float16` 来保存 `KV cache`，那么 `KV cache` 的峰值显存占用计算公式为:

$$\text{memory\_kv-cache} = 2*2*nh*b(s+o) = 4nh*b(s+o)$$

上式，第一个 `2` 表示 K/V cache，第二个 `2`表示 float16 占 2 个 bytes。**每个 token 的 kv 缓冲大小 $ = 4nh$，单位为字节 `byte`**。

综上分析可知，llm 推理时，gpu 显存占用主要是：模型权重和 kv cahce，**总显存消耗计算如下**:

$$\begin{aligned}\text{inference\_memory} &\simeq [n(12h^2 + 13h) + Vh]*2 + 8bsh + 4nhb(s+o) \\
&\simeq 1.2 \cdot 24nh^2 + 4nhb(s+o)\end{aligned}$$
> 模型推理时，中间激活最大不会超过模型权重参数内存的 20%。当 $h$ 较大时，忽律掉一次项。

中间激活和 `kv cache` 显存和批次大小 $b$ 以及序列长度 $s$ 成正比，**在 bs > 某个阈值时，占推理显存大头的是 kv cache**。以 llama13b 为例分析，权重参数占用 26GB，当 b = 64, s = 512 时，输出序列长度 o = 512, kv cache 显存占用 = $4nhb(s+o) = 42,949,672,960\ bytes \simeq 42GB$，是模型参数显存的 1.6 倍。

> `b` 的增加能带来近乎线性的 `throughput` 增加，llm 服务模块的调度策略就是动态调整批次大小，并尽可能让它最大。

### 4.3 显存占用计算的定性分析和定量结论

1. 模型推理阶段，当输入输出上下文长度之和比较小的时候，占用显存的大头主要是模型参数，但是当输入输出上下文长度之和很大的时候，占用显存的大头主要是 `kv cache`。
2. 每个 `GPU` `kv cache` 显存所消耗的量和**输入 + 输出序列长度**成正比，和 `batch_size` 也成正比。
3. 有[文档](https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model)指出，`13B` 的 `LLM` 推理时，每个 `token` 大约消耗 `1MB` 的显存。

以 A100-40G GPU 为例，llama-13b 模型参数占用了 26GB，那么剩下的 14GB 显存中大约可以容纳 14,000 个 token。在部署项目中，如果将输入序列长度限制为 512，那么该硬件下最多只能同时处理大约 `28` 个序列。

### 4.4 LLM 并发支持估算

以集群上的单节点 `8` 卡 `V100` 机器运行 `llama-13b` 模型为例，估算极端情况下聊天系统同时服务 10000 人并发所需要的节点数量。这里的**极端情况是指每个请求的输入长度为 512、输出长度为 1536（即上下文长度为 2048）且没有 latency 要求**。
> LLaMA 系列模型配置文件中 "max_sequence_length": 2048, 即代表预训练的 LLaMA 模型的最大 Context Window 只有 `2048`。

结合前面的显存分析章节可知，k、v cache 优化中对于每个 `token` 需要存储的字节数为 $4nh^2$

1，**对于 llama-13b 模型而言， 其推理时，每个 token 大约消耗 `1MB` 的显存**（其实是 kv cache 占用的缓冲），对于输入输出上下文长度（512+1536）和为 2048 的请求，其每个请求需要的显存是 2GB。这里对每个请求所需要显存的估算是没有计算推理中间结果所消耗显存（其比较小，可忽略），另外不同框架支持张量并行所需要的额外显存也各不相同，这里暂时也忽略不计。

- 在模型权重为 `float16` 的情况下，支持的理论 batch 上限为 （32*8-24.6）/ 2 = 115.7。
- 在模型权重为 `int8` 的情况下，支持的理论 batch 上限为 （32*8-24.6/2）/ 2 = 121.85。（deepspeed 框架不支持 llama 模型的 int8 量化）

以上是理论值即上限值，float16  权重的实际 batch 数量会小于 115.7，目前的 deepspeed 框架运行模型推理时实测 `batch` 数量只可以达到  $50$ 左右。

10000/50 = 200 (台 8 卡 V100 服务器)。

实际场景中的并发请求具有稀疏性，不可能每个请求都是 `2048` 这么长的上下文长度，因此实际上 200 台 8 卡 V100 服务器能服务的并发请求数目应该远多于 10000，可能是几倍。

2，**对于 llama-65b 模型而言，其推理时，每个 token 大约消耗 `2.5MB`（估算的 $4nh = 4*80*8192 / (1024*1024) = 2.5 \text{MB}$）的显存**，因此，极限情况下每个请求需要的显存是 5GB。
- 在模型权重为 float16 的情况下，支持的理论 batch 上限为 （32*8 - 121.6）/ 5 = 26.88。
- 在模型权重为 int8 的情况下，支持的理论 batch 上限为 （32*8 - 121.6/2）/ 5 = 39.04。（deepspeed 框架不支持 llama 模型的 int8 量化）

另外，如果输入能量化为 int8 数据类型，理论上支持的 batch 数量会翻倍。

## 结论

对于典型自回归 `llm`，假设 decoder layers 层数为 $n$，隐藏层大小（Embedding 向量维度）为 $h$，输入输入数据形状为 $[b,s]$。当隐藏维度 $h$ 比较大，且远大于序列长度 $s$ 时，则参数量和计算量的估算都可以忽略一次项，则有以下关于参数量、计算量和显存占用计算分析结论。

**一些定性结论：**
1. 参数量和输入序列长度无关。$\text{Parmas} = 12nh^2$。
2. 每个 `token` 对应的 $\text{Flops} = 24nh^2$，计算量随序列长度呈线性增长。其中 $\text{Prefill flops} = 24nh^2*bs$；$\text{Decode flops} = 24nh^2*b$。
3. 每个 `token` 消耗的 $ \text{memory} = 4nh$，`kv cache` 显存占用量随（输入 + 输出序列长度）以及批量大小 `batch_size` 呈线性增长。kv cache 显存占用量，即 $\text{memory\_kv-cache} = b(s+o)h*n * 2*2 = 4nh*b(s+o)$，单位为字节 `byte`。

**定量结论（近似估算）：**
1. 一次迭代训练中，对于每个 token 和 每个模型参数，需要进行 6 次浮点数运算。
2. 随着模型变大，`MLP` 和 `Attention` 层参数量占比越来越大，最后分别接近 `66%` 和 `33%`。
3. 有[文档](https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model)指出，`13B` 的 `LLM` 推理时，每个 `token` 大约消耗 `1MB` 的显存。



## 参考资料

1. [Transformer Deep Dive: Parameter Counting](https://orenleung.com/transformer-parameter-counting)
2. [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
3. [Estimating memory requirements of transformer networks](https://www.linkedin.com/pulse/estimating-memory-requirements-transformer-networks-schartz-rehan/?trackingId=q8AzwkgCSK6DhhcafunTgA%3D%3D)
4. [Formula to compute approximate memory requirements of Transformer models](https://stats.stackexchange.com/questions/563919/formula-to-compute-approximate-memory-requirements-of-transformer-models)
5.  [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
6.  [如何估算transformer模型的显存大小](https://avoid.overfit.cn/post/6724eec842b740d482f73386b1b8b012)
7.  [大模型推理性能优化之KV Cache解读](https://zhuanlan.zhihu.com/p/630832593)
8.  [如何生成文本: 通过 Transformers 用不同的解码方法生成文本](https://huggingface.co/blog/zh/how-to-generate)
9.  [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
