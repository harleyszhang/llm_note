- [RoPE 算法推导](#rope-算法推导)
  - [PE 和 Self-Attention 概述](#pe-和-self-attention-概述)
  - [2D 的 RoPE 算法](#2d-的-rope-算法)
  - [多维的 RoPE 算法](#多维的-rope-算法)
- [RoPE 代码实现](#rope-代码实现)
- [参考资料](#参考资料)

旋转位置编码（Rotary Position Embedding，RoPE）是论文 Roformer: Enhanced Transformer With Rotray Position Embedding 提出的一种能够**将相对位置信息依赖集成到 self-attention 中**并提升 transformer 架构性能的位置编码方式。

和相对位置编码相比，RoPE 具有更好的外推性，目前是大模型相对位置编码中应用最广的方式之一。这里的外推性实质是一个**训练和预测的文本长度不一致的问题**。具体来说，不一致的地方有两点：
1. 预测的时候用到了没训练过的位置编码（不管绝对还是相对）；
2. 预测的时候注意力机制所处理的 token 数量远超训练时的数量。

## RoPE 算法推导

### PE 和 Self-Attention 概述

设 $q_m$ 表示第 $m$ 个 `token` 对应的词向量 $x_m$ 集成**位置信息** $m$ 之后的 $query$ 向量；$k_n$ 和 $v_n$ 则表示词向量 $x_n$ 集成其位置信息 $n$（第 $n$ 个 `token`）之后的 `key` 和 `value` 向量，$q_m、k_n、v_n$ 的表达用如下公式:

$$q_m = f_q(x_m, m)  \tag{1} \\
k_n = f_k(x_n, n) \\
v_n = f_v(x_n, n) 
$$

> 注意，这里的 $f_q$ 其实是把 $\text{embedding}\_\text{vector} \times W_q$ 的矩阵乘法过程包含进去了，至于为什么要这样构造，下文会讲。

其中函数 $f_q、f_k、f_v$ 正是我们需要构造的位置编码函数。有了 query、key 和 value 向量表达式，接着就可以利用查询和键的值来计算注意力权重（$softmax(qk^T)$），输出则是对 $v_n$ 的加权求和。

$$
a_{m,n} = \frac{\exp\left(\frac{q_m^T k_n}{\sqrt{d}}\right)}{\sum_{j=1}^{N} \exp\left(\frac{q_m^T k_j}{\sqrt{d}}\right)} \\
o_m = \sum_{n=1}^{N} a_{m,n} v_n \quad (2)$$

方程 (1) 的一种常见选择是：


$$f_t:t∈\{q,k,v\}(x_i, i) := W_{t}(x_i + p_i)，\quad (3)$$

其中，$p_i \in \mathbb{R}^d$  是与 `token` $x_i$  的位置相关的 $d$ 维向量。Devlin 等人 [2019]、Lan 等人 [2020]、Clark 等人 [2020]、Radford 等人 [2019]、Radford 和 Narasimhan [2018] 使用了一组可训练向量  $p_i \in \{p_t\}_{t=1}^L$ ，其中 $L$ 表示最大序列长度。Vaswani 等人 [2017] 则提出了通过正弦函数来生成 $p_i$ 的方法:

$$p_{i,2t} = \sin\left(\frac{k}{10000^{2t/d}}\right) \\
p_{i,2t+1} = \cos\left(\frac{k}{10000^{2t/d}}\right)\quad (4)$$

其中， $p_{i,2t}$ 是 $p_i$ 的第 $2t$ 个维度。下一节会描述 RoPE 与这种基于正弦函数的直觉之间的关系。但是，**RoPE 并不是直接将位置信息 $p_i$ 和嵌入向量元素 $x_i$ 相加，而是通过与正弦函数相乘的方式引入相对位置信息**。

### 2D 的 RoPE 算法

[RoPE 论文](https://arxiv.org/pdf/2104.09864)提出为了能**利用 token 之间的相对位置信息（$m-n$）**，假定 query 向量 $q_m$ 和 key 向量 $k_n$ 之间的内积操作可以被一个函数 $g$ 表示，该函数 $g$ 的输入是词嵌入向量 $x_m$、$x_n$ 以及它们之间的相对位置 $m - n$，公式表达如下所示：

$$\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n) \quad (5)$$

> 注意，这里只有 $f_q(x_m, m)$, $f_k(x_n, n)$ 是需要求解的函数，$\langle  \rangle$ 表示内积操作，而对于 $g$，我们要求是表达式中有 $x_m, x_n, (m-n)$，也可以说是 **$q_m, k_n$ 的内积会受相对位置 $m-n$ 影响**。

接下来的目标就是**找到一个等价的位置编码方式 $f$，从而使得上述关系成立**，函数 $f_q$ 包含了位置编码和 $W_q \times q$（嵌入向量转换为 $q$ 向量）过程。

假设现在词嵌入向量的维度是两维 $d=2$，这样就可以利用上 $2$ 维度平面上的向量的几何性质，然后论文中提出了一个满足上述关系的 $f$ 和 $g$ 的形式如下:

$$
f_q(x_m, m) = (W_q x_m) e^{im\theta} \\
f_k(x_n, n) = (W_k x_n) e^{in\theta} \\
g(x_m, x_n, m - n) = Re \left[ (W_q x_m)(W_k x_n)^* e^{i(m-n)\theta} \right] \quad (6)$$
> 其中 \( Re \) 表示复数的实部，\( (W_k x_n)^* \) 表示 \( (W_k x_n) \) 的共轭复数。

$f_q、f_k$ 的推导需要基于三角函数定理、欧拉公式等，推导过程参考[这里](https://zhuanlan.zhihu.com/p/642884818)，本文直接给出结论：

1，**$f_q(x_m, m)$ 其实等于 `query` 向量乘以了一个旋转矩阵**，即:

$$f_q(x_m, m) = \begin{pmatrix} 
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix} 
q_m^{(1)} \\
q_m^{(2)} 
\end{pmatrix} \quad (7)$$

2，**$f_k(x_n, n)$ 其实等于 `key` 向量乘以了一个旋转矩阵**，即:

$$f_k(x_n, n) = \begin{pmatrix} 
\cos(n\theta) & -\sin(n\theta) \\
\sin(n\theta) & \cos(n\theta)
\end{pmatrix}
\begin{pmatrix} 
k_n^{(1)} \\
k_n^{(2)} 
\end{pmatrix} \quad (8)$$

3，同样可得 $g(x_m, x_n, m - n)$ 等于 $q_m^T$ 乘以旋转矩阵再乘以 $k_n$，即:

$$\langle f_q(x_m, m), f_k(x_n, n) \rangle  = \mathbf{q}_m^T R(m - n) \mathbf{k}_n \quad (9)$$

$$\begin{aligned}
g(x_m, x_n, m - n) &= (q_m^{(1)} k_n^{(1)} + q_m^{(2)} k_n^{(2)}) \cos((m - n)\theta) - (q_m^{(2)} k_n^{(1)} - q_m^{(1)} k_n^{(2)}) \sin((m - n)\theta) \\
&= \begin{pmatrix}
q_m^{(1)} & q_m^{(2)}
\end{pmatrix}
\begin{pmatrix}
\cos((m - n)\theta) & -\sin((m - n)\theta) \\
\sin((m - n)\theta) & \cos((m - n)\theta)
\end{pmatrix}
\begin{pmatrix}
k_n^{(1)} \\
k_n^{(2)}
\end{pmatrix} \\
 &= \mathbf{q}_m^T R(m - n) \mathbf{k}_n
\end{aligned} \quad(10)$$

公式（9）的证明可通过旋转矩阵性质得到，先将公式 (9) 抽象成 $\langle R_a X, R_b Y \rangle = \langle X, R_{b-a} Y \rangle$, 该等式的证明过程如下：


$$\begin{aligned}
\langle R_a X, R_b Y \rangle &= (R_aX)^T R_bY \\
&= X^T R_a^T R_bY \\
&=  X^T R(-a)R_bY \\
&=  X^T R_{(b-a)}Y = \langle X, R_{(b-a)}Y \rangle\\
\end{aligned} \quad(11)$$

上述推导过程分别应用了：展开内积、矩阵乘法的结合律、旋转矩阵性质1、旋转矩阵性质2。

### 多维的 RoPE 算法

前面的公式推导，是假设的词嵌入维度是 2维向量，将二维推广到任意维度，$f_{\{q,k\}}$ 可以表示如下：


$$f_{\{q,k\}}(x_m, m) = R_{\Theta, m}^d W_{\{q,k\}} x_m \tag{12}$$

其中，$R_{\Theta, m}^d$ 为 $d$ 维度的旋转矩阵，表示为：

$$R_{\Theta, m}^d =
\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix} \tag{13}$$

可以看出，对于$d >= 2$ 的通用情况，则是将词嵌入向量元素按照两两一组分组，每组应用同样的旋转操作且每组的旋转角度计算方式如下：

\[
\Theta = \left\{ \theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, \dots, d/2] \right\}
\]

将 RoPE 应用到前面公式（2）的 Self-Attention 计算，可以得到包含相对位置信息的Self-Attetion：

$$q_m^T k_n = \left( R_{\Theta, m}^d W_q x_m \right)^T \left( R_{\Theta, n}^d W_k x_n \right) = x_m^T W_q R_{\Theta, n-m}^d W_k x_n \tag{14}$$

其中，
$$R_{\Theta, n-m}^d = \left( R_{\Theta, m}^d \right)^T R_{\Theta, n}^d$$

Rotary Position Embedding(RoPE) 实现的可视化如下图所示:

<img src="../images/rope/figure1.png" width="60%" alt="figure1">

最后总结**结合 RoPE 的 self-attention 操作的流程**如下：
1. 首先，对于 `token` 序列中的每个词嵌入向量，都计算其对应的 query 和 key 向量;
2. 然后在得到 query 和 key 向量的基础上，应用公式（7）和（8）对每个 `token` 位置都计算对应的旋转位置编码；
3. 接着对每个 `token` 位置的 query 和 key 向量的元素按照**两两一组**应用旋转变换；
4. 最后再计算 `query` 和 `key` 之间的内积得到 self-attention 的计算结果。

RoPE的核心思想是将位置编码与词向量通过旋转矩阵相乘，使得词向量不仅包含词汇的语义信息，还融入了位置信息，其具有以下优点：

1. 相对位置感知：RoPE 能够自然地捕捉词汇之间的相对位置关系。
2. 无需额外的计算：位置编码与词向量的结合在计算上是高效的。
3. 适应不同长度的序列：RoPE 可以灵活处理不同长度的输入序列。

## RoPE 代码实现

llama 中代码实现如下:

```python
# LLaMA 官方实现代码 [4] 如下（经过简化）：
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)
        
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)
  # ......
```

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)
- [一文看懂 LLaMA 中的旋转式位置编码（Rotary Position Embedding）](https://zhuanlan.zhihu.com/p/642884818)

