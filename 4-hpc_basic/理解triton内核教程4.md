- [1. flash-attention 算法原理](#1-flash-attention-算法原理)
- [参考资料](#参考资料)


### 1. flash-attention 算法原理

标准自注意力的数学表达式如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中，$d_k$ 是键向量的维度，用于缩放点积结果以防止梯度消失。

### 参考资料

- [FlashAttentions](https://jcf94.com/2024/02/24/2024-02-24-flash-attention/)
- [榨干 GPU 效能的 Flash Attention 3](https://tomohiroliu22.medium.com/%E6%A6%A8%E4%B9%BEgpu%E6%95%88%E8%83%BD%E7%9A%84flashattention%E7%AC%AC%E4%B8%89%E4%BB%A3-4a8b0a2a812e)