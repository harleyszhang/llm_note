- [1. flash-attention 算法原理](#1-flash-attention-算法原理)
- [参考资料](#参考资料)

### 1. flash-attention 算法原理

对于 FA 的设计思路, 2023 年我的建议是不要去读 FA-V1 论文了，推荐学习路线：

1. 学习 Online Softmax: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
2. 看 FlashAttention-2 论文。因为 FlashAttention V1 的关键设计在V2中被推翻了，FA-V2 论文也介绍了 FA-V1。
3. 看 `Flash Decoding`（不在本文介绍范围内）。


### 参考资料

- [FlashAttentions](https://jcf94.com/2024/02/24/2024-02-24-flash-attention/)
- [榨干 GPU 效能的 Flash Attention 3](https://tomohiroliu22.medium.com/%E6%A6%A8%E4%B9%BEgpu%E6%95%88%E8%83%BD%E7%9A%84flashattention%E7%AC%AC%E4%B8%89%E4%BB%A3-4a8b0a2a812e)