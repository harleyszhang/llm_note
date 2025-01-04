---
layout: post
title: LLaVA 系列模型结构详解
date: 2024-11-28 11:50:00
summary: 多模态大模型 MLLM 架构通常都是 LLM + 视觉编码器 + 映射层的组合。本文详细总结了 LLaVA 系列多模态模型的模型结构，以及视觉编码器如何支持高分辨率输入图像。
categories: Transformer
---

- [前言](#前言)
- [LLaVA1](#llava1)
  - [ViT-L/14 模型结构](#vit-l14-模型结构)
- [LLaVA1.5](#llava15)
  - [LLaVA-1.5-HD](#llava-15-hd)
- [LLaVA1.6（LLaVA-NeXT）](#llava16llava-next)
- [参考资料](#参考资料)

## 前言

视觉语言模型 VIA 或者说多模态大模型 `MLLM` 架构通常都是: LLM + 视觉编码器 + 映射层的组合。英伟达发布的视觉语言模型 VILA 架构和训练流程如下图所示：

<div align="center">
<img src="../images/llava_model/VILA_infer_train.jpg" width="80%" alt="VILA_infer_train">
</div>

可以看出视觉语言模型的架构是由视觉 encoder、映射层和语言 decoder 组成。常见的视觉语言模型如下所示，模型架构都很相似。
- VILA-1.5 (B/8B/13B/40B)
- LLaVA(1.5,1.6) (7B-34B)
- InternLM-XComposer2 (7B, 4khd-7B)
- QWen-VL (7B)
- DeepSeek-VL (7B)

## LLaVA1

Llava1 的模型结构很简洁，**CLIP 模型的视觉编码器 + 映射层 + LLM（Vicuna、LLama）** ，利用 CLIP 模型的 Vison Encoder 结构对输入图片提取视觉特征，即转换为形状为 `[N=1, grid_H x grid_W, hidden_dim]` 的 feature map，然后通过一个映射层（线性层）将图像特征对齐到文本特征维度，即得到形状为 `[N=1, grid_H x grid_W, embedding_dim]` 的 image tokens embedding 向量，再然后将图片 tokens 向量和输入文本 tokens 向量 `concat` 后作为 `LLM` 的输入，生成回答文本。

LLaVA 模型架构如下图所示吧:

![llava_model](../images/llava_model/llava_model.png)

具体来说，对于输入图像 $X_v$，采用预训练 `CLIP` 模型的视觉编码器 `ViT-L/14(224²)`，其生成的视觉特征为 $Z_v = g(X_v)$，在作者的实验中，只用最后一个 Transformer 层之前和之后的网格特征。并使用一个**简单的线性层**将图像特征连接（映射）到词嵌入空间，通过一个可训练的投影矩阵 $W$ 将 $Z_v$ 转换为语言嵌入标记 $H_v$，$Z_v$ 向量的最后一个维度就是 `LLM` 的**词嵌入空间维度** `embedding_dim`。

$$H_v = W\cdot X_v, with Z_v = g(X_v)$$

### ViT-L/14 模型结构

`ViT-L/14` 模型的 `L` 表示模型的规模，为 “Large”，ViT(Vision Transformer) 模型有不同规模的模型，例如：
- `ViT-B`（Base）：通常有 12 层 Transformer。
- `ViT-L`（Large）：通常有 24 层 Transformer。
- `ViT-H`（Huge）：通常有 32 层 Transformer。

`ViT` 会将输入图像分割成固定大小的 `patch`（例如 14x14），`ViT-L/14` 即表示 `patch` 大小为 14，
> CLIP 模型的视觉部分使用 ViT 来编码图像的特征，文本部分使用 Transformer 来编码文本的特征。

不同版本 `ViT` 模型的参数总结：

| 模型版本   | Transformer 层数 | 隐藏维度 | 参数量  | Patch 分辨率 |
|------------|------------------|----------|---------|--------------|
| ViT-B/16   | 12               | 768      | 86M     | 16x16        |
| ViT-L/16   | 24               | 1024     | 307M    | 16x16        |
| ViT-L/14   | 24               | 1024     | 307M    | 14x14        |
| ViT-H/14   | 32               | 1280     | 632M    | 14x14        |

## LLaVA1.5

模型结构上和前作相比，LLaVA1.5 将之前用于维度映射的的简单一层线性层替换为 `2` 层 线性层的 `MLP` 结构，并将 `clip-L/14` 的输入分辨率从 `224*224` 提升到 `336*336`，因为作者发现提高输入图像分辨率能够增强模型性能，LLM 换成了 Vicuna1.5（在 `LLama2` 上微调的模型）

### LLaVA-1.5-HD

目前开源的 CLIP 视觉编码器的分辨率上限为 `336*336`，这意味着无法简单地替换视觉编码器来支持更高分辨率的图像。为了解决这个问题，论文探索了一种方法，既能让多模态语言模型（LMM）处理高分辨率图像，又能保持 LLaVA-1.5 的高效数据使用。

将输入图像划分为若干小块，每块的分辨率与视觉编码器原始训练时一致，然后分别对这些块进行编码。编码完成后，我们将这些块的特征图合并为目标分辨率的大特征图，并将其输入到 LLM 中。同时，为了给 `LLM` 提供**全局上下文信息**并减少图像分割、编码和合并操作带来的不良影响，我们还将一个经过下采样（`resize`）的图像特征连接到合并后的特征图中。

![llava1.5_model](../images/llava_model/llava1.5_model.png)

这样的设计允许我们处理任意分辨率的图像，同时保持 LLaVA-1.5 的数据高效性。这一新模型被作者命名为 `LLaVA-1.5-HD`。

## LLaVA1.6（LLaVA-NeXT）

模型推理层面新的升级点在于，Vision Encoder 分辨率支持更大的分辨率，包括 672x672, 336x1344, 1344x336 几种分辨率的输入，并且支持通过图片裁切，编码，合并来实现，和前作一样的方法。毕竟，当提供高分辨率图像和保留细节的表征时，模型感知图像中复杂细节的能力会显著提高。它减少了面对低分辨率图像时的模型幻觉，即猜测想象的视觉内容。

## 参考资料

- [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
- [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/pdf/2310.03744)
- [【多模态大模型】llava系列：llava、llava1.5、llava-next](https://zhuanlan.zhihu.com/p/695100288)