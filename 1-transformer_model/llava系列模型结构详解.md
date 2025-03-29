---
layout: post
title: LLaVA 系列模型结构详解
date: 2024-11-28 11:50:00
summary: 多模态大模型 MLLM 架构通常都是 LLM + 视觉编码器 + 映射层的组合。本文详细总结了 LLaVA 系列多模态模型的模型结构，以及视觉编码器如何支持高分辨率输入图像。
categories: Transformer
---

- [1. 前言](#1-前言)
- [二 LLaVA 系列模型](#二-llava-系列模型)
- [2.1 LLaVA1](#21-llava1)
		- [2.1.1 ViT-L/14 模型结构](#211-vit-l14-模型结构)
	- [2.2. LLaVA1.5](#22-llava15)
		- [2.2.1 LLaVA-1.5-HD](#221-llava-15-hd)
	- [2.3. LLaVA1.6（LLaVA-NeXT）](#23-llava16llava-next)
- [三. LLaVA 多模态模型推理流程](#三-llava-多模态模型推理流程)
	- [3.1 查看模型结构信息](#31-查看模型结构信息)
	- [3.2 实现 LLaVA 模型结构](#32-实现-llava-模型结构)
- [参考资料](#参考资料)

## 1. 前言

NVIDIA 和 MIT 的研究人员推出的视觉语言模型 `VILA`，其模型架构和训练流程如下图所示：

<div align="center">
<img src="../images/llava_model/VILA_infer_train.jpg" width="80%" alt="VILA_infer_train">
</div>

上图可以看出 VILA 模型架构是由视觉 encoder（ViT）、映射层（线性层）和 LLM 组成。

目前常见的视觉语言模型（也叫多模态模型）的类别有如下所示，它们的模型架构都很相似，都是: **视觉编码器 + 映射层 + LLM 的组合**。
- VILA-1.5 (B/8B/13B/40B)
- LLaVA(1.5,1.6) (7B-34B)
- InternLM-XComposer2 (7B, 4khd-7B)
- QWen-VL (7B)
- DeepSeek-VL (7B)

## 二 LLaVA 系列模型

## 2.1 LLaVA1

Llava1 的模型结构很简洁，**CLIP 模型的视觉编码器 + 映射层 + LLM（Vicuna、LLama）** ，利用 `CLIP` 模型的 Vison Encoder 结构对输入图片提取视觉特征，即转换为形状为 `[N=1, grid_H x grid_W, hidden_dim]` 的 feature map，然后通过一个映射层（线性层）将图像特征对齐到文本特征维度，即得到形状为 `[N=1, grid_H x grid_W, embedding_dim]` 的 image tokens embedding 向量，再然后将图片 tokens 向量和输入文本 tokens 向量 `concat` 后作为 `LLM` 的输入，生成回答文本。

LLaVA 模型架构如下图所示吧:

![llava_model](../images/llava_model/llava_model.png)

具体来说，对于输入图像 $X_v$，采用预训练 `CLIP` 模型的视觉编码器 `ViT-L/14(224²)`，其生成的视觉特征为 $Z_v = g(X_v)$，在作者的实验中，只用最后一个 Transformer 层之前和之后的网格特征。并使用一个**简单的线性层**将图像特征连接（映射）到词嵌入空间，通过一个可训练的投影矩阵 $W$ 将 $Z_v$ 转换为语言嵌入标记 $H_v$，$Z_v$ 向量的最后一个维度就是 `LLM` 的**词嵌入空间维度** `embedding_dim`。

$$H_v = W\cdot X_v, with Z_v = g(X_v)$$

#### 2.1.1 ViT-L/14 模型结构

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

### 2.2. LLaVA1.5

模型结构上和前作相比，LLaVA1.5 将之前用于维度映射的的简单一层线性层替换为 `2` 层 线性层的 `MLP` 结构，并将 `clip-L/14` 的输入分辨率从 `224*224` 提升到 `336*336`，因为作者发现提高输入图像分辨率能够增强模型性能，LLM 换成了 Vicuna1.5（在 `LLama2` 上微调的模型）

#### 2.2.1 LLaVA-1.5-HD

目前开源的 CLIP 视觉编码器的分辨率上限为 `336*336`，这意味着无法简单地替换视觉编码器来支持更高分辨率的图像。为了解决这个问题，论文探索了一种方法，既能让多模态语言模型（LMM）处理高分辨率图像，又能保持 LLaVA-1.5 的高效数据使用。

将输入图像划分为若干小块，每块的分辨率与视觉编码器原始训练时一致，然后分别对这些块进行编码。编码完成后，我们将这些块的特征图合并为目标分辨率的大特征图，并将其输入到 LLM 中。同时，为了给 `LLM` 提供**全局上下文信息**并减少图像分割、编码和合并操作带来的不良影响，我们还将一个经过下采样（`resize`）的图像特征连接到合并后的特征图中。

![llava1.5_model](../images/llava_model/llava1.5_model.png)

这样的设计允许我们处理任意分辨率的图像，同时保持 LLaVA-1.5 的数据高效性。这一新模型被作者命名为 `LLaVA-1.5-HD`。

### 2.3. LLaVA1.6（LLaVA-NeXT）

模型推理层面新的升级点在于，Vision Encoder 分辨率支持更大的分辨率，包括 672x672, 336x1344, 1344x336 几种分辨率的输入，并且支持通过图片裁切，编码，合并来实现，和前作一样的方法。毕竟，当提供高分辨率图像和保留细节的表征时，模型感知图像中复杂细节的能力会显著提高。它减少了面对低分辨率图像时的模型幻觉，即猜测想象的视觉内容。

## 三. LLaVA 多模态模型推理流程

LLaVA 多模态模型推理 pipline：
1. prompts 预处理；
2. 视觉特征预处理；
3. 视觉特征模型 clip 推理；
4. 视觉特征和文本特征合并成一组 tokens；
5. 语言模型 llama 推理。

### 3.1 查看模型结构信息

查看模型结构信息最简单直接的办法是去看模型源代码，但是直接源代码可能没那么直观，因此也可以通过 transformers 库加载模型并打印模型结构信息的方式，代码如下所示：

```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import LlavaConfig
import sys, os

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lite_llama.models.llava import LlavaLlama

hf_model_path = "/gemini/code/liuhaotian/llava-v1.5-7b"

def test_LlavaLlama_structure(hf_model_path):
    
    # 使用 init_empty_weights 初始化空模型
    with init_empty_weights():
        config = LlavaConfig.from_pretrained(hf_model_path)
        model = LlavaLlama(config)
        
        # 打印没有加载权重的 LlavaLlama 模型结构
        print(model)
        # 打印模型的简单摘要
        print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

        # 可选择打印部分参数信息
        for name, param in list(model.named_parameters())[:]:  # 打印模型参数
            print(name, param.shape)

if __name__ == "__main__":
    test_LlavaLlama_structure(hf_model_path)
```

模型结构信息输出如下所示:

```bash
LlavaForConditionalGeneration(
  (vision_tower): CLIPVisionModel(
    (vision_model): CLIPVisionTransformer(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPSdpaAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (multi_modal_projector): LlavaMultiModalProjector(
    (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
    (act): GELUActivation()
    (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
  )
  (language_model): LlamaForCausalLM(
    (model): LlamaModel(
      (embed_tokens): Embedding(32064, 4096)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaSdpaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
          (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        )
      )
      (norm): LlamaRMSNorm((4096,), eps=1e-05)
      (rotary_emb): LlamaRotaryEmbedding()
    )
    (lm_head): Linear(in_features=4096, out_features=32064, bias=False)
  )
)
```

从上述模型结构信息也能明显看出 LlaVA 模型结构主要包括 3 个模块: 

1. vision_tower 视觉模块：`CLIPVisionModel`；
2. multi_modal_projector 映射层: LlavaMultiModalProjector（实际是两个直连的线性层）。
3. language_model 大语言模型: LlamaForCausalLM。

占据 LLaV1.5 模型主要参数量和计算量的是 LlamaForCausalLM, 视觉模块和特征映射模块只有几百MB的参数量。

### 3.2 实现 LLaVA 模型结构

1，模型初始化函数 __init__

主要是解析模型配置类，主要是获取视觉模块配置 + 映射层配置 + llama 模型配置，代码如下所示:

```python
class LlavaLlama(nn.Module):
    def __init__(self, llava_config: LlavaConfig):
        super().__init__()
        self.device = "cuda"  # 默认运行在 GPU 上
        self.llava_config = llava_config

        # 提取文本模型配置并转为 LlamaConfig 类型（为了自定义加载）
        text_config = self.llava_config.text_config
        self.llama_config = LlamaConfig.from_dict(text_config.to_dict())

        # 指定提取哪一层视觉特征
        self.select_layer = llava_config.vision_feature_layer
        self.select_feature = llava_config.vision_feature_select_strategy

        # 初始化视觉编码器（比如 CLIP）
        self.vision_tower = AutoModel.from_config(llava_config.vision_config)

        # 初始化多模态投影模块
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=llava_config.vision_config.hidden_size,
            text_hidden_size=llava_config.text_config.hidden_size,
            projector_hidden_act=llava_config.projector_hidden_act
        )

        # 初始化 LLaMA 语言模型
        self.language_model = LlamaModel(self.llama_config)

        # 设置 pad token（防止 None 类型报错）
        self.pad_token_id = self.llava_config.pad_token_id if self.llava_config.pad_token_id is not None else -1
```

2，定义视觉编码函数 vision_encode

__init__ 初始化函数通过解析 LlavaConfig 配置，并通过 transformers 库的 `AutoModel.from_config`从配置中获取 vision_tower 模型结构，也就是初始化函数中已经定义好了视觉编码模块结构。

视觉编码函数的流程：

1. **视觉特征提取**：提取图像（视频）视觉特征；
2. **特征筛选**：根据策略选择图像特征是 "default" 还是 "pad";
3. **特征空间对齐**：最后通过 `multi_modal_projector` 特征投影模块，将提取的视觉特征投影到与文本模型相同的表示空间中，本质上是让**视觉特征张量的最后一个维度是 `hidden_size`**。这一步是多模态融合的关键，它确保视觉信息能够以语言模型理解的方式表示。

```python
def vision_encode(self, image_tensor):
	x = image_tensor.half().to(device=self.device)
	
	# 1. 通过视觉处理模块提取图像特征
	x = self.vision_tower(x, output_hidden_states = True)
	x = x.hidden_states[self.select_layer]
	x = self._select_image_features(x, self.select_feature)
	
	# 2. 通过多模态投影器将图像特征转换为多模态嵌入
	image_features = self.multi_modal_projector(x)

	assert not torch.isnan(image_features).any(), f"After vision_tower image_features tensor contains NaN values!"
	return image_features
```

3，文本和图像特征合并函数 get_multi_modal_input_embeddings

get_multi_modal_input_embeddings 方法是 LLaVA 模型中实现多模态融合的核心函数，它将文本和图像特征整合成一个统一的表示空间，使得语言模型能够同时理解和处理两种模态的信息。

1. 首先，该方法接收两个关键参数：input_ids（文本的词元ID序列）和可选的 vision_embeddings（已经通过视觉编码器处理过的图像特征）。方法开始时，它调用语言模型的词嵌入层将文本词元ID转换为对应的嵌入向量，这一步将形状为 [1, 22] 的输入转换为形状为 [1, 22, 4096] 的嵌入表示，其中4096是嵌入维度。

2. 其次，当提供了视觉嵌入（vision_embeddings）时，方法会调用 merge_input_ids_with_image_features 函数将文本嵌入和图像特征合并。这个合并过程非常精巧：它首先在文本序列中定位特殊的图像词元（由 image_token_index 指定），然后用对应的图像特征向量替换这些词元。从相关实现看，一个图像词元通常会被展开为多个图像块（patches）的特征表示，这使得最终的序列长度会比原始文本序列长得多。合并过程还会同时生成适当的位置ID（position_ids），这对于Transformer模型中的位置编码至关重要，确保模型能够识别每个词元在序列中的相对位置，无论它是来自文本还是图像。

3. 最后，函数通过断言确保生成的嵌入不包含任何NaN值，这是一种质量控制措施，防止后续计算中出现数值问题。

```python
def get_multi_modal_input_embeddings(
        self,
        input_ids: torch.Tensor,
        vision_embeddings = None,
    ) -> torch.Tensor:
        """获取输入嵌入，包括文本和视觉嵌入的合并。"""
        llm_inputs_embeds = self.language_model.get_input_embeddings(input_ids) # torch.Size([1, 22]) --> torch.Size([1, 22, 4096])
        
        # torch.Size([1, 576, 4096]) torch.Size([1, 22, 4096]) torch.Size([1, 22])
        # print("self.llava_config.image_token_index is ", self.llava_config.image_token_index)
        if vision_embeddings is not None:
            inputs_embeds, position_ids = merge_input_ids_with_image_features(
                input_ids, llm_inputs_embeds, vision_embeddings, 
                self.llava_config.pad_token_id,
                self.llava_config.image_token_index,
            )
        
        assert not torch.isnan(inputs_embeds).any(), f"After merge inputs_embeds tensor contains NaN values!"

        return inputs_embeds, position_ids
```

## 参考资料

- [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
- [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/pdf/2310.03744)
- [【多模态大模型】llava系列：llava、llava1.5、llava-next](https://zhuanlan.zhihu.com/p/695100288)