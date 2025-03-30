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
		- [模型初始化函数](#模型初始化函数)
		- [定义视觉编码函数 `vision_encode`](#定义视觉编码函数-vision_encode)
		- [文本和图像特征合并函数 `get_multi_modal_input_embeddings`](#文本和图像特征合并函数-get_multi_modal_input_embeddings)
		- [merge\_input\_ids\_with\_image\_features 合并文本和图像特征函数](#merge_input_ids_with_image_features-合并文本和图像特征函数)
		- [forward 推理函数](#forward-推理函数)
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

#### 模型初始化函数

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

#### 定义视觉编码函数 `vision_encode`

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

#### 文本和图像特征合并函数 `get_multi_modal_input_embeddings`

`get_multi_modal_input_embeddings` 函数有两个参数，其实现流程可以总结如下:

1. **获取文本的嵌入向量**：使用语言模型的嵌入层（`nn.Embedding`）将 `input_ids` 映射到固定尺寸的连续稠密向量（`embedding vectors`）。
2. **合并文本 `embedding` 向量和视觉 `embedding` 向量**：这个过程很复杂，通过抽象出一个专门的函数 `merge_input_ids_with_image_features` 将文本嵌入和图像特征合并。

```python
def get_multi_modal_input_embeddings(
    self,
    input_ids: torch.Tensor,
    vision_embeddings = None,
) -> torch.Tensor:
    """获取输入嵌入，包括文本和视觉嵌入的合并。"""
    # torch.Size([1, 22]) --> torch.Size([1, 22, 4096])
    llm_inputs_embeds = self.language_model.get_input_embeddings(input_ids) 
    
    if vision_embeddings is not None:
        inputs_embeds, position_ids = merge_input_ids_with_image_features(
            input_ids,              # 文本 token ID
            llm_inputs_embeds,      # 文本嵌入向量 torch.Size([1, 22, 4096]) 
            vision_embeddings,      # 视觉嵌入向量 shape torch.Size([1, 576, 4096])
            self.llava_config.pad_token_id,       # pad token ID
            self.llava_config.image_token_index,  # 图像 token 的插入索引 32000
        )

    return inputs_embeds, position_ids
```

#### merge_input_ids_with_image_features 合并文本和图像特征函数

函数声明如下:

```python
def merge_input_ids_with_image_features(
    input_ids: torch.Tensor, 
    inputs_embeds: torch.Tensor, 
    image_features: torch.Tensor,
    pad_token_id: int,
    image_token_index: int
):
```

这里的函数参数不好理解，先看下它们各自的意义和作用：
- `input_ids`: 输入的 token IDs, 形状为 (batch_size, sequence_length)。
- `inputs_embeds`: 文本嵌入，形状为 (batch_size, sequence_length, embed_dim)。
- `image_features (torch.Tensor)`: 视觉编码后的图像特征，形状为 (num_images, num_image_patches, embed_dim)。
- `pad_token_id` (int): 填充 token 的 ID，因为 batch 输入的请求长短不一。
- `image_token_index` 参数用于**标识输入文本中预留来插入图像特征的位置**。也就是说，当输入的 token 序列中出现值等于 `image_token_index` 的 token 时，说明这个位置不是真正的文本 token，而是一个**占位符**，后续将用图像特征来替换或扩展该位置的信息。示例：llava 系列模型，image_token_index = 32000.

代码来源 [transformers 库](https://github.com/jianxx/transformers/blob/72d1a4cd53d90d5db384df948ccc293b3c1e3b9d/src/transformers/models/llava/modeling_llava.py)，代码详解如下所示：

```python
def merge_input_ids_with_image_features(
    input_ids: torch.Tensor, 
    inputs_embeds: torch.Tensor, 
    image_features: torch.Tensor,
    pad_token_id: int,
    image_token_index: int
):
    """
    将 input_ids 与 image_features 合并，生成最终的嵌入和位置 ID。
    
    Args:
        input_ids (torch.Tensor): 输入的 token IDs, 形状为 (batch_size, sequence_length)
        inputs_embeds (torch.Tensor): 文本嵌入，形状为 (batch_size, sequence_length, embed_dim)
        image_features (torch.Tensor): 视觉编码后的图像特征，形状为 (num_images, num_image_patches, embed_dim)
        pad_token_id (int): 填充 token 的 ID
        image_token_index (int): 图像 token 的 ID
    
    Returns:
        final_embedding (torch.Tensor): 合并后的嵌入，形状为 (batch_size, max_embed_dim, embed_dim)
        position_ids (torch.Tensor): 位置 ID, 形状为 (batch_size, max_embed_dim)
    """
    num_images, num_image_patches, embed_dim = image_features.shape # torch.Size([1, 576, 4096])
    batch_size, sequence_length = input_ids.shape # torch.Size([1, 22])

    # 计算 attention_mask 从 input_ids
    attention_mask = (input_ids != pad_token_id).long()

    # 检查每个样本的最后一个 token 是否为填充 token
    left_padding = not torch.sum(input_ids[:, -1] == pad_token_id).bool().any() # True

    # 创建图像占位符 token 的掩码，获取特殊图像 token 的位置
    """
    tensor([[False, False, False, False,  True, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False]], device='cuda:0')
    """
    special_image_token_mask = input_ids == image_token_index
    # 统计每个样本中图像 token 的数量, 形状为 [batch_size, *]
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1) # 1

    # 计算文本和图像特征合并后的新序列的最大长度。
    # 每个图像占位符 token 的位置会被替换为 (num_image_patches - 1) 个图像 patches embedding token。
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length # tensor(597, device='cuda:0')

    # 获取非图像占位符 token 的位置索引
    """
    torch.where() 的输出
	    - batch_indices 包含满足条件元素所在的行号
        - non_image_indices 包含对应元素在行中的列索引
    """
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index) 

    # 计算文本 token 在新序列中的位置
    """
    对于每个 token：
	    - 如果该 token 不是特殊图像 token（mask 为 0）：意味着该 token 占用 1 个位置。
        - 如果该 token 是特殊图像 token（mask 为 1）：意味着该 token将扩展成 num_image_patches 个位置，
        其中后面 num\_image\_patches - 1 位置用于放置图像 patch 嵌入，而原位置仍保留（但后续会用图像特征覆盖）。
    使用 torch.cumsum(..., dim=-1) 对上一步结果做累积和，得到每个 token 在新序列中的“终止位置”，再减 1 得到 token 实际开始的索引。
    这一步给出了新序列中，每个原始 token 对应的新位置索引。
    """
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1 
    # new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1).float(), dim=-1).long() - 1 # torch.Size([1, 22])
    # nb_image_pad 表示新序列中需要额外填充的图像 token 数量，以使总长度达到 max_embed_dim
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1] 

    # 如果存在左侧填充 (left_padding 为 True)，则将 new_token_positions 进行偏移调整。
    """
    tensor([[  0,   1,   2,   3, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588,
         589, 590, 591, 592, 593, 594, 595, 596]], device='cuda:0')
    """
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding

    # 确定文本 token 在新序列中的位置
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 初始化最终的嵌入, torch.Size([1, 597, 4096])
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    
    # 将 tensors 移动到目标设备
    target_device = inputs_embeds.device
    batch_indices = batch_indices.to(target_device)
    non_image_indices = non_image_indices.to(target_device)
    text_to_overwrite = text_to_overwrite.to(target_device)

    # 填充文本嵌入
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]

    # 确定图像特征插入位置，通过找到 final_embedding 中所有全 0 的位置
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # 找出 final_embedding 中所有维度为0的位置
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    if image_to_overwrite.sum() != image_features.shape[0] * image_features.shape[1]:
        raise ValueError(      
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    # 将 image_features 重新排列为 (num_images * num_image_patches, embed_dim)，并填充到 final_embedding 的相应位置。
    final_embedding[image_to_overwrite] = image_features.contiguous().view(-1, embed_dim).to(target_device)
    
    # 生成 position_ids
    position_ids = torch.arange(max_embed_dim, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

    # 处理填充位置的嵌入, 将填充位置的嵌入设为0：
    batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]

    final_embedding[batch_indices_pad, indices_to_mask] = 0

    return final_embedding, position_ids
```

#### forward 推理函数


## 参考资料

- [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
- [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/pdf/2310.03744)
- [【多模态大模型】llava系列：llava、llava1.5、llava-next](https://zhuanlan.zhihu.com/p/695100288)