- [我的自制大模型推理框架课程介绍](#我的自制大模型推理框架课程介绍)
- [一 transformer 模型](#一-transformer-模型)
	- [1.1 transformer 系列模型](#11-transformer-系列模型)
	- [1.2 LLM 性能分析](#12-llm-性能分析)
- [二 大语言模型压缩](#二-大语言模型压缩)
- [三 大语言模型推理及部署（服务化）](#三-大语言模型推理及部署服务化)
	- [3.1 LLM 综合性能分析](#31-llm-综合性能分析)
	- [3.2 LLM 推理优化-算法层面](#32-llm-推理优化-算法层面)
	- [3.3 LLM 推理服务框架解析](#33-llm-推理服务框架解析)
	- [3.4 系统优化方法](#34-系统优化方法)
	- [3.5 LLM 可视化](#35-llm-可视化)
- [四 高性能计算](#四-高性能计算)
	- [4.1 triton 笔记](#41-triton-笔记)
	- [4.2 cuda 笔记](#42-cuda-笔记)
	- [4.3 高性能编程学习资料推荐](#43-高性能编程学习资料推荐)
- [参考资料](#参考资料)

LLM notes, including model inference, transformer model structure, and lightllm framework code analysis notes.

## 我的自制大模型推理框架课程介绍

1. **框架亮点**：基于 Triton + PyTorch 开发的轻量级、且简单易用的大模型推理框架，采用类 Pytorch 语法的 Triton 编写算子，绕开 Cuda 复杂语法实现 GPU 加速。
2. **价格：499**。非常实惠和便宜，课程、项目、面经、答疑质量绝对对得起这个价格。
3. **课程/项目优势​**：**手把手教你从 0 到 1 实现大模型推理框架，2025 最新的高性能计算/推理框架岗位的大厂面经分享**。
	- 架构清晰，代码简洁且注释详尽，覆盖大模型离线推理全流程。​
    - 运用 OpenAI Triton 编写高性能计算 Kernel，开发矩阵乘法内核，效率堪比 cuBLAS。​
    - 依托 PyTorch 进行高效显存管理。​
    - 课程项目完美支持 FlashAttentionV1、V2、V3 与 `GQA`，以及 `PageAttention` 的具体实现。​
    - 使用 `Triton` 编写融合算子，如 KV 线性层融合等。​
    - 适配最新的 `llama/qwen2.5/llava1.5` 模型，相较 transformers 库，在 llama3 1B 和 3B 模型上，加速比最高可达 `4` 倍。
4. **项目运行效果**。

llama3.2-1.5B-Instruct 模型流式输出结果测试：

![流式输出](./images/read_me/generate.gif)

Qwen2.5-3B 模型（社区版本）流式输出结果测试：

![流式输出](./images/read_me/output.gif)

Llava1.5-7b-hf 模型流式输出结果测试:

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center"><img src="./images/read_me/llava_output2.gif" width="90%" alt="llava_output2"></td>
    <td align="center"><img src="./images/read_me/llava_output1.gif" width="100%" alt="llava_output"></td>
  </tr>
</table>

感兴趣的同学可以扫码联系课程购买，这个课程是我和[《自制深度学习推理框架》作者](https://space.bilibili.com/1822828582)一起合力打造的，内容也会持续更新优化。

<div align="center">
<img src="./images/read_me/fu_qcode.jpg" width="40%" alt="transformer_block_mp">
</div>

## 一 transformer 模型
### 1.1 transformer 系列模型

- [transformer 论文解读](./1-transformer_model/transformer论文解读.md)
- [transformer 模型代码实现](./1-transformer_model/transformer模型结构详解及实现.md)
- [llama1-3 模型结构详解](./1-transformer_model/llama1-3模型结构详解.md)
- [vit 论文速读](./1-transformer_model/vit论文速读.md)
- [gpt1-3 论文解读](./1-transformer_model/gpt1-3论文解读.md)
- [RoPE 位置编码算法详解](./1-transformer_model/RoPE位置编码算法详解.md)
- [Sinusoida 位置编码算法详解](./1-transformer_model/Sinusoida位置编码详解.md)

### 1.2 LLM 性能分析

- [llm 参数量-计算量-显存占用分析](./1-transformer_model/llm参数量-计算量-显存占用分析.md)
- [llm 推理 latency 分析](1-transformer_model/llm推理latency分析.md)

## 二 大语言模型压缩

- [SmoothQuant 论文解读](./2-llm_compression/SmoothQuant论文解读.md)
- [SmoothQuant 算法源码剖析](./2-llm_compression/SmoothQuant源码剖析.md)
- [AWQ 论文解读](./2-llm_compression/SmoothQuant论文解读.md)

## 三 大语言模型推理及部署（服务化）

### 3.1 LLM 综合性能分析

- [Roofline 论文解读](./3-llm_infer_deploy/Roofline论文解读.md)
- [llm 推理揭秘论文翻译](3-llm_infer_deploy/llm推理揭秘论文翻译.md)
- [llm 综合分析论文翻译](3-llm_infer_deploy/llm综合分析论文翻译.md)

### 3.2 LLM 推理优化-算法层面

- [online-softmax 论文解读](./3-llm_infer_deploy/fast_algorithm/online-softmax论文解读.md)
- [flashattention-1 论文解读](./3-llm_infer_deploy/fast_algorithm/flashattention-1论文解读.md)
- [flashattention-2 论文解读](./3-llm_infer_deploy/fast_algorithm/flashattention-2论文解读.md)
- [flashattention-3 论文解读](./3-llm_infer_deploy/fast_algorithm/flashattention-3论文解读.md)
- [flashattention1-2-3 系列总结](./3-llm_infer_deploy/fast_algorithm/flashattention1-2-3系列总结.md)

### 3.3 LLM 推理服务框架解析

LLM 推理服务框架技术总结和源码解析：

- [tgi 框架初步解析](./3-llm_infer_deploy/deepspeed_note/tgi框架解析.md)
- [vllm 优化技术速览](./3-llm_infer_deploy/lightllm_analysis/vllm优化技术速览.md)
- [lightllm 框架速览](./3-llm_infer_deploy/lightllm_analysis/lightllm框架速览.md)
- [lightllm 模型推理概述](./3-llm_infer_deploy/lightllm_analysis/lightllm模型推理概述.md)

**DeepSpeed 框架学习笔记**：

- [DeepSpeed:通过系统优化和压缩加速大规模模型推理和训练](./3-llm_infer_deploy/deepspeed_note/deepspeed-通过系统优化和压缩加速大规模模型推理和训练.md)
- [DeepSpeed 推理:具有定制推理内核和量化支持的多 GPU 推理](./3-llm_infer_deploy/deepspeed_note/deepspeed推理-具有定制推理内核和量化支持的多GPU推理.md)

### 3.4 系统优化方法

图优化、算子融合、深度学习推理框架系统层面的优化。

### 3.5 LLM 可视化

- [http://llm-viz-cn.iiiai.com/llm](http://llm-viz-cn.iiiai.com/llm)

## 四 高性能计算

### 4.1 triton 笔记

- [理解 triton 之基础知识](./4-hpc_basic/理解triton之基础知识.md)
- [理解 triton 内核教程 1](./4-hpc_basic/理解triton内核教程1.md)
- [理解 triton 内核教程 2](./4-hpc_basic/理解triton内核教程2.md)
- [理解 triton 内核教程 3](./4-hpc_basic/理解triton内核教程3.md)
- [理解 triton 内核教程 4](./4-hpc_basic/理解triton内核教程4.md)

### 4.2 cuda 笔记

- [英伟达 GPU 架构总结](./4-hpc_basic/英伟达GPU架构总结.md)
- [英伟达 GPU 通信理解](./4-hpc_basic/英伟达GPU通信理解.md)
- [英伟达 GPU 性能分析指导](./4-hpc_basic/英伟达GPU性能分析指导.md)
- [理解 Roofline 性能分析模型](./4-hpc_basic/深入理解Roofline模型.md)
- [CUDA 背景知识](./4-hpc_basic/CUDA背景知识.md)
- [CUDA 编程模型概述](./4-hpc_basic/CUDA编程模型概述.md)
- [CUDA 编程模型进阶](./4-hpc_basic/CUDA编程模型进阶.md)
- [CUDA 内存组织](./4-hpc_basic/CUDA内存组织.md)
- [CUDA 执行模型](./4-hpc_basic/CUDA执行模型.md)
- [CUDA 内核执行配置及线程索引计算](./4-hpc_basic/CUDA内核执行配置及线程索引计算.md)
- [CUDA 内核优化策略](./4-hpc_basic/CUDA内核优化策略.md)
- [CUDA 流介绍](./4-hpc_basic/CUDA流介绍.md)

### 4.3 高性能编程学习资料推荐

1, 英伟达 gpu cuda 编程语法和特性学习资料推荐：

- [GPU Architecture and Programming](https://homepages.laas.fr/adoncesc/FILS/GPU.pdf): 了解 GPU 架构和 cuda 编程的入门文档资料，学完可以理解 gpu 架构的基本原理和理解 cuda 编程模型（cuda 并行计算的基本流程）。建议当作学习 cuda 高性能计算编程的第一篇文档（文章）。
- [CUDA Tutorial](https://cuda-tutorial.github.io/): CUDA 教程，分成四部分：CUDA 基础、GPU 硬件细节、最近的特性和趋势和基于任务的编程实例，提供了完整清晰的 PDF 文档和 cuda 代码实例。**建议当作系统性学习 cuda 编程的教程**。
- [learn-cuda](https://github.com/rshipley160/learn-cuda?tab=readme-ov-file): 完整的 cuda 学习教程，包含高级异步方法内容，特点是有性能实验的代码实例。建议当作学习 cuda 高级特性的教程。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)：内容很全，直接上手学习比较难，建议当作查缺补漏和验证细节的 cuda 百科全书，目前版本是 12.6。
- 《CUDA C 编程权威指南》：翻译的国外资料，说实话很多内容翻译的非常不行，我最开始跟着这个学习的，学了一周，只是了解了线程、内存概念和编程模型的概述，但是细节和系统性思维没学到，而且翻译的不行，内容也比较过时，完全不推荐，我已经替大家踩过坑了。
- 《CUDA 编程：基础与实践\_樊哲勇》：国内自己写的教材，我查资料时候挑着看了一点，基本逻辑是通的，虽然很多原理、概念都讲的特别啰嗦，但实践需要的关键知识点都有讲到，想学中文教程的，可以当作当作了解一个方向的快速阅读资料。
- [CUDA-Kernels-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes/tree/main)： CUDA 内核编程笔记及实战代码，有很强的实践性，后期可以重点学习，我也准备认真看下代码和文档。

2, `cuda/triton` 编写 `kernel` 笔记资料：

- 最基本的通用矩阵乘法（gemm）：https://zhuanlan.zhihu.com/p/657632577
- [kernl](https://github.com/ELS-RD/kernl/tree/main): 提供了一些 llm 的 triton 版 kernels
- [unsloth](https://github.com/unslothai/unsloth/tree/main)：专注于大型语言模型推理加速的微调训练方向。提供了开源版本，可以显著提高训练效率，减少内存使用，并且支持 NVIDIA、Intel 和 AMD 的 GPU。Unsloth 的主要特点包括使用 OpenAI 的 Triton 语言重写所有内核。Gemma LLMs 速度提高 2-5 倍，内存减少 `80%` 内核基于 triton 实现。
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main): 用于训练的高效 triton 内核实现。
- [Efficient-LLM-Inferencing-on-GPUs](https://github.com/yinuotxie/Efficient-LLM-Inferencing-on-GPUs/tree/main): README 图片不错，改天看看。

3, Star History Chart：

[![Star History Chart](https://api.star-history.com/svg?repos=HarleysZhang/llm_note)](https://api.star-history.com/svg?repos=HarleysZhang/llm_note)

## 参考资料

- [CUDA-Kernels-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes/tree/main)
- [CUDA and Applications to Task-based Programming](https://cuda-tutorial.github.io/)
- [transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363)
- [CUDATutorial](https://github.com/RussWong/CUDATutorial/tree/main)
- [NVIDIA CUDA Knowledge Base](https://github.com/rshipley160/learn-cuda/wiki)
- [cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming/tree/master)
- [GitHub Repo for CUDA Course on FreeCodeCamp](https://github.com/Infatoshi/cuda-course/tree/master)
