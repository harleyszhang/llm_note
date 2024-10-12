# llm_note overview
LLM notes, including model inference, transformer model structure, and lightllm framework code analysis notes

## 一，transformer 模型结构

- [transformer论文解读](./1_transformer_model/transformer论文解读.md)
- [transformer模型代码实现](./1_transformer_model/transformer模型代码实现.md)
- [LLaMA 及其子孙模型详解](./1_transformer_model/llama及其子孙模型概述.md)
- [vit论文速读](1_transformer_model/vit论文速读.md)
- [gpt1-3论文解读](./1_transformer_model/gpt1-3论文解读.md)
- [llm参数量-计算量-显存占用分析](./1_transformer_model/llm参数量-计算量-显存占用分析.md)
- [llm推理latency分析](1_transformer_model/llm推理latency分析.md)

## 二，大语言模型压缩


## 三，大语言模型推理及部署（服务化）

- [llm推理揭秘论文翻译](3_llm_infer_deploy/llm推理揭秘论文翻译.md)
- [llm综合分析论文翻译](3_llm_infer_deploy/llm综合分析论文翻译.md)

**DeepSpeed 框架学习笔记**：

- [DeepSpeed:通过系统优化和压缩加速大规模模型推理和训练](./3_llm_infer_deploy/deepspeed_note/deepspeed-通过系统优化和压缩加速大规模模型推理和训练.md)
- [DeepSpeed推理:具有定制推理内核和量化支持的多GPU推理](./3_llm_infer_deploy/deepspeed_note/deepspeed推理-具有定制推理内核和量化支持的多GPU推理.md)

## 四，系统化优化

图优化、算子融合、深度学习推理框架系统层面的优化。

## 五，高性能计算

英伟达 gpu cuda 高性能计算编程学习资料推荐：

- [GPU Architecture and Programming](https://homepages.laas.fr/adoncesc/FILS/GPU.pdf): 了解 GPU 架构和 cuda 编程的入门文档资料，学完可以理解 gpu 架构的基本原理和理解 cuda 编程模型（cuda 并行计算的基本流程）。建议当作学习 cuda 高性能计算编程的第一篇文档（文章）。
- [CUDA Tutorial](https://cuda-tutorial.github.io/): CUDA 教程，分成四部分：CUDA 基础、GPU 硬件细节、最近的特性和趋势和基于任务的编程实例，提供了完整清晰的 PDF 文档和 cuda 代码实例。**建议当作系统性学习 cuda 编程的教程**。
- [learn-cuda](https://github.com/rshipley160/learn-cuda?tab=readme-ov-file): 完整的 cuda 学习教程，包含高级异步方法内容，特点是有性能实验的代码实例。建议当作学习 cuda 高级特性的教程。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)：内容很全，直接上手学习比较难，建议当作查缺补漏和验证细节的 cuda 百科全书，目前版本是 12.6。
- 《CUDA C编程权威指南》：翻译的国外资料，说实话很多内容翻译的非常不行，我最开始跟着这个学习的，学了一周，只是了解了线程、内存概念和编程模型的概述，但是细节和系统性思维没学到，而且翻译的不行，内容也比较过时，完全不推荐，我已经替大家踩过坑了。
- 《CUDA 编程：基础与实践_樊哲勇》：国内自己写的教材，我查资料时候挑着看了一点，基本逻辑是通的，虽然很多原理、概念都讲的特别啰嗦，但实践需要的关键知识点都有讲到，想学中文教程的，可以当作当作了解一个方向的快速阅读资料。

## 参考资料

- [CUDA and Applications to Task-based Programming](https://cuda-tutorial.github.io/)
- [transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363)
- [CUDATutorial](https://github.com/RussWong/CUDATutorial/tree/main)
- [NVIDIA CUDA Knowledge Base](https://github.com/rshipley160/learn-cuda/wiki)
- [cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming/tree/master)
- [GitHub Repo for CUDA Course on FreeCodeCamp](https://github.com/Infatoshi/cuda-course/tree/master)
