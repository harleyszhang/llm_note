### 1. Expert 负载不均衡

在 `MOE` 模块中，`Router` 负责决定发送 token 到 指定 `expert` 中，在实际模型推理过程中，可能某些 experts 接收 tokens 过多，某些 experts 接收 tokens 过少，这种现象叫做 **expert 负载不均衡**。这种情况不仅脱离了 MoE 的设计初衷（术业有专攻），也影响了 MoE 计算效率（分布式训练推理中各卡通讯时的负载不均衡, 各卡接收的 tokens 不均衡），因此需要解决这个问题。

虽然基于 MoE 的算法开辟了巨大的扩展机会，但 MoE 的动态特性带来了以前大多数深度学习算法和系统中未曾见过的基本系统级挑战。具体来说，每个 MoE 层由一定数量的并行专家组成，这些专家分布在加速器（本文中为 GPU）上，每个 GPU 根据智能门控函数将每个输入数据分派给最匹配的几个专家，并将相应的输出返回以进行组合。这意味着专家的工作负载从根本上是不确定的——它取决于输入数据和门控函数。在实际中，这两者都在每次迭代中发生变化。在我们的实验中（见图 1），单个训练中的工作负载变化高达 $4.38×$，不同层的工作负载也不同。

### 2. MoE 计算过程

DeepSeek-V3 模型配置文件中 `"num_experts_per_tok": 8`,表示对于每个 token 都会激活 8 个 expert 做计算。

### 3. MoE 并行训练/推理策略

MoE Transformer layer 的并行方式一般如下：
- 非专家部分（注意力机制）采用张量并行和数据并行（`TP + DP`）；
- 对于 MoE 部分，采取 EP + DP。

`MoE` 并行的核心思想：专家并行 = 专家分布 + 动态路由 + `All2All` 通信。

1, 背景知识:
- `world size`：代表将要参与训练的进程数（或者计算设备数）。
- `rank`: 每个进程都会被分配一个 `rank`，rank 是一个介于 `0` 和 `world size-1` 之间的数字，该数字在作业中是唯一的。它作为进程标识符，并用于代替地址，将张量发送到指定的 `rank`（进程）。

2, `EP + DP` 的并行策略：

假设每个 `MoE 层有若干个专家（统称其为一套专家），如何把这一套专家分布排列到若干 GPU 上呢？可以先定好要用几块 GPU 装下一套专家（EP: ep_world_size），然后可确认全局上共有多少套专家副本在跑（DP: ep_dp_world_size）。假设一共 8 张 GPU，则：
- `ep_world_size = 4`: 表示希望用 4 块 GPU 装下一套完整的专家。ep_group = 8 / ep_world_size = 8 /4 = 2，即一共 $2$ 个专家组。我们需要在每个专家组内做 All-to-All 通信，将 token 发送去对应的专家。
- `ep_dp_world_size = 2`: MoE 层的数据并行的大小。构成 ep_dp_group 的条件不仅是 expert 相同，还需要每个 expert 接受到的的 batch 数据不同。
- 同一个 TP group 中的所有 TP rank 处理相同的数据，在固定 world size 的情况下，**开启 TP 会使 DP 变为原来的 1/TP**。

### 4. MoE 通讯量分析

#### EP 通讯量分析

### 5. SGLang 的 EP 特性

###  6. MoE 论文速览

#### Switch Transformers

### 参考资料

- [图解大模型训练系列之：DeepSpeed-Megatron MoE并行训练（原理篇）](https://zhuanlan.zhihu.com/p/681154742)
- [MoE 训练到底是开 TP 还是 EP？](https://zhuanlan.zhihu.com/p/13997146226): moe 的 ep 通讯量分析.
- [MoE Inference On AnyScale MoE-On-AnyScale](https://zhuanlan.zhihu.com/p/28680264165)
- [TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE](https://arxiv.org/pdf/2206.03382)