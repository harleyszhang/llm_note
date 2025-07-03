- [摘要](#摘要)
- [引言](#引言)
- [2. 架构](#2-架构)
  - [2.1 基础架构](#21-基础架构)
    - [2.1.1 多头潜在注意力机制 MLA](#211-多头潜在注意力机制-mla)
    - [2.1.2 无辅助损失负载均衡的 DeepSeekMoE](#212-无辅助损失负载均衡的-deepseekmoe)
  - [2.2 多 token 预测（MTP）](#22-多-token-预测mtp)
- [3. 基础设施](#3-基础设施)
  - [3.1 计算集群](#31-计算集群)
  - [3.2 训练框架](#32-训练框架)
    - [3.2.1 双流水与计算-通信重叠](#321-双流水与计算-通信重叠)
    - [3.2.2 跨节点全通通信的高效实现](#322-跨节点全通通信的高效实现)
    - [3.2.3 极低内存开销的优化方案](#323-极低内存开销的优化方案)
  - [3.3 FP8 训练](#33-fp8-训练)
- [参考资料](#参考资料)

## 摘要

DeepSeek-V3，是一个强大的混合专家（MoE）语言模型，总参数量达 6710 亿（`671B`），每个 token 激活 370 亿（`37B`）参数。为实现高效推理和经济高效的训练，DeepSeek-V3 采用了多头潜在注意力（`MLA`）和 `DeepSeekMoE` 架构，这些架构已在 DeepSeek-V2 中得到充分验证。

此外，**DeepSeek-V3 首创了无辅助损失的负载均衡策略，并设置了多 token 预测训练目标（`MTP`）以获得更强性能**。论文在 14.8 万亿个多样且高质量的 token 上对 DeepSeek-V3 进行了预训练，随后通过监督微调和强化学习阶段充分释放其潜力。全面评估表明，DeepSeek-V3 优于其他开源模型，性能可比肩领先的闭源模型。尽管表现出色，DeepSeek-V3 的完整训练仅需 278.8 万 H800 GPU 小时。其训练过程也异常稳定，在整个训练周期中未出现任何不可恢复的损失突增或执行回滚操作。

## 引言

DeepSeek-V3 在 DeepSeek-V2 提出的多头潜在注意力机制（MLA）和 DeepSeekMoE 架构基础上，引入了两项创新策略以进一步提升模型能力：
- 首创**无辅助损失的负载均衡策略**（Wang 等，2024a），旨在最小化传统负载均衡方法对模型性能的负面影响；
- 采用多令牌预测（multi-token prediction `MTP`）训练目标，经实证可显著提升模型在评估基准上的综合表现。

为了实现更高效训练，论文支持了 **FP8 混合精度训练**并对训练框架进行了全面优化。通过支持 FP8 计算与存储，我们同时实现了**训练加速和 GPU 显存占用降低**。针对训练框架，论文设计了** `DualPipe` 算法以实现高效流水线并行**，该算法具有更少的气泡间隙，并通过**计算-通信重叠**隐藏了训练过程中的大部分通信开销。这种重叠设计确保了随着模型规模的进一步扩大，只要保持恒定的计算通信比，我们仍可在跨节点部署**细粒度专家模块**的同时，实现近乎零成本的全员通信开销。此外，我们还开发了高效的**跨节点全员通信内核**，以充分利用 `InfiniBand`（IB）和 `NVLink` 的带宽优势。

DeepSeek-V3 的核心贡献包括：

**1. 架构：创新的负载均衡策略与训练目标**
- 在 DeepSeek-V2 高效架构的基础上，我们首创了无辅助损失的负载均衡策略（an auxiliary-loss-free strategy for load balancing），该策略能最大限度减少因促进负载均衡而导致的性能下降。
- 研究了多令牌预测（`MTP`）目标并证实其对模型性能的增益作用。该技术还可用于推测解码以实现推理加速。

**2. 预训练：追求极致的训练效率**
- 设计了 **FP8 混合精度训练框架**，并首次在超大规模模型上验证了 FP8 训练的可行性与有效性。
- 通过算法、框架与硬件的协同设计，突破了跨节点 `MoE` 训练中的通信瓶颈，实现了近乎完全的**计算-通信重叠**。这一突破显著提升了训练效率并降低了训练成本，使我们能够在无需额外开销的情况下进一步扩大模型规模。
- 仅以经济高效的 266.4 万 H800 GPU 小时成本，完成了 DeepSeek-V3 在 14.8 万亿 token 上的预训练，打造出当前最强的开源基础模型。后续训练阶段仅需 10 万 GPU 小时即可完成。

**3. 后训练阶段：基于 DeepSeek-R1 的知识蒸馏**

- 创新性地提出从长思维链（`CoT`）模型（具体采用 DeepSeek R1 系列中的某个模型）向标准 LLMs（特别是 DeepSeek-V3）蒸馏推理能力的方法。该流程巧妙地将 R1 的验证与反思机制融入 DeepSeek-V3，显著提升了其推理性能。同时，我们还保持了对 DeepSeek-V3 输出风格与长度的控制能力。

## 2. 架构

DeepSeek-V3 基础架构示意图如下图所示。延续 DeepSeek-V2 的设计，依然采用 MLA 和 DeepSeekMoE 架构，DeepSeek-V3 模型配置方案 DeepSeek-V2 一致。

![DeepSeekV3_archetecture](../../images/DeepSeekV3/DeepSeekV3_archetecture.png)

### 2.1 基础架构

#### 2.1.1 多头潜在注意力机制 MLA

#### 2.1.2 无辅助损失负载均衡的 DeepSeekMoE

**1, Basic Architecture of DeepSeekMoE DeepSeekMoE 的基础架构**

**2, Auxiliary-Loss-Free Load Balancing 无辅助损失负载均衡**

**3, Complementary Sequence-Wise Auxiliary Loss 互补序列级辅助损失**

**4, Node-Limited Routing 节点受限路由**

与 DeepSeek-V2 采用的**设备受限路由**类似，DeepSeek-V3 同样采用受限路由机制来**限制 MoE 相关的通信成本**（节点间的通信成本很大）。

`Node-Limited Routing` 策略简单来说，**每个 `token` 最多只能被发送到 $M=4$ 个节点上**。具体来说，对于每个 token：首先，根据每个节点上分布的专家的最高 $K_r/M$ 亲和度分数之和来选择节点。然后，在 $M$ 个设备上的专家中执行 `top-K` 选择，每个设备选择 $\frac{K_r}{M}$ 个专家。在此约束下，混合专家训练框架几乎可以实现计算与通信的完全重叠。

**参数说明**:
- $K_r$: 每个 token 路由到的专家总数: $8$，对应 config.json 文件中的 `num_experts_per_tok` 字段。
- $M$: 每个 token 最多路由到的节点数: $4$。对应 config.json 文件中的 `topk_group` 字段。
- $K_r/M$: 每个节点平均分配的专家数。
- 节点选择分数 = Σ(节点 $i$ 上专家的最高 $K_r/M$ 亲和度分数)

选择 `top-k` 专家的路由策略代码如下所示:

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        #############省略#########
     def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
            ### 选择 M (top_k_group)个 Group
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

        return topk_idx, topk_weight
```

**5, No Token-Dropping 路由阶段无令牌丢弃机制**

得益于高效的负载均衡策略，DeepSeek-V3 在整个训练过程中保持了良好的负载平衡，因此在训练阶段不会丢弃任何 token。此外，还实施了**特定的部署策略以确保推理阶段的负载均衡**，使得 DeepSeek-V3 在推理过程中同样不会丢失 `token`。

### 2.2 多 token 预测（MTP）

受 Gloeckle 等人（2024）启发，我们为 DeepSeek-V3 研究并设定了多令牌预测（`MTP`）目标，该目标**将预测范围扩展到每个位置的多个未来 token**，它有以下作用：
- MTP 目标使训练信号更加密集，可能提高数据效率；
- MTP 可能使模型能够预先规划其表示，以更好地预测未来令牌。

图 3 展示了论文的 MTP 实现方式。与 Gloeckle 等人（2024）使用独立输出头并行预测 $D$ 个额外令牌不同，论文采用顺序预测额外令牌的方式，并在每个预测深度保持完整的因果链。本节将详细介绍我们的 MTP 实现方案。


![MTP](../../images/DeepSeekV3/MTP.png)
> 图 3：多 token 预测（MTP）实现示意图。

省略


## 3. 基础设施

### 3.1 计算集群

DeepSeek-V3 在配备 2048 块 NVIDIA `H800 GPU` 的集群上进行训练。H800 集群中的每个节点包含 $8$ 块通过 `NVLink` 和 `NVSwitch` 互联的 GPU，不同节点间则采用 `InfiniBand(IB)` 网络互连以实现高效通信。

### 3.2 训练框架

DeepSeek-V3 的训练由 `HAI-LLM` 框架支持，这是 DeepSeek 公司自主研发的高效轻量级训练框架。整体上，DeepSeek-V3 采用了 $16$ 路流水线并行(`PP`)(Qi 等, 2023a)、跨 $8$ 个节点的 $64$ 路专家并行(`EP`)(Lepikhin 等, 2021)以及 `ZeRO-1` 数据并行(`DP`)(Rajbhandari 等, 2020)的混合并行策略。

为高效训练 DeepSeek-V3 模型，我们实施了精细的工程优化方案。

1. 首先，创新设计了 `DualPip`e 算法实现高效的流水线并行——相较于现有 PP 方法，该算法能显著减少流水线气泡。更重要的是，它**通过重叠前向传播与反向传播过程中的计算与通信阶段**，有效解决了跨节点专家并行带来的沉重通信开销难题。
2. 其次，开发了高效的跨节点全连接通信内核，可充分榨取 `InfiniBand` 与 `NVLink` 带宽潜力，同时节省专用于通信的流式多处理器（SM）资源。
3. 最后，我们对训练过程中的内存占用进行了极致优化，从而实现了无需依赖昂贵张量并行（`TP`）技术即可完成 DeepSeek-V3 的训练。

#### 3.2.1 双流水与计算-通信重叠

![DualPipe](../../images/DeepSeekV3/DualPipe.png)
> 图 4：单个前向与反向分块的重叠策略（Transformer 块边界未对齐）。橙色表示前向计算，绿色表示"输入反向计算"，蓝色表示"权重反向计算"，紫色表示流水线并行通信，红色表示同步屏障。全连接通信与流水线通信均可被完全隐藏。

在 DeepSeek-V3 中，跨节点专家并行引入的通信开销导致计算与通信比低至约 1:1（跨节点的 EP 带来的通信开销居然占比 50%！）。为解决这一挑战，论文设计了名为 `DualPipe` 的创新流水线并行算法，该算法不仅通过有效重叠前向/反向计算与通信阶段来加速模型训练，同时显著减少了流水线气泡。

DualPipe 的核心思想是通过**成对的前向与反向计算块**实现计算与通信的重叠。具体而言，我们将每个计算块划分为四个组件：**注意力机制、全分发通信（all-to-all dispatch）、多层感知机（MLP）以及全聚合通信（all-to-all combine）**。特别地，对于反向计算块，注意力机制和 MLP 会进一步拆分为两部分——输入反向传播与权重反向传播，如 ZeroBubble（Qi 等人，2023b）所述。

此外，还设置了流水线并行（PP）通信组件。如图 4 所示，针对成对的前向-反向计算块，我们重新编排这些组件并手动调节 GPU 流式多处理器（SM）在通信与计算之间的分配比例。通过这种重叠策略，可以确保全分发通信和流水线并行通信在执行过程中完全被隐藏。

基于高效的重叠策略，图 5 展示了完整的 DualPipe 调度方案：采用双向流水线调度机制，从流水线两端同时输入微批次数据，使得绝大部分通信操作能够实现完全重叠。 这种重叠设计还确保，随着模型规模进一步扩大，只要我们**保持恒定的计算与通信比率**，仍可在各节点间部署细粒度专家模块，同时实现近乎零的全员通信开销。

![Example_DualPipe_scheduling](../../images/DeepSeekV3/Example_DualPipe_scheduling.png)

此外，即便在通信负担较轻的通用场景下，DualPipe 仍展现出效率优势。表 2 汇总了不同流水线并行方法的流水线气泡与内存使用情况。数据显示，相较于 ZB1P（Qi 等，2023b）和 1F1B（Harlap 等，2018），DualPipe 在仅增加 $\frac{1}{PP}$ 倍峰值激活内存的同时，显著降低了流水线气泡。虽然 DualPipe 需要维护两份模型参数副本，但由于训练时采用较大的 EP 尺寸，这并未显著增加内存消耗。与 Chimera（Li 和 Hoefler，2021）相比，DualPipe 仅要求流水线阶段数和微批次量能被 2 整除，而无需微批次量被流水线阶段数整除。更重要的是，DualPipe 的气泡与激活内存均不会随微批次数量增加而增长。

![Table2](../../images/DeepSeekV3/Table2.jpg)

表 2：不同管道并行方法间管道气泡与内存使用量的对比。 $F$ 表示前向块执行时间，$B$ 表示完整反向块执行时间， $W$ 表示"权重反向"块执行时间， $F\&B$ 表示两个相互重叠的前向与反向块执行时间。

#### 3.2.2 跨节点全通通信的高效实现

为确保 DualPipe 具备足够的计算性能，论文定制了**高效的跨节点全通信内核**（包括分发与聚合），以节省专用于通信的流处理器数量。该内核实现与 MoE 门控算法及集群网络拓扑协同设计。

具体而言，在我们的集群中，跨节点 GPU 通过 InfiniBand 全互联，节点内通信则经由 NVLink 处理。NVLink 提供 160GB/s 的单向带宽，约为 InfiniBand（50GB/s）的 $3.2$ 倍。为有效利用两种互连的带宽差异，我们将每个 `token` 的分发目标节点限制在最多 $4$ 个，从而降低 InfiniBand 流量。

每个 `token` 完成路由决策后，会先通过 `InfiniBand` 传输至目标节点中具有相同节点内索引的 GPU。抵达目标节点后，系统将确保其立即通过 NVLink 转发至托管目标专家的特定 GPU，且不会被后续到达的 token 阻塞。 通过这种方式，InfiniBand 与 NVLink 的通信实现了完全重叠，每个令牌可高效平均选择每节点 $3.2$ 个专家，且不会产生额外的 NVLink 通信开销。这意味着，尽管 DeepSeek-V3 实际仅选择 8 个路由专家，但在保持相同通信成本的前提下，该数量可扩展至最多 13 个专家（4 节点×3.2 专家/节点）。总体而言，在此通信策略下，仅需 20 个流式多处理器即可充分释放 InfiniBand 与 NVLink 的带宽潜力。

具体而言，论文采用**线程束专业化**（`warp specialization`）技术（Bauer 等人，2014 年），将 $20$ 个流式多处理器（SMs）划分为 $10$ 个通信通道（communication channels）。`dispatching` 过程分为 3 步：
1. InfiniBand 发送
2. IB 至 NVLink 转发
3. NVLink 接收分别由专用线程束处理。

各通信任务分配的 `warps` 数量会根据所有 `SMs` 的实际工作负载动态调整。

同样地，在 `combining` 也分为 3 步：
1. NVLink 发送
2. NVLink 至 IB 转发与累加
3. IB 接收与累加也由动态调整的线程束处理。

另外，dispatching 和 combining 内核均与计算流重叠执行，因此我们还需考量其对其他 SM 计算内核的影响。特别地，我们采用定制化 `PTX`（并行线程执行）指令并自动调优通信分块大小，这样可以**显著降低了二级缓存的使用量及对其他 `SM` 的干扰**。

#### 3.2.3 极低内存开销的优化方案

为降低**训练过程**中的内存占用，我们采用了以下技术手段。

1. RMSNorm 与 MLA 上投影层的重计算。
2. CPU 中的指数移动平均。
3. 多令牌预测的共享嵌入层与输出头。

### 3.3 FP8 训练

![The overall mixed precision framework with FP8 data format](../../images/DeepSeekV3/FP8_framework.png)
> 图 6：采用 FP8 数据格式的整体混合精度框架。为清晰起见，图中仅展示了线性算子部分。

虽然低精度训练前景广阔，但其应用常受限于激活值、权重和梯度中的异常值问题（Sun 等，2024；He 等；Fishman 等，2024）。为应对这一挑战并有效扩展 FP8 格式的动态范围，论文引入了一种细粒度的量化策略：采用 $1\times N_c$ 元素的 `tile-wise` 分组或 $N_c \times N_c$ 元素的 `block-wise` 分组。通过论文设计的**高精度累加过程**，相关反量化开销得以大幅降低，这是实现精确 FP8 通用矩阵乘法（GEMM）的关键所在。此外，为进一步降低混合专家训练中的内存与通信开销，论文采用 `FP8` 格式缓存并分发激活值，同时以 `BF16` 精度存储优化器状态。

我们在与 DeepSeek-V2-Lite 和 DeepSeek-V2 规模相近的两个模型上验证了所提出的 FP8 混合精度框架，训练约 1 万亿 token（详见附录 B.1）。值得注意的是，相较于 BF16 基线方案，采用 FP8 训练的模型相对损失误差始终保持在 0.25%以下，这一数值完全处于训练随机性可接受范围内。

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)