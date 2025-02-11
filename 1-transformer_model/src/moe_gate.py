import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from dataclasses import dataclass

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        
        # 静态化推理配置（假设配置固定）
        self.inference_norm = self.norm_topk_prob and (self.top_k > 1)
        self.use_group_limited = (self.topk_method == "group_limited_greedy")

        # 门控权重
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.inference_mode()  # 禁用梯度与训练逻辑
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, h)
        
        # 门控分数计算（保持原始数据类型）
        logits = F.linear(hidden_states, self.weight)  # [n_tokens, n_experts]
        scores = logits.softmax(dim=-1)  # 自动推断 dtype

        # Top-K 选择（静态分支）
        if self.use_group_limited:
            # 分组限制逻辑优化
            group_scores = scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, -1, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1)
            scores = scores.masked_fill(~score_mask.bool(), 0.0)
        
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 权重归一化（静态分支）
        if self.inference_norm:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, None  # aux_loss 始终为 None

@dataclass
class DeepseekV2Config:
    # 1, Position Config
    max_position_embeddings: int = 163840
    vocab_size: int = 102400

    # 2, MLA Config
    # down_linear config
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512

    # head_dim、heads and hidden_size config
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    
    attention_bias: bool = False

    attention_dropout: float = 0.1
    # rope config
    rope_theta: float = 10000

    # 3, MOE Config
    n_group: int = 8
    n_routed_experts: int = 160
    num_experts_per_tok: int = 6
    topk_group: int = 3
    routed_scaling_factor: float = 1.0
    scoring_func: str="softmax"
    topk_method: str="greedy"
    norm_topk_prob: bool = True

# 初始化配置
config = DeepseekV2Config()

# 模拟输入，CPU 电脑可直接跑，去除了 cuda 设备限制代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_states = torch.randn(32, 64, 5120, device=device)

# 创建模块
moe_gate = MoEGate(config)  # 半精度推理

# gate 网络推理
topk_idx, topk_weight, _ = moe_gate(hidden_states)

print("topk_idx shape ", topk_idx.shape) # 32 * 64 = 2048 个 tokens
print("topk_weight shape", topk_weight.shape)

"""
# 输出如下，表示每个 token 会激活 6 个专家参与计算
topk_idx shape  torch.Size([2048, 6]) 
topk_weight shape torch.Size([2048, 6])
"""