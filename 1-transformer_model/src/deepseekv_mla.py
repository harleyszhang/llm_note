import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
import math

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
# copied formed https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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


class DeepseekV2MLA(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        # MHA 初始化相关
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.v_head_dim = config.v_head_dim

        self.o_proj = nn.Linear(
            self.v_head_dim * self.num_heads, 
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.attention_dropout = config.attention_dropout
        self.training = False
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        # MLA 相关 part1: 压缩
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank)
        self.q_down_rmsnorm = DeepseekV2RMSNorm(self.q_lora_rank)
        
        self.kv_down_proj = nn.Linear(
            self.hidden_size, 
            self.kv_lora_rank + config.qk_rope_head_dim
        )
        self.kv_down_rmsnorm = DeepseekV2RMSNorm(self.kv_lora_rank)
        
        # MLA 相关 part2: 解压缩
        self.q_head_dim = self.qk_nope_head_dim  + self.qk_rope_head_dim
        self.q_up_proj = nn.Linear(
            self.q_lora_rank, 
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        # qk_nope_head_dim = q_head_dim - qk_rope_head_dim
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        
        # MLA 相关 part3: 切片 q k 张量，以及 rope 旋转位置编码
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        ) 

    def forward(self, hidden_states, position_ids, casual_mask=None):
        batch_size, q_len, hidden_size = hidden_states.shape

        # 1，q 压缩和解压缩，以及 split to q_nope, q_rope
        q = self.q_up_proj(
            self.q_down_rmsnorm(self.q_down_proj(hidden_states))
        )

        q = q.view(batch_size, q_len, self.num_heads, self.q_head_dim).transpose(1,2)
        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim = -1,
        )

        # 2, kv 压缩和解压缩
        kv_down = self.kv_down_proj(hidden_states)
        
        # compressed_kv 压缩后的 kv 张量
        compressed_kv, k_rope = torch.split(
            kv_down,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim = -1,
        )
        # num_heads = 1 后续广播其它 heads 上
        k_rope = k_rope.view(batch_size, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # 对 compressed_kv 解压缩
        kv = (
            self.kv_up_proj(self.kv_down_rmsnorm(compressed_kv))
            .view(batch_size, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim = -1,
        )

        # 3, 计算 cos 和 sin，并应用 rope 旋转位置编码
        kv_seq_len = value_states.shape[-2] # shape (b, nums_head, seq_len, v_head_dim)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)

        # 4, 执行 self-attention 计算
        query_states = torch.concat([q_nope, q_rope], dim=-1)
        key_states = torch.concat(
            [k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], 
            dim=-1
        )
        # qk^t
        scores = (
            torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.q_head_dim)
        )

        if casual_mask is not None:
            scores = scores.masked_fill(casual_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        ) # attn_weights shape: [bs, num_heads, seq_len, seq_len]
        
        attn_output = torch.matmul(attn_weights, value_states) # shape: [bs, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.num_heads * self.v_head_dim)

        # 5, MLA 输出映射
        output = self.o_proj(attn_output)

        return output, attn_weights
    
# 写一个测试函数
def test_mla():
    config = DeepseekV2Config(
        rope_theta=128000,
        max_position_embeddings=1024,
        hidden_size = 5120
    )
    
    mla = DeepseekV2MLA(config)
    # shape: (batch_size, sequence_length, hidden_size)
    embedding_states = torch.randn(2, config.max_position_embeddings, config.hidden_size)
    position_ids = torch.arange(
        config.max_position_embeddings,
    ).unsqueeze(0).expand(embedding_states.size(0), -1) # (batch_size, seq_len)
    
    attn_output, attn_weights = mla(embedding_states, position_ids=position_ids)
    print(attn_output.shape)
    print(attn_weights.shape)

if __name__ == "__main__":
    test_mla()

"""
输出结果:
torch.Size([2, 1024, 5120])
torch.Size([2, 128, 1024, 1024])
"""