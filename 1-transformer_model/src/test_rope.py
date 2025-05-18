import torch

def build_rope_cache_fixed(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    base: int = 10000,
    max_position_embeddings: int = 32768,
    dtype: torch.dtype = torch.float32,
):
    """
    构建 RoPE 频率缓存.

    Args:
        batch_size (int): 批大小 B
        seq_len (int): 序列长度 S
        head_dim (int): 每个注意力头维度 H (必须是偶数)
        base (int, optional): 对数基数(常用 10000). Defaults to 10000.
        max_position_embeddings (int, optional): 支持的最长位置. Defaults to 32768.
        dtype (torch.dtype, optional): 输出张量 dtype. Defaults to torch.float32.

    Returns:
        positions_cpu (LongTensor): [B, S] 随机位置 id
        cos (Tensor): [max_position_embeddings, head_dim//2] 余弦频率
        sin (Tensor): [max_position_embeddings, head_dim//2] 正弦频率
    """
    # 1) 为每个 batch 生成位置 id（演示用随机，实际推理时应为 torch.arange）
    positions_cpu = torch.randint(0, seq_len, (batch_size, seq_len), dtype=torch.long)

    # 2) 计算逆频率 inv_freq: [H/2]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    # 3) 生成全局位置索引 t: [max_position_embeddings]
    t = torch.arange(max_position_embeddings, dtype=torch.float32)

    # 4) 外积得到每个位置对应的旋转角度 freqs: [max_position_embeddings, H/2]
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    return positions_cpu, cos, sin

# ========== 简单单元测试 / demo ========== #
torch.manual_seed(0)
B, S, H = 2, 4, 8
pos, cos, sin = build_rope_cache_fixed(B, S, H)

print("positions shape:", pos.shape)
print("cos shape:", cos.shape, " dtype:", cos.dtype)
print("sin shape:", sin.shape, " dtype:", sin.dtype)
print("pos[0]:", pos[0])
print("cos[0, :4]:", cos[0, :4])

"""
positions shape: torch.Size([2, 4])
cos shape: torch.Size([32768, 4])  dtype: torch.float32
sin shape: torch.Size([32768, 4])  dtype: torch.float32
pos[0]: tensor([0, 3, 1, 0])
cos[0, :4]: tensor([1., 1., 1., 1.])
"""

import torch
from typing import Optional, Tuple

def apply_rotary_pos_embedding(
    query: torch.Tensor,          # [B, S, H]
    key:   torch.Tensor,          # [B, S, H]
    cos:   torch.Tensor,          # [L_max, H/2]  — 由 build_rope_cache 预先生成
    sin:   torch.Tensor,          # [L_max, H/2]
    positions_ids: Optional[torch.Tensor] = None,  # [B, S] 或 [S]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 旋转应用到 query / key。
    """
    # 1) 若未显式传入位置 id，则默认按 0..S-1 顺序
    if positions_ids is None:                                      # :contentReference[oaicite:1]{index=1}
        positions_ids = torch.arange(cos.size(0), device=cos.device)

    # 2) 取出对应行；reshape(-1) 兼容批维
    # positions_ids 可能是一维，也可能是二维 ⇒ 统一成 [...,1,-1]
    cos_sel = cos.index_select(0, positions_ids.reshape(-1))
    sin_sel = sin.index_select(0, positions_ids.reshape(-1))

    # 3) 恢复批次形状，并在 head 维前插入空维，便于广播
    target_shape = (*positions_ids.shape, 1, -1)    # [B,S,1,H/2] 或 [S,1,H/2]
    cos_sel = cos_sel.view(target_shape)
    sin_sel = sin_sel.view(target_shape)

    # 若缺少 batch 维，自动 unsqueeze 再广播
    if cos_sel.dim() == 3:                          # [S,1,H/2] 情况
        cos_sel = cos_sel.unsqueeze(0)              # → [1,S,1,H/2]
        sin_sel = sin_sel.unsqueeze(0)

    # 4) 内部旋转函数：偶/奇维度各自为实部/虚部
    def _rotate(t: torch.Tensor) -> torch.Tensor:
        """
        将 (x0,x1,x2,x3,...) 视为复数 (x0+i·x1 , x2+i·x3, ...)
        与 (cos+i·sin) 相乘，再拆成偶/奇分量。
        """
        t1, t2 = t[..., ::2], t[..., 1::2]        # 偶、奇
        rot1 = t1 * cos_sel - t2 * sin_sel        # 实部
        rot2 = t2 * cos_sel + t1 * sin_sel        # 虚部
        return torch.stack((rot1, rot2), dim=-1).flatten(-2)

    # 5) 分别旋转 Q / K
    return _rotate(query), _rotate(key)


import torch
from math import isclose

B, S, num_heads, head_dim = 2, 4, 4, 8, 
# 简易 rope 缓存（完整实现见上一轮 build_rope_cache）
pos = torch.arange(S)
inv_freq = 1 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
freqs = torch.outer(pos, inv_freq)
cos, sin = freqs.cos(), freqs.sin()

# 随机生成 q, k 并应用 RoPE
q = torch.randn(B, S, num_heads, head_dim)
k = torch.randn(B, S, num_heads, head_dim)
q_rot, k_rot = apply_rotary_pos_embedding(q, k, cos, sin, pos)

# 断言形状
assert q_rot.shape == (B, S, num_heads, head_dim)
assert k_rot.shape == (B, S, num_heads, head_dim)

# 慢速公式验证单点正确性
def naive_rot(t, i):
    c, s = cos[i], sin[i]
    t1, t2 = t[..., ::2], t[..., 1::2]
    rot = torch.stack((t1 * c - t2 * s, t2 * c + t1 * s), -1).flatten(-2)
    return rot

i = 1  # 第 1 个 token
gold = naive_rot(q[0, i], i)
assert torch.allclose(q_rot[0, i], gold, atol=1e-5)
print("通过！q_rot[0,1] ≈ gold，形状一致")