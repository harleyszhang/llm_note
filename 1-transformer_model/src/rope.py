import torch, math
import torch.nn as nn
import torch.nn.functional as F
import llama

def compute_theta(dim: int, base: float = 10000.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    计算旋转位置编码中的 Theta 角度值。

    参数：
    - d (int): 嵌入向量的维度（必须为偶数）。
    - base (float): 基础频率参数, 默认为10000.0。
    - device (torch.device): 计算设备, 默认为CPU。

    返回：
    - torch.Tensor: 包含Theta值的1D张量, 形状为 [d/2]。
    """
    if dim % 2 != 0:
        print("嵌入维度 dim 必须为偶数")
    i = torch.arange(1, (dim//2) + 1, dtype=torch.float32, device=device)
    theta_i = base ** (-2*(i - 1) / dim)

    return theta_i

def precompute_freqs_cis(dim: int, seq_len: int, base: float = 10000.0, device: torch.device = torch.device('cpu')):
    theta = compute_theta(dim, base, device) # theta 角度值序列，向量, 大小为 dim // 2
    m = torch.arange(seq_len, device=device) # # token 位置值序列，向量，大小为 seq_len
    m_theta = torch.outer(m, theta) # 所有 token 位置的所有 Theta 值范围, 矩阵，尺寸为 [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(m_theta), m_theta) # e^{i*m*\theta}，本质上是旋转矩阵
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[1],x.shape[-1]), "the last two dimension of freqs_cis, x must match"
    shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, device: torch.device = torch.device('cpu')):
    """
    参数:
        - x_q(torch.Tensor): 实际上是权重 W_q * 词嵌入向量值, 来自上一个线性层的输出, 形状为 [batch_size, seq_len, n_heads, head_dim]
        - x_k(torch.Tensor): 实际上是权重 W_k * 词嵌入向量值, 来自上一个线性层的输出, 形状为 [batch_size, seq_len, n_heads, head_dim]
        - freqs_cis (torch.Tensor): 频率复数张量, 形状为 [max_seq_len, head_dim]
    返回:
        - Tuple[torch.Tensor, torch.Tensor]: 旋转编码后的查询和键张量
    """
    # 实数域张量转为复数域张量
    xq_reshape = xq.reshape(*xq.shape[:-1], -1, 2) # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim//2, 2] 
    xk_reshape = xk.reshape(*xk.shape[:-1], -1, 2) # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim//2, 2] 
    xq_complex = torch.view_as_complex(xq_reshape) # 复数形式张量
    xk_complex = torch.view_as_complex(xk_reshape) # 复数形式张量

    # 旋转矩阵（freqs_cis）的维度在序列长度（seq_len，维度 1）和头部维度（head_dim，维度 3）上需要与嵌入的维度一致。
    # 此外，freqs_cis 的形状必须与 xq 和 xk 相匹配，因此我们需要将 freqs_cis 的形状从 [max_seq_len, head_dim] 调整为 [1, max_seq_len, 1, head_dim]。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex) # [max_seq_len, 1, 1, dim // 2]

    # 应用旋转操作，并将结果转回实数域
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3) # flatten(2) 将后面两个维度压成一个维度
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    """compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    
    def __init__(self, dim, max_seq_len, n_heads):
        super(Attention, self).__init__()
        self.wq = nn.Linear(dim, dim)  # Q 线性变换层
        self.wk = nn.Linear(dim, dim)  # K 线性变换层
        self.wv = nn.Linear(dim, dim)  # V 线性变换层
        self.out = nn.Linear(dim, dim)  # 输出线性变换层

        self.softmax = nn.Softmax(dim = -1)
        self.dim  = dim                # 模型中嵌入层尺寸
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = dim // n_heads # 每个 head 的尺寸

    def forward(self, x: torch.Tensor, start_pos=0, inference=True, mask=None):
        # 1, 计算 xq xk xv 并调整它们的形状为 [bs, seq_len, n_heads, head_dim]
        bs, seq_len, dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_heads, self.head_dim)
        
        if inference:
            # 2, Compute rotation matrix for each position in the sequence
            freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2)
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
            # 3, Apply RoPE to Queries and Keys embeddings
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
            
        # 4, 因为 self-attention 算子中矩阵乘法操作处理的事 [seq_len, d_model] 维度，所以需要 
        # 对查询（queries）、键（keys）和值（values）进行转置操作，重新调整它们的形状，
        # 使“头”（heads）位于维度 1，而“序列”（seq）位于维度 2。
        querys = xq.transpose(1, 2)          # [bs, seq_len, n_heads, head_dim] -> [bs, n_heads, seq_len, head_dim]
        keys = xk.transpose(1, 2)            # keys:[bs,n_heads,seq_len,head_dim]
        values = xv.transpose(1, 2)          # values:[bs,n_heads,seq_len,head_dim]

        # 5, self-attention 算子的内部操作
        scores = torch.matmul(querys, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # 计算 Q, K^T 矩阵的点积，再除以 sqrt(d_k) 得到注意力分数矩阵
        if mask is not None: # 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores.float(), dim = -1) # 将注意力权重矩阵乘以 V，得到最终的输出矩阵
        atten_output = torch.matmul(attn_weights, values)

        # Shape change: output[bsz,n_heads,seq_len,head_dim] -> output[bsz,seq_len, n_heads,head_dim] -> output[bsz,seq_len, n_heads * head_dim]
        output = atten_output.transpose(1,2).contiguous().view(bs, seq_len, -1)
        
        return output
    
def test_compute_theta():
    dim = 8
    theta = compute_theta(dim)
    assert theta.shape == (dim // 2,), f"Expected shape {(dim // 2,)}, got {theta.shape}"
    print("test_compute_theta passed.")

def test_precompute_freqs_cis():
    dim = 8
    seq_len = 10
    freqs_cis = precompute_freqs_cis(dim, seq_len)
    freqs_cis_llama = llama.precompute_freqs_cis(dim, seq_len)

    assert freqs_cis.shape == (seq_len, dim // 2), f"Expected shape {(seq_len, dim // 2)}, got {freqs_cis.shape}"
    print("test_precompute_freqs_cis passed.")

def test_apply_rotary_emb():
    batch_size = 2
    seq_len = 5
    n_heads = 2
    head_dim = 4  # dim // n_heads
    dim = n_heads * head_dim

    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    # git clone https://github.com/meta-llama/llama/tree/main and pip install -e .
    xq, xk = llama.apply_rotary_emb(xq, xk, freqs_cis)

    assert torch.allclose(xq_out, xq, atol=1e-4)
    assert xq_out.shape == xq.shape, f"Expected xq_out shape {xq.shape}, got {xq_out.shape}"
    assert xk_out.shape == xk.shape, f"Expected xk_out shape {xk.shape}, got {xk_out.shape}"

    print(f"test_apply_rotary_emb passed, xq_out and xq [0][0][0][0]: {xq_out[0,0,0,0]} {xq[0,0,0,0]}.")

def test_attention():
    batch_size = 2
    seq_len = 5
    dim = 8
    n_heads = 2
    x = torch.randn(batch_size, seq_len, dim)
    attn = Attention(dim=dim, max_seq_len=seq_len, n_heads=n_heads)
    output = attn(x)
    assert output.shape == (batch_size, seq_len, dim), f"Expected output shape {(batch_size, seq_len, dim)}, got {output.shape}"
    print("test_attention passed.")

if __name__ == "__main__":
    test_compute_theta()
    test_precompute_freqs_cis()
    test_apply_rotary_emb()
    test_attention()
