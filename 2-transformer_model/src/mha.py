
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 缩放因子：防止点乘结果过大
        self.scaling = self.head_dim ** -0.5
        
        # 对 query、key、value 进行线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        """
        query, key, value: Tensor, shape = (batch_size, seq_length, embed_dim)
        attn_mask: 可选，形状兼容 (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, _ = query.size()
        
        # 线性投影
        q = self.q_proj(query)  # (batch_size, seq_length, embed_dim)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头形式: (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算缩放点乘注意力得分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和得到注意力输出
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        # 拼接所有头： (batch_size, seq_length, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        # 经过输出投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights