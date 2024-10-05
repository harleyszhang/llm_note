import numpy as np
import torch.nn as nn
import torch, math

def normal_softmax(x):
    N, N = x.shape
    out = np.array(x)
    for r in range(0, N):
        maxi = 0
        for i in range(0, N):
            maxi = max(maxi, x[r, i])
        e_sum = 0
        for i in range(0, N):
            e_sum += np.exp(x[r, i] - maxi)
        for i in range(0, N):
            out[r, i] = np.exp(x[r, i] - maxi) / e_sum
    return out

def np_softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x

def np_selfattn(q, k, v):
    x = np.matmul(q, np.transpose(k))
    softmax = np_softmax(x)
    o = np.matmul(softmax, v)
    return o

def normal_selfattn(q, k, v):
    N, Dim = q.shape
    # [N, Dim] * [Dim, N] -> [N, N]
    x = np.zeros([N, N], "float32")
    for r in range(0, N):
        for i in range(0, N):
            for j in range(0, Dim):
                x[r, i] += q[r, j] * k[i, j]

    # [N, N]
    softmax = np_softmax(x)

    # [N, N] * [N, Dim] -> [N, Dim]
    o = np.zeros([N, Dim], "float32")
    for r in range(0, N):
        for i in range(0, Dim):
            for j in range(0, N):
                o[r, i] += softmax[r, j] * v[j, i]

    return o

def online_softmax_update(m0, d0, m1, d1):
    #                             x   1
    m = max(m0, m1) # flops: 1
    d = d0 * np.exp(m0 - m) + d1 * np.exp(m1-m) # flops: 5
    return m, d

def flashattn_0(Q, K, V):
    N, Dim = Q.shape
    
    # 1, Load Q K and write S. and Compute S[r][i] by matrix multiply 
    S = np.zeros([N, N], "float32")
    for r in range(0, N):
        for i in range(0, N):
            for j in range(0, Dim):
                S[r][i] += Q[r][j] * K[i][j] # K^T 的列就是 K 的行
    
    # 2, Load S and write O. Compute softmax[i] and O[r][c]
    O = np.zeros([N, Dim], "float32") 
    for r in range(0, N):
        m = S[r][0]
        d = 1
        for i in range(1, N):
            m, d = online_softmax_update(m, d, S[r][i], 1) # flops 为 6
        
        softmax = np.zeros([N], "float32")
        for i in range(0, N):
            softmax[i] = np.exp(S[r][i] - m) / d
        
        for c in range(0, Dim):
            for i in range(0, N):
                O[r][c] += softmax[i] * V[i][c] # V[i][c] 的加载不连续
    
    return O

def flashattn_update(m, d, m0, d0, o0, m1, d1, o1):
    #                      |   |   |   |   |   |
    #                      |   |   |   x   v   1
    # Init value:        MIN_M 0   0
    
    o = o0 * np.exp(m0 - m) * d0 / d + o1 * np.exp(m1 - m) * d1 / d
    return o

def flashattn_1(Q, K, V):
    N, Dim = Q.shape
    
    # 1, Load Q K and write S. and Compute S[r][i] by matrix multiply 
    S = np.zeros([N, N], "float32")
    O = np.zeros([N, Dim], "float32")
    m = np.zeros([N],  "float32")
    d = np.zeros([N],  "float32")
                
    for r in range(0, N):
        # 计算 QK^T 的第 i 行结果 S[r][i]
        for i in range(0, N):
            # QK^T
            for j in range(0, Dim):
                S[r][i] += Q[r][j] * K[i][j] # K^T 的列就是 K 的行
            
            # softmax: [N,N] -> [N,N]
            if i == 0:
                mm = S[r][0]
                dd = 0
            mm, dd = online_softmax_update(mm, dd, S[r][i], 1) # flops 为 6
            m[i] = mm
            d[i] = dd
        
        # PV: [N, N] * [N, Dim] -> [N, dim]
        for c in range(0, Dim):
            o = 0
            for i in range(0, N):
                # 迭代更新注意力计算输出
                o = flashattn_update(
                   m[i],
                   d[i],
                   m[i-1] if i > 0 else MIN_M,
                   d[i-1] if i > 0 else 0,
                   o,
                   S[r][i],
                   V[i][c],
                   1
                )
            O[r][c] = o
      
    return O

class SelfAttention(nn.Module):
    def __init__(self, ):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-1, -2) # 计算矩阵 K 的转置  
        d_k = Q.size(-1)
        # 1, 计算 Q, K^T 矩阵的点积.(再除以 sqrt(d_k) 得到注意力分数矩阵，暂时忽略)
        scores = torch.matmul(Q, K_T)
        # 2, 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 3, 对注意力分数矩阵按照最后一个维度进行 softmax 操作，得到注意力权重矩阵，值范围为 [0, 1]
        attn_weights = self.softmax(scores)
        # 4, 将注意力权重矩阵乘以 V，得到最终的输出矩阵
        output = torch.matmul(attn_weights, V)

        return output
    
if __name__ == "__main__":
       
    N = 256
    Dim = 64
    BLK = 4
    MIN_M = -10000

    Q = np.random.uniform(-1, 1, [N, Dim]).astype("float32")
    K = np.random.uniform(-1, 1, [N, Dim]).astype("float32")
    V = np.random.uniform(-1, 1, [N, Dim]).astype("float32")

    np_attn_res = np_selfattn(Q, K, V)
    normal_selfattn_res = normal_selfattn(Q, K, V)
    flashattn_0_res = flashattn_0(Q, K, V)
    flashattn_1_res = flashattn_1(Q, K, V)

    np.testing.assert_allclose(np_attn_res, normal_selfattn_res, 1e-6, 1e-6)
    np.testing.assert_allclose(np_attn_res, flashattn_0_res, 1e-6, 1e-6)
    np.testing.assert_allclose(np_attn_res, flashattn_1_res, 1e-6, 1e-6)
    
    attention = SelfAttention() # 创建 ScaleDotProductAttention 层
    pytorch_attention_output = attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V)) # 将 Q、K、V 三个张量传递给 ScaleDotProductAttention 层进行计算
   
    # 打印输出矩阵和注意力权重矩阵的形状
    print(f"standard attention output shape: {pytorch_attention_output.shape}") 
    print(f"flashattn_0 output shape: {flashattn_0_res.shape}")

    # print(f"attn_weights shape: {attn_weights.shape}") # torch.Size([5, 10, 10])

    if torch.allclose(pytorch_attention_output, torch.tensor(flashattn_1_res, dtype=torch.float), atol=1e-2):
        print("flashattention0 与 PyTorch 标准 attention 结果一致!")
    else:
        print("flashattention1 与 PyTorch 标准 attention结果不一致!")