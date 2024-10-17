# -*- coding  : utf-8 -*-
# Author: honggao.zhang + chatgpt
# Create: 2023-03-27
# Version     : 0.1.0
# Description: transformer 模型的 PyTorch 实现

import torch.nn as nn
import torch
import time
import math

def timer(func):
    """decorator: print the cost time of run function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end_time - start_time:.6f} seconds to run')
        return result
    return wrapper

################## 1，nn.Linear 层理解 ##################
# 创建一个输入张量
input_tensor = torch.randn(10, 20)

# 创建一个 Linear 层，将输入维度从 20 变换到 30
linear_layer = nn.Linear(20, 30)

# 将输入张量传递给 Linear 层进行变换
start_time = time.time()
output_tensor = linear_layer(input_tensor)
end_time = time.time()

# 打印输出张量的形状
print(f"Output shape: {output_tensor.shape}")
print(f"Time taken: {end_time - start_time:.6f} seconds")

################## 2，词向量层算法 ##################
# 定义一个包含10个单词的词表，每个单词的嵌入维度为5
vocab_size = 10
embedding_dim = 5

# 定义一个词向量层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 假设输入句子为 "hello world"
input_sentence = ["hello", "world"]

# 将输入句子转化为词向量表示
input_embedding = embedding(torch.LongTensor([0, 1]))
# 最后，得到输入句子的词向量表示，尺寸为 (2, 5)。
print(input_embedding)
"""输入句子的词向量表示:
tensor([[ 0.1137, -0.4891, -0.2531, -0.6857, -0.2842],
        [-0.0712,  0.2119, -2.2433, -0.2108,  0.3078]],
       grad_fn=<EmbeddingBackward0>)
"""

################## word embedding ##################
# import torch
# import torch.nn as nn
# from torchtext.vocab import Vocab

# from collections import Counter

# text = "hello world hello" # 假设我们有一个文本
# counter = Counter(text.split()) # 统计每个单词的出现次数
# vocab = Vocab(counter) # 创建词汇表

# # 构建嵌入层
# vocab_size = 1000
# embedding_dim = 100
# embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# # 假设我们有一个输入单词列表，每个单词都是从词汇表中随机选择的
# input_words = ["hello", "world", "this", "is", "a", "test"]

# # 将单词转换为索引列表
# word_indexes = [vocab.index(word) for word in input_words]

# # 将索引列表转换为PyTorch张量
# word_indexes_tensor = torch.LongTensor(word_indexes)

# # 将索引列表输入嵌入层以获取嵌入向量
# word_embeddings = embedding_layer(word_indexes_tensor)

# # 输出嵌入向量
# print(word_embeddings)

################## 3，ScaleDotProductAttention 层实现 ##################
class ScaleDotProductAttention(nn.Module):
    """compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    
    def __init__(self, ):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-1, -2) # 计算矩阵 K 的转置  
        d_k = Q.size(-1)
        # 1, 计算 Q, K^T 矩阵的点积，再除以 sqrt(d_k) 得到注意力分数矩阵
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        # 2, 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 3, 对注意力分数矩阵按照最后一个维度进行 softmax 操作，得到注意力权重矩阵，值范围为 [0, 1]
        attn_weights = self.softmax(scores)
        # 4, 将注意力权重矩阵乘以 V，得到最终的输出矩阵
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# 创建 Q、K、V 三个张量
Q = torch.randn(5, 10, 64)  # (batch_size, sequence_length, d_k)
K = torch.randn(5, 10, 64)  # (batch_size, sequence_length, d_k)
V = torch.randn(5, 10, 64)  # (batch_size, sequence_length, d_k)

# 创建 ScaleDotProductAttention 层
attention = ScaleDotProductAttention()

# 将 Q、K、V 三个张量传递给 ScaleDotProductAttention 层进行计算
output, attn_weights = attention(Q, K, V)

# 打印输出矩阵和注意力权重矩阵的形状
print(f"ScaleDotProductAttention output shape: {output.shape}") # torch.Size([5, 10, 64])
print(f"attn_weights shape: {attn_weights.shape}") # torch.Size([5, 10, 10])

################## 4, MultiHeadAttention 层实现 ##################
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer
    Args:
        d_model: Dimensions of the input embedding vector, equal to input and output dimensions of each head
        n_head: number of heads, which is also the number of parallel attention layers
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)  # Q 线性变换层
        self.w_k = nn.Linear(d_model, d_model)  # K 线性变换层
        self.w_v = nn.Linear(d_model, d_model)  # V 线性变换层
        self.fc = nn.Linear(d_model, d_model)   # 输出线性变换层
        
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # size is [batch_size, seq_len, d_model]
        # 2, split by number of heads(n_head) # size is [batch_size, n_head, seq_len, d_model//n_head]
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3, compute attention
        sa_output, attn_weights = self.attention(q, k, v, mask)
        # 4, concat attention and linear transformation
        concat_tensor = self.concat(sa_output)
        mha_output = self.fc(concat_tensor)
        
        return mha_output
    
    def split(self, tensor):
        """
        split tensor by number of head(n_head)

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, n_head, seq_len, d_model//n_head], 输出矩阵是四维的，第二个维度是 head 维度
        
        # 将 Q、K、V 通过 reshape 函数拆分为 n_head 个头
        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, n_head, d_model // n_head)
        k = k.reshape(batch_size, seq_len, n_head, d_model // n_head)
        v = v.reshape(batch_size, seq_len, n_head, d_model // n_head)
        """
        
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        split_tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        
        return split_tensor
    
    def concat(self, sa_output):
        """ merge multiple heads back together

        Args:
            sa_output: [batch_size, n_head, seq_len, d_tensor]
            return: [batch_size, seq_len, d_model]
        """
        batch_size, n_head, seq_len, d_tensor = sa_output.size()
        d_model = n_head * d_tensor
        concat_tensor = sa_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return concat_tensor
    
################## 5, Layer Norm 层实现 ##################
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # 1，计算均值和标准差
        mean = x.mean(-1, keepdim=True) # '-1' means last dimension. 
        var = x.var(-1, keepdim=True)
        # 2，应用缩放和偏移系数
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        
        return out

# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)

# 1，Activate nn.LayerNorm module
layer_norm1 = nn.LayerNorm(embedding_dim)
pytorch_ln_out = layer_norm1(embedding)

# 2，Activate my nn.LayerNorm module
layer_norm2 = LayerNorm(embedding_dim)
my_ln_out = layer_norm2(embedding)

# 比较结果
print(torch.allclose(pytorch_ln_out, my_ln_out, rtol=0.1,atol=0.01))  # 输出 True
################## 6, Positionwise Feed Forward 层实现 ##################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

################## 7, TransformerEmbedding 层实现 ##################
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, max_len, d_model, drop_prob, device):
        """
        class for word embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

################## 8, EncoderLayer 层实现 ##################
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
    
    def forward(self, x, mask=None):
        x_residual1 = x
        
        # 1, compute multi-head attention
        x = self.mha(q=x, k=x, v=x, mask=mask)
        
        # 2, add residual connection and apply layer norm
        x = self.ln1( x_residual1 + self.dropout1(x) )
        x_residual2 = x
        
        # 3, compute position-wise feed forward
        x = self.ffn(x)
        
        # 4, add residual connection and apply layer norm
        x = self.ln2( x_residual2 + self.dropout2(x) )
        
        return x

################## 9, Encoder 实现 ##################
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, seq_len, d_model, ffn_hidden, n_head, n_layers, drop_prob=0.1, device='cpu'):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size = enc_voc_size,
                                        max_len = seq_len,
                                        d_model = d_model,
                                        drop_prob = drop_prob,
                                        device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) 
                                     for _ in range(n_layers)])
    
    def forward(self, x, mask=None):
        
        x = self.emb(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        return x

################## 10, DecoderLayer 层实现 ##################
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_head)
        self.ln1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.ln2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
    
    def forward(self, x, mask=None):
        x_residual1 = x
        
        # 1, compute multi-head attention
        x = self.mha(q=x, k=x, v=x, mask=mask)
        
        # 2, add residual connection and apply layer norm
        x = self.ln1( x_residual1 + self.dropout1(x) )
        x_residual2 = x
        
        # 3, compute position-wise feed forward
        x = self.ffn(x)
        
        # 4, add residual connection and apply layer norm
        x = self.ln2( x_residual2 + self.dropout2(x) )
        
        return x
    
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, n_head)
        self.ln1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.mha2 = MultiHeadAttention(d_model, n_head)
        self.ln2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.ln3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    def forward(self, dec_out, enc_out, trg_mask, src_mask):
        x_residual1 = dec_out
        
        # 1, compute multi-head attention
        x = self.mha1(q=dec_out, k=dec_out, v=dec_out, mask=trg_mask)
        
        # 2, add residual connection and apply layer norm
        x = self.ln1( x_residual1 + self.dropout1(x) )
        
        if enc_out is not None:
            # 3, compute encoder - decoder attention
            x_residual2 = x
            x = self.mha2(q=x, k=enc_out, v=enc_out, mask=src_mask)
    
            # 4, add residual connection and apply layer norm
            x = self.ln2( x_residual2 + self.dropout2(x) )
        
        # 5. positionwise feed forward network
        x_residual3 = x
        x = self.ffn(x)
        # 6, add residual connection and apply layer norm
        x = self.ln3( x_residual3 + self.dropout3(x) )
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output