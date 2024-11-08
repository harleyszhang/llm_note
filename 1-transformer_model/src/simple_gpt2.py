import torch
import torch.nn as nn 
from dataclasses import dataclass
from typing import List
from transformers import GPT2Tokenizer

@dataclass
class ModelConfig:
    # config reference: https://huggingface.co/openai-community/gpt2/blob/main/config.json
    num_layers: int = 12  # n_layer
    embedding_dim: int  = 768 # hidden_size, n_embd
    num_heads: int = 12   # n_head
    vocab_size: int = 50257 # vocab_size
    

class SimpleGPT2(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(SimpleGPT2, self).__init__()
        self.num_layers = model_config.num_layers
        self.embedding_dim = model_config.embedding_dim
        self.num_heads = model_config.num_heads
        self.vocab_size = model_config.vocab_size

        self.embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ) 
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        h = self.embed_layer(x) # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        # h = h.transpose(0, 1)  # 调整维度 [seq_len, batch_size, embedding_dim]

        for transformer_block in self.transformer_blocks:
            h = transformer_block(h)
        
        # h = h.transpose(0, 1)  # 转回 [batch_size, seq_len, embedding_dim]
        logits = self.lm_head(h)

        return logits

# 在 Python 的 typing 模块中，Union、Optional 和 List 用于类型注解，
# 帮助开发者明确变量、函数参数和返回值的类型，提高代码的可读性和可靠性。

def generate_text(
    model: SimpleGPT2,
    tokenizer: GPT2Tokenizer,
    texts: List[str], 
    max_gen_len: int = 50
):
    model.eval()
    # 一个包含编码后文本的张量，形状为 (batch_size, sequence_length)
    input_ids = tokenizer.encode(texts, return_tensors="pt")
    generated_ids = input_ids # shape: (1, 4)

    with torch.no_grad():
        for _ in range(max_gen_len):
            outputs = model(generated_ids) # outputs shape: (batch_size, max_gen_len, vocab_size)
            next_token_logits = outputs[:, -1, :] # (1, vocab_size) 默认 batch_size = 1
            # 沿着指定维度方向寻找并返回最大值的索引（需要哪个维度的最大值索引就指定哪个，跟 torch.max 维度的意义不一致）
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # (1, 1)
            
            # 没有使用 kv cache 优化
            generated_ids = torch.cat((generated_ids, next_token_id), dim = 1) # [batch_size, seq_len, hidden_size]
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # decode 的输入 token_ids 类型: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True) # (max_gen_len)
    
    return generated_text

def test_model_gen(input_text: List[str]):
    model_config = ModelConfig()
    model = SimpleGPT2(model_config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    output_text = generate_text(model, tokenizer, input_text, max_gen_len=8)
    print(output_text)


if __name__ == "__main__":
    test_model_gen("Once upon a time") 
    # 因为没有加载权重，所以输出 text 是随机输出，但形状是对的，10 个 token
    # 输出 Once upon a timeurdue Smartstocks hereditarySpanishlect flourish



"""
模型配置和结构信息
ModelConfig(num_layers=12, embedding_dim=768, num_heads=12, vocab_size=50257)
SimpleGPT2(
  (embed_layer): Embedding(50257, 768)
  (transformer_blocks): ModuleList(
    (0-11): 12 x TransformerEncoderLayer(
      (self_attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
      )
      (linear1): Linear(in_features=768, out_features=2048, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (linear2): Linear(in_features=2048, out_features=768, bias=True)
      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.1, inplace=False)
      (dropout2): Dropout(p=0.1, inplace=False)
    )
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=True)
)
"""