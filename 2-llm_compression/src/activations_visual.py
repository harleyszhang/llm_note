import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import LlamaForCausalLM, LlamaTokenizer

# 加载模型和分词器
def load_model_and_tokenizer(model_name, device):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 eos_token 为 pad_token
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    return model, tokenizer

# 运行推理并获取激活值和权重
@torch.no_grad()
def get_activations_and_weights(model, tokenizer, texts, layer_index = 4, channel_indexs = 200, device="cpu"):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    print("outputs.hidden_states shape is ", len(outputs.hidden_states))
    
    activation = outputs.hidden_states[layer_index].abs()[:, :, :]  # 最后一层激活值，选择前100个token和通道
    q_weight = model.model.layers[layer_index].self_attn.q_proj.weight.abs()[:, :]  # 第 layer_index 层的 q 映射层权重的前 200 个通道
    k_weight = model.model.layers[layer_index].self_attn.k_proj.weight.abs()[:, :]  # 第 layer_index 层的 q 映射层权重
    v_weight = model.model.layers[layer_index].self_attn.v_proj.weight.abs()[:, :]  # 第 layer_index 层的 q 映射层权重
    fcs = [q_weight,k_weight, v_weight]
    
    print(f"activation shape is {activation.shape} self_attn.q_proj.weight shape is {fcs[0].shape}")
    return activation, fcs

# 计算 SmoothQuant 的缩放因子
@torch.no_grad()
def calculate_scales(activation, fcs, alpha=0.5):
    original_shape = activation.shape
    act_reshaped = activation.view(-1, original_shape[-1]).abs().detach() # 将激活张量 shape 转换成 [batch_size * seq_len, hidden_size]
    act_max = torch.max(act_reshaped, dim=0)[0].float().cpu()  # 计算每个隐藏维度上的最大值并转移到 CPU
    
    # 如果 fcs 是线性层列表，则取整体的最大值，用来计算平滑因子 scales。
    weight_max_list = torch.cat([fc.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    w_max = weight_max_list.max(dim=0)[0].clamp(min=1e-5)

    print(f"act_max shape is {act_max.shape}, w_max shape is {w_max.shape}")
    scales = act_max.pow(alpha) / w_max.pow(1 - alpha)
    print(f"scales shape is {scales.shape}")
    return scales

# 应用 SmoothQuant 缩放因子到激活值和权重
@torch.no_grad()
def apply_smoothquant_scaling(activation, weights, scales):
    smooth_activation = activation / scales.view(1, 1, -1)
    q_proj_weight = weights[0]
    smooth_q_weight = q_proj_weight * scales.view(1, -1)
    print(f"smooth_activation_sample shape is {smooth_activation.shape} q_proj smooth_weight shape is {smooth_q_weight.shape}")

    return smooth_activation, smooth_q_weight

# 检测离群值并打印通道索引
def find_outlier_channels(activation_sample, threshold=10):
    mean = activation_sample.mean(dim=(0, 1))
    std = activation_sample.std(dim=(0, 1))
    z_scores = (activation_sample - mean) / std

    outliers = torch.where(z_scores > threshold)
    unique_channels = torch.unique(outliers[2])
    print(f"离群值所在的通道索引: {unique_channels.tolist()}")

# 3D 绘图函数
def plot_3d(data, title, xlabel, ylabel, zlabel, color, ax, y_max):
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x, y = x.flatten(), y.flatten()
    z = np.zeros_like(x)
    dx = dy = 1
    dz = data.flatten()
    ax.bar3d(x, y, z, dx, dy, dz, color=color, zsort='average')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(0, y_max)  # 设置统一的 y 轴范围

# 主函数，执行各个步骤
def main():
    model_name = "/Users/zhg/llm-awq/hf_weight/llama-2-7b/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # 处理输入文本并获取激活值和权重
    input_texts = [
        "The quick brown fox jumps over the lazy dog. " * 2,  # 通过重复句子生成超过64个词的文本
        "Artificial intelligence is revolutionizing the world. " * 2,
        "Large language models are powerful tools for NLP tasks. " * 2,
        "The meaning of life is to find " * 2
    ]
    activation_sample, weight_sample = get_activations_and_weights(model, tokenizer, input_texts, layer_index = 4,channel_indexs=200, device=device)

    # 检查离群值所在通道
    find_outlier_channels(activation_sample)

    # 计算 SmoothQuant 缩放因子并应用平滑转换
    scales = calculate_scales(activation_sample, weight_sample)

    smooth_activation_sample, smooth_weight_sample = apply_smoothquant_scaling(activation_sample, weight_sample, scales)

    # 确定所有图的统一 y 轴范围
    y_max = max(
        np.max(activation_sample.cpu().numpy()),
        np.max(smooth_activation_sample.cpu().numpy()),
        np.max(weight_sample[0].cpu().numpy()),
        np.max(smooth_weight_sample.cpu().numpy())
    )
    
    # 创建图表
    fig = plt.figure(figsize=(18, 8))
    batch_size, seq_len, hidden_size = activation_sample.shape
    activation_sample = activation_sample.view(-1, hidden_size)
    smooth_activation_sample = smooth_activation_sample.view(-1, hidden_size)
    
    # 绘制原始和平滑后的激活值和权重, weight_sample 是 q、k、v 映射层权重组合的列表
    plot_titles = [
        ("Activation (Original)\nHard to quantize", activation_sample, "brown"),
        ("Activation (SmoothQuant)\nEasy to quantize", smooth_activation_sample, "blue"),
        ("Weight (Original)\nVery easy to quantize", weight_sample[0], "blue"),
        ("Weight (SmoothQuant)\nHarder but still easy to quantize", smooth_weight_sample, "blue")
    ]
    
    for i, (title, data, color) in enumerate(plot_titles, start=1):
        ax = fig.add_subplot(1, 4, i, projection='3d')
        xlabel = "Channel" if "Activation" in title else "In Channel"
        ylabel = "Token" if "Activation" in title else "Out Channel"
        plot_3d(data.detach().cpu().numpy(), title, xlabel, ylabel, "Absolute Value", color, ax, y_max)
    
    # 添加主标题并保存图表
    fig.suptitle("SmoothQuant Visualization", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("llama2_7b_smoothquant_visualization2.png", format='png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

###################模型结构信息#######################3
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
"""