import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

# 定义装饰器（如果没有动态更新需求，可以使用空装饰器）
def dynamic_rope_update(func):
    return func

# 定义RoPE初始化函数字典
ROPE_INIT_FUNCTIONS = {
    "default": lambda config, device: default_rope_init(config, device),
    # 添加其他初始化类型...
}

def default_rope_init(config, device) -> Tuple[torch.Tensor, float]:
    """默认RoPE初始化函数"""
    # 计算每个头的维度
    head_dim = config.hidden_size // config.num_attention_heads
    
    # 计算基础频率
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
    # 默认缩放因子为1.0
    attention_scaling = 1.0
    return inv_freq, attention_scaling

# 简化的配置类
class Qwen3Config:
    def __init__(self, rope_scaling=None, max_position_embeddings=4096, 
                 hidden_size=4096, num_attention_heads=32, rope_theta=10000.0):
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta

class Qwen3RotaryEmbedding(nn.Module):
    """
    用于 Qwen-3 系列模型的旋转位置编码（RoPE）张量构造器。
    该模块只负责计算 cos/sin 两张查表张量，供后续 q,k 张量做旋转。
    """
    def __init__(self, config: Qwen3Config, device: Optional[str] = None):
        super().__init__()
        
        # ---- ① 解析 RoPE 子类型 ----
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type",  # 新字段
                config.rope_scaling.get("type", "default")  # 旧字段兜底
            )
        else:
            self.rope_type = "default"  # 若没配，则走默认实现

        # ---- ② 记录最大序列长度（缓存大小）----
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        # ---- ③ 保存 config 并选择初始化函数 ----
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # ---- ④ 生成 inv_freq 与缩放因子 ----
        #   inv_freq shape: (head_dim//2,)，内容为 1/θ^i
        #   attention_scaling: 针对部分 RoPE 变体的额外缩放
        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)

        # 注册为 buffer → 保存到 state_dict，但不算模型参数
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq  # 再存一份备份，便于动态扩展
        head_dim = config.hidden_size // config.num_attention_heads
        # 打印初始化信息
        print(f"🔧 初始化 RoPE 编码器: 类型={self.rope_type}, 最大位置={self.max_seq_len_cached}")
        print(f"  头维度={head_dim}, 频率参数形状={inv_freq.shape}, 缩放因子={self.attention_scaling}")

    # --- 前向计算 ---
    @torch.no_grad()          # 不需要梯度
    @dynamic_rope_update      # 高阶装饰器：支持在线扩展 RoPE 长度
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        参数
        ----
        x:            (bs, seq, hidden_size) 只是用来拿 dtype/device
        position_ids: (bs, seq)              每个 token 的绝对位置
        
        返回
        ----
        cos, sin: 二张查表张量，shape=(bs, seq, head_dim)
        """
        # 0. 设备处理（确保在正确设备上）
        device = x.device
        
        # 1. 将 inv_freq 尺寸 [head_dim//2] 扩展到 (bs, head_dim//2, 1)
        inv_freq_expanded = (self.inv_freq[None, :, None]  # [1, head_dim//2, 1]
                            .float()                      # 确保fp32精度
                            .expand(position_ids.shape[0], -1, 1)  # [bs, head_dim//2, 1]
                            .to(device)) # 确保张量位于正确的计算设备上（CPU或GPU）
        print("【频率因子】inv_freq shape:", inv_freq_expanded.shape)
        print("【拓展后的频率因子】inv_freq_expanded shape:", inv_freq_expanded.shape)

        # 2. 将 position_ids (bs, seq) 扩展到 (bs, 1, seq)
        position_ids_expanded = position_ids[:, None, :].float()  # [bs, 1, seq]

        # 3. 指定 autocast 的设备类型（MPS 例外需退回 cpu）
        device_type = "cpu" if device.type == "mps" else device.type

        # 4. 强制禁用 autocast → 用 fp32 计算角度，防止精度损失
        with torch.autocast(device_type=device_type, enabled=False):
            # 矩阵乘法: [bs, head_dim//2, 1] @ (bs, 1, seq) → (bs, head_dim//2, seq)
            # 结果 freqs 张量包含了用于后续sin/cos计算的角度值。
            freqs = torch.matmul(inv_freq_expanded, position_ids_expanded)
            
            # 转置: (bs, head_dim//2, seq) → (bs, seq, head_dim//2)
            freqs = freqs.transpose(1, 2)
            
            # 拼接偶、奇维度 → (bs, seq, head_dim)
            emb = torch.cat((freqs, freqs), dim=-1)

            # 取 cos / sin（再乘可选 scaling）
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # 5. 与输入张量保持一致的 dtype 返回
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# ===== 增强版测试函数 =====
def test_rotary_embedding(visualize: bool = True):
    """全面测试 RoPE 实现并输出详细分析"""
    print("\n" + "="*60)
    print("🔥 Qwen-3 RoPE 旋转位置编码 全面测试")
    print("="*60)
    
    # 配置参数
    bs, seq, n_heads, head_dim = 2, 16, 32, 128
    hidden_size = n_heads * head_dim
    
    print(f"\n📋 测试配置:")
    print(f"  批大小 (bs) = {bs}")
    print(f"  序列长度 (seq) = {seq}")
    print(f"  注意力头数 (n_heads) = {n_heads}")
    print(f"  每个头的维度 (head_dim) = {head_dim}")
    print(f"  总隐藏大小 (hidden_size) = {hidden_size}")
    
    # 创建配置对象
    dummy_cfg = Qwen3Config(
        rope_scaling={"rope_type": "default"},
        max_position_embeddings=4096,
        hidden_size=hidden_size,
        num_attention_heads=n_heads,
        rope_theta=10000.0
    )

    # 创建RoPE模块
    print("\n🛠 创建 RoPE 编码器...")
    rot = Qwen3RotaryEmbedding(dummy_cfg, device="cpu")
    
    # 打印频率参数信息
    inv_freq = rot.inv_freq.cpu().numpy()
    print(f"\n📊 频率参数分析 (inv_freq):")
    print(f"  形状: {inv_freq.shape}")
    print(f"  最小值: {inv_freq.min():.6f}")
    print(f"  最大值: {inv_freq.max():.6f}")
    print(f"  平均值: {inv_freq.mean():.6f}")
    print(f"  前5个值: {inv_freq[:5].round(6)}")
    
    # 创建测试数据
    x = torch.randn(bs, seq, hidden_size)  # 模拟输入
    position_ids = torch.arange(seq).repeat(bs, 1)  # (bs, seq)
    
    print("\n⚡ 计算 RoPE 编码...")
    cos, sin = rot(x, position_ids)
    
    # 验证输出
    print("\n✅ 输出验证:")
    print(f"  cos 形状: {cos.shape} → (批大小, 序列长度, head_dim)")
    print(f"  sin 形状: {sin.shape}")
    
    # 验证三角函数性质
    cos_sin_sum = cos**2 + sin**2
    error = (cos_sin_sum - 1).abs().max()
    print(f"\n🔍 数学性质验证 (cos²θ + sin²θ = 1):")
    print(f"  最大误差: {error.item():.3e}")
    print(f"  是否接近1 (误差 < 1e-6): {'是' if error < 1e-6 else '否'}")

    # 分析位置差异
    print("\n🌐 位置编码差异分析:")
    for pos_diff in [0, 1, 4, 8]:
        # 计算位置差为pos_diff时的点积
        dot_products = []
        for i in range(0, seq - pos_diff):
            q = cos[0, i] * sin[0, i + pos_diff] - sin[0, i] * cos[0, i + pos_diff]
            dot_products.append(q.mean().item())
        
        avg_dot = np.mean(dot_products)
        print(f"  位置差 {pos_diff:2d}: 平均点积 = {avg_dot:.4f}")

    # 可视化部分
    if visualize:
        try:
            import matplotlib.pyplot as plt
            # 1. 频率参数可视化
            plt.figure(figsize=(12, 10))
            
            # 频率参数
            plt.subplot(2, 1, 1)
            plt.plot(inv_freq, 'o-', markersize=3)
            plt.title("RoPE Frequency Parameters (inv_freq)")
            plt.xlabel("Dimension Index")
            plt.ylabel("Frequency Value")
            plt.grid(True)
            
            # 2. 不同位置的角度变化
            plt.subplot(2, 1, 2)
            positions_to_plot = [0, 1, 4, 8]
            dims_to_plot = min(32, head_dim)
            
            # 取第一个batch的不同位置
            for pos in positions_to_plot:
                # 计算角度 (θ = position * inv_freq)
                angles = (position_ids[0, pos] * rot.inv_freq).cpu().numpy()
                plt.plot(angles[:dims_to_plot], label=f"Position {pos}")
            
            plt.title(f"Angle Values for First {dims_to_plot} Dimensions")
            plt.xlabel("Dimension Index")
            plt.ylabel("Angle (radians)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("rope_analysis.png")
            print("\n📈 可视化已保存至 rope_analysis.png")
            
            # 3. 位置差异热力图
            plt.figure(figsize=(10, 8))
            max_pos = 10  # 只显示前10个位置
            
            # 计算相对位置编码的差异
            position_diff = np.zeros((max_pos, max_pos))
            for i in range(max_pos):
                for j in range(max_pos):
                    # 计算点积作为相似度
                    sim = (cos[0, i] * cos[0, j] + sin[0, i] * sin[0, j]).mean().item()
                    position_diff[i, j] = sim
            
            plt.imshow(position_diff, cmap='viridis', origin='lower')
            plt.colorbar(label='Position Similarity')
            plt.title("Position Encoding Similarity Heatmap")
            plt.xlabel("Position j")
            plt.ylabel("Position i")
            plt.xticks(range(max_pos))
            plt.yticks(range(max_pos))
            plt.savefig("position_similarity.png")
            print("📊 位置相似度热力图已保存至 position_similarity.png")
            
        except ImportError:
            print("\n⚠ 无法导入 matplotlib，跳过可视化部分")
    
    print("\n🎉 测试完成!")

# 执行测试
if __name__ == "__main__":
    test_rotary_embedding(visualize=True)