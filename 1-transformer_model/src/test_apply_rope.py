import torch, math
import numpy as np

def rotate_half(x):
    """旋转输入的一半隐藏维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def test_apply_rotary_pos_emb():
    """全面测试 RoPE 应用函数并输出详细分析"""
    print("\n" + "="*90)
    print("🔥 旋转位置编码(RoPE)应用函数 - 全面测试")
    print("="*90)
    
    # 测试配置
    batch_size, seq_len, num_heads, head_dim = 2, 16, 32, 128
    total_params = batch_size * num_heads * seq_len * head_dim * 2
    
    print("\n📋 测试配置:")
    print(f"  批大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  注意力头数量: {num_heads}")
    print(f"  头维度: {head_dim}")
    print(f"  处理参数总量: {total_params:,} (≈{total_params/1e6:.1f}M)")
    print(f"  使用 RoPE 实现: {apply_rotary_pos_emb.__name__}")
    
    # 创建模拟数据
    torch.manual_seed(42)
    
    print("\n🔧 创建测试数据...")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print("\n📐 旋转前张量形状验证:")
    print(f"  q 形状: {q.shape} [批大小, 头数, 序列长度, 头维度]")
    print(f"  k 形状: {k.shape} [批大小, 头数, 序列长度, 头维度]")

    # 修复原始查询向量显示问题 - 确保使用正确的索引
    print("\n🔍 原始查询向量 (批次=0, 头=0, 位置=0):")
    print(f"  形状: {q[0, 0, 0].shape}")
    print(f"  前10个元素: {q[0, 0, 0][:10].numpy().round(4)}")
    print(f"  范数: {torch.norm(q[0, 0, 0]).item():.4f}")
    
    # 显示不同位置的向量以增强可观察性
    print("\n🔍 原始查询向量 (批次=0, 头=0, 位置=15):")
    print(f"  前10个元素: {q[0, 0, 15][:10].numpy().round(4)}")
    print(f"  范数: {torch.norm(q[0, 0, 15]).item():.4f}")
    
    # 创建位置编码
    print("\n⚙️ 生成位置编码...")
    cos = torch.zeros(batch_size, seq_len, head_dim)
    sin = torch.zeros(batch_size, seq_len, head_dim)
    
    print("\n  位置编码（旋转矩阵）形状验证:")
    print(f"  cos 形状: {cos.shape} [批大小, 序列长度, 头维度]")
    print(f"  sin 形状: {sin.shape} [批大小, 序列长度, 头维度]")

    for b in range(batch_size):
        for pos in range(seq_len):
            angle = pos * 0.1
            for d in range(head_dim):
                freq = 0.5 ** (d // 2)
                cos[b, pos, d] = math.cos(angle * freq)
                sin[b, pos, d] = math.sin(angle * freq)
    
    print("\n📊 位置编码示例:")
    print("  批次=0, 位置=0:")
    print(f"    cos[:10]: {cos[0, 0, :10].numpy().round(4)}")
    print(f"    sin[:10]: {sin[0, 0, :10].numpy().round(4)}")
    
    print("  批次=0, 位置=1:")
    print(f"    cos[:10]: {cos[0, 1, :10].numpy().round(4)}")
    print(f"    sin[:10]: {sin[0, 1, :10].numpy().round(4)}")
    
    # 应用旋转位置嵌入
    print("\n⚡ 应用旋转位置编码...")
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    print("  ✅ 旋转应用成功")
    
    # 增加 q_embed 和 k_embed 形状打印
    print("\n📐 旋转后张量形状验证:")
    print(f"  q_embed 形状: {q_rot.shape} [批大小, 头数, 序列长度, 头维度]")
    print(f"  k_embed 形状: {k_rot.shape} [批大小, 头数, 序列长度, 头维度]")

    # 验证输出形状
    print("\n✅ 输出验证:")
    print(f"  旋转后查询向量形状: {q_rot.shape} (应与输入相同)")
    print(f"  旋转后键向量形状: {k_rot.shape}")
    
    # 增强的旋转前后对比
    print("\n🔄 旋转前后对比 (批次=0, 头=0, 位置=0):")
    print("  原始向量:")
    print(f"    前10个元素: {q[0, 0, 0][:10].numpy().round(4)}")
    print(f"    范数: {torch.norm(q[0, 0, 0]).item():.4f}")
    
    print("  旋转后向量:")
    print(f"    前10个元素: {q_rot[0, 0, 0][:10].numpy().round(4)}")
    print(f"    范数: {torch.norm(q_rot[0, 0, 0]).item():.4f}")
    
    print("\n🔄 旋转前后对比 (批次=0, 头=0, 位置=15):")
    print("  原始向量:")
    print(f"    前10个元素: {q[0, 0, 15][:10].numpy().round(4)}")
    print(f"    范数: {torch.norm(q[0, 0, 15]).item():.4f}")
    
    print("  旋转后向量:")
    print(f"    前10个元素: {q_rot[0, 0, 15][:10].numpy().round(4)}")
    print(f"    范数: {torch.norm(q_rot[0, 0, 15]).item():.4f}")
    
    # 验证模长不变性
    print("\n🔍 范数保持验证:")
    max_diff = 0.0
    for b in range(batch_size):
        for h in range(num_heads):
            for pos in range(seq_len):
                orig_norm = torch.norm(q[b, h, pos]).item()
                rot_norm = torch.norm(q_rot[b, h, pos]).item()
                diff = abs(orig_norm - rot_norm)
                max_diff = max(max_diff, diff)
    
    print(f"  最大范数差异: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("  ✅ 通过: 所有向量范数保持稳定 (<1e-5)")
    else:
        print(f"  ⚠ 警告: 某些位置的范数差异超出容差范围 (最大差异 = {max_diff:.6f})")
    
    # 验证相对位置性质
    print("\n🌐 相对位置特性验证:")
    print("  计算位置0与其他位置的相似度...")
    print("  (RoPE中，相似度应随位置差增大而减小)")
    
    print("\n  位置对 | 相似度 | 与前一位置差异")
    print("  -------|--------|-----------------")
    
    prev_sim = None
    for pos_diff in [0, 1, 2, 4, 8]:
        if seq_len > pos_diff:
            dot_product = (q_rot[0, 0, 0] @ q_rot[0, 0, pos_diff]).item()
            
            diff_str = ""
            if prev_sim is not None:
                diff = prev_sim - dot_product
                diff_str = f"{diff:+.6f}" if diff > 0 else f"{diff:.6f}"
            
            print(f"  0 vs {pos_diff:2d} | {dot_product:.6f} | {diff_str}")
            prev_sim = dot_product

# 执行测试
if __name__ == "__main__":
    test_apply_rotary_pos_emb()