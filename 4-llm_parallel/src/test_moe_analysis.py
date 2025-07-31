"""
DeepSeek V3 MoE 通信量分析测试版本
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_moe_communication():
    """分析MoE通信量"""
    print("🚀 DeepSeek V3 MoE 通信量分析")
    print("=" * 60)
    
    # 基础配置
    total_experts = 256
    experts_per_token = 8
    hidden_size = 7168
    dtype_size = 2  # BF16
    
    # Prefill阶段配置
    prefill_ep_size = 32
    prefill_nodes = 4
    prefill_gpus_per_node = 8
    
    # Decode阶段配置
    decode_ep_size = 144
    decode_nodes = 18
    decode_gpus_per_node = 8
    
    # 测试参数
    batch_size = 1024
    prefill_seq_len = 2048
    decode_seq_len = 1
    
    print(f"📋 基础配置:")
    print(f"   - 总专家数: {total_experts}")
    print(f"   - 每个token选择专家数: {experts_per_token}")
    print(f"   - 隐藏层维度: {hidden_size}")
    print(f"   - 批次大小: {batch_size}")
    
    # Prefill阶段分析
    print(f"\n📊 Prefill阶段分析:")
    print(f"   - 专家并行度: {prefill_ep_size}")
    print(f"   - 节点数: {prefill_nodes}")
    print(f"   - 每节点GPU数: {prefill_gpus_per_node}")
    
    prefill_total_tokens = batch_size * prefill_seq_len
    prefill_total_assignments = prefill_total_tokens * experts_per_token
    prefill_tokens_per_expert = prefill_total_assignments / total_experts
    prefill_experts_per_gpu = total_experts // prefill_ep_size
    prefill_tokens_per_gpu = prefill_tokens_per_expert * prefill_experts_per_gpu
    
    prefill_send_volume_gb = prefill_tokens_per_gpu * hidden_size * dtype_size / (1024**3)
    prefill_total_volume_gb = prefill_send_volume_gb * 2
    
    print(f"   - 总token数: {prefill_total_tokens:,}")
    print(f"   - 每专家token数: {prefill_tokens_per_expert:.0f}")
    print(f"   - 每GPU专家数: {prefill_experts_per_gpu}")
    print(f"   - 每GPU token数: {prefill_tokens_per_gpu:.0f}")
    print(f"   - 每GPU发送量: {prefill_send_volume_gb:.2f} GB")
    print(f"   - 总通信量: {prefill_total_volume_gb:.2f} GB")
    
    # Decode阶段分析
    print(f"\n📊 Decode阶段分析:")
    print(f"   - 专家并行度: {decode_ep_size}")
    print(f"   - 节点数: {decode_nodes}")
    print(f"   - 每节点GPU数: {decode_gpus_per_node}")
    
    decode_total_tokens = batch_size * decode_seq_len
    decode_total_assignments = decode_total_tokens * experts_per_token
    decode_tokens_per_expert = decode_total_assignments / total_experts
    decode_experts_per_gpu = total_experts // decode_ep_size
    decode_tokens_per_gpu = decode_tokens_per_expert * decode_experts_per_gpu
    
    decode_send_volume_gb = decode_tokens_per_gpu * hidden_size * dtype_size / (1024**3)
    decode_total_volume_gb = decode_send_volume_gb * 2
    
    print(f"   - 总token数: {decode_total_tokens:,}")
    print(f"   - 每专家token数: {decode_tokens_per_expert:.0f}")
    print(f"   - 每GPU专家数: {decode_experts_per_gpu}")
    print(f"   - 每GPU token数: {decode_tokens_per_gpu:.0f}")
    print(f"   - 每GPU发送量: {decode_send_volume_gb:.4f} GB")
    print(f"   - 总通信量: {decode_total_volume_gb:.4f} GB")
    
    # 对比分析
    print(f"\n📈 对比分析:")
    volume_ratio = prefill_total_volume_gb / decode_total_volume_gb
    print(f"   - 通信量比例: {volume_ratio:.0f}:1")
    print(f"   - Prefill通信量是Decode的 {volume_ratio:.0f} 倍")
    
    return {
        'prefill': {
            'ep_size': prefill_ep_size,
            'nodes': prefill_nodes,
            'experts_per_gpu': prefill_experts_per_gpu,
            'tokens_per_gpu': prefill_tokens_per_gpu,
            'send_volume_gb': prefill_send_volume_gb,
            'total_volume_gb': prefill_total_volume_gb
        },
        'decode': {
            'ep_size': decode_ep_size,
            'nodes': decode_nodes,
            'experts_per_gpu': decode_experts_per_gpu,
            'tokens_per_gpu': decode_tokens_per_gpu,
            'send_volume_gb': decode_send_volume_gb,
            'total_volume_gb': decode_total_volume_gb
        }
    }

def visualize_all2all_process():
    """可视化All2All通信过程"""
    print("\n🎨 生成All2All可视化图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Prefill阶段可视化
    ep_size = 32
    nodes = 4
    gpus_per_node = 8
    
    print(f"📊 绘制Prefill阶段 (EP{ep_size}, {nodes}节点)")
    
    for node in range(nodes):
        for gpu in range(gpus_per_node):
            x = node * 3
            y = gpu * 2
            
            # GPU框
            rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor='lightblue', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(x, y, f'{node}-{gpu}', ha='center', va='center', fontsize=8)
            
            # 专家分布 (每GPU 8个专家)
            experts_per_gpu = 256 // ep_size
            for j in range(experts_per_gpu):
                ex = x - 0.3 + j * 0.15
                ey = y + 0.6
                circle = patches.Circle((ex, ey), 0.05, 
                                      facecolor='orange', edgecolor='black')
                ax1.add_patch(circle)
                ax1.text(ex, ey, f'E{j}', ha='center', va='center', fontsize=6)
    
    ax1.set_xlim(-1, nodes * 3 - 1)
    ax1.set_ylim(-1, gpus_per_node * 2 - 1)
    ax1.set_title('Prefill阶段专家分布 (EP32, 4节点)', fontsize=14)
    ax1.set_xlabel('节点', fontsize=12)
    ax1.set_ylabel('GPU', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Decode阶段可视化
    ep_size = 144
    nodes = 18
    gpus_per_node = 8
    
    print(f"📊 绘制Decode阶段 (EP{ep_size}, {nodes}节点)")
    
    # 简化的GPU分布 (只显示前32个GPU)
    for i in range(min(ep_size, 32)):
        node = i // gpus_per_node
        gpu = i % gpus_per_node
        x = node * 2
        y = gpu * 1.5
        
        rect = patches.Rectangle((x-0.3, y-0.3), 0.6, 0.6, 
                               linewidth=2, edgecolor='black', 
                               facecolor='lightgreen', alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(x, y, f'{node}-{gpu}', ha='center', va='center', fontsize=6)
        
        # 专家分布 (每GPU 1-2个专家)
        experts_per_gpu = 256 // ep_size
        for j in range(experts_per_gpu):
            ex = x - 0.2 + j * 0.1
            ey = y + 0.4
            circle = patches.Circle((ex, ey), 0.03, 
                                  facecolor='red', edgecolor='black')
            ax2.add_patch(circle)
            ax2.text(ex, ey, f'E{j}', ha='center', va='center', fontsize=5)
    
    ax2.set_xlim(-1, min(nodes, 8) * 2 - 1)
    ax2.set_ylim(-1, gpus_per_node * 1.5 - 1)
    ax2.set_title('Decode阶段专家分布 (EP144, 18节点)', fontsize=14)
    ax2.set_xlabel('节点', fontsize=12)
    ax2.set_ylabel('GPU', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moe_all2all_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ 可视化图已保存为 'moe_all2all_visualization.png'")
    plt.show()

def analyze_communication_patterns():
    """分析通信模式"""
    print("\n🔍 通信模式分析:")
    print("=" * 60)
    
    # 带宽分析
    nvlink_bandwidth = 400.0  # GB/s
    ib_bandwidth = 50.0  # GB/s
    
    print(f"📡 带宽配置:")
    print(f"   - NVLink带宽: {nvlink_bandwidth} GB/s")
    print(f"   - InfiniBand带宽: {ib_bandwidth} GB/s")
    
    # Prefill阶段通信分析
    prefill_send_volume = 0.5  # 假设值，实际从上面计算
    prefill_intra_node_comm = prefill_send_volume * 7/8  # 节点内通信
    prefill_inter_node_comm = prefill_send_volume * 1/8  # 节点间通信
    
    prefill_intra_time = prefill_intra_node_comm / nvlink_bandwidth
    prefill_inter_time = prefill_inter_node_comm / ib_bandwidth
    prefill_total_time = max(prefill_intra_time, prefill_inter_time)
    
    print(f"\n📊 Prefill通信时间分析:")
    print(f"   - 节点内通信: {prefill_intra_node_comm:.3f} GB")
    print(f"   - 节点间通信: {prefill_inter_node_comm:.3f} GB")
    print(f"   - 节点内时间: {prefill_intra_time*1000:.2f} ms")
    print(f"   - 节点间时间: {prefill_inter_time*1000:.2f} ms")
    print(f"   - 总通信时间: {prefill_total_time*1000:.2f} ms")
    
    # Decode阶段通信分析
    decode_send_volume = 0.001  # 假设值
    decode_intra_node_comm = decode_send_volume * 7/8
    decode_inter_node_comm = decode_send_volume * 1/8
    
    decode_intra_time = decode_intra_node_comm / nvlink_bandwidth
    decode_inter_time = decode_inter_node_comm / ib_bandwidth
    decode_total_time = max(decode_intra_time, decode_inter_time)
    
    print(f"\n📊 Decode通信时间分析:")
    print(f"   - 节点内通信: {decode_intra_node_comm:.6f} GB")
    print(f"   - 节点间通信: {decode_inter_node_comm:.6f} GB")
    print(f"   - 节点内时间: {decode_intra_time*1000:.4f} ms")
    print(f"   - 节点间时间: {decode_inter_time*1000:.4f} ms")
    print(f"   - 总通信时间: {decode_total_time*1000:.4f} ms")

def main():
    """主函数"""
    print("🚀 DeepSeek V3 MoE 通信量分析与可视化")
    print("=" * 80)
    
    # 分析通信量
    results = analyze_moe_communication()
    
    # 分析通信模式
    analyze_communication_patterns()
    
    # 可视化
    visualize_all2all_process()
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main() 