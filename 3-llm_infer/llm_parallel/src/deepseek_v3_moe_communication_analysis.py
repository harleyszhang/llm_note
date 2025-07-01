"""
DeepSeek V3 MoE 通信量分析与 All2All 可视化

基于DeepSeek V3/R1专家并行策略的详细通信分析：
- Prefill: 路由专家 EP32、MLA 和共享专家 DP32，4节点部署单元
- Decode: 路由专家 EP144、MLA 和共享专家 DP144，18节点部署单元

作者: 基于DeepSeek V3论文和实际部署策略
日期: 2024
"""

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class EPConfig:
    """专家并行配置"""
    # 基础配置
    total_experts: int = 256  # 总专家数
    experts_per_token: int = 8  # 每个token选择的专家数
    hidden_size: int = 7168  # 隐藏层维度
    moe_intermediate_size: int = 1407  # MoE中间层大小
    
    # Prefill阶段配置
    prefill_ep_size: int = 32  # 路由专家并行度
    prefill_dp_size: int = 32  # 共享专家数据并行度
    prefill_nodes: int = 4  # 节点数
    prefill_gpus_per_node: int = 8  # 每节点GPU数
    prefill_experts_per_gpu: int = 9  # 每GPU路由专家数
    prefill_shared_experts_per_gpu: int = 1  # 每GPU共享专家数
    
    # Decode阶段配置
    decode_ep_size: int = 144  # 路由专家并行度
    decode_dp_size: int = 144  # 共享专家数据并行度
    decode_nodes: int = 18  # 节点数
    decode_gpus_per_node: int = 8  # 每节点GPU数
    decode_experts_per_gpu: int = 2  # 每GPU路由专家数
    decode_shared_experts_per_gpu: int = 1  # 每GPU共享专家数

class MoECommunicationAnalyzer:
    """MoE通信量分析器"""
    
    def __init__(self, config: EPConfig):
        self.config = config
        self.nvlink_bandwidth = 400.0  # GB/s (H800)
        self.ib_bandwidth = 50.0  # GB/s (InfiniBand)
        
    def calculate_token_distribution(self, batch_size: int, seq_len: int, 
                                   experts_per_token: int = 8) -> Dict:
        """计算token到专家的分布"""
        total_tokens = batch_size * seq_len
        total_expert_assignments = total_tokens * experts_per_token
        
        # 理想负载均衡：每个专家获得相同数量的token
        tokens_per_expert = total_expert_assignments / self.config.total_experts
        
        return {
            'total_tokens': total_tokens,
            'total_expert_assignments': total_expert_assignments,
            'tokens_per_expert': tokens_per_expert,
            'load_balance_ratio': 1.0  # 理想情况
        }
    
    def analyze_prefill_communication(self, batch_size: int, seq_len: int) -> Dict:
        """分析Prefill阶段的通信量"""
        print("=" * 80)
        print("📊 Prefill阶段通信量分析")
        print("=" * 80)
        
        # 基础参数
        ep_size = self.config.prefill_ep_size
        nodes = self.config.prefill_nodes
        gpus_per_node = self.config.prefill_gpus_per_node
        experts_per_gpu = self.config.prefill_experts_per_gpu
        
        # Token分布
        token_dist = self.calculate_token_distribution(batch_size, seq_len)
        tokens_per_expert = token_dist['tokens_per_expert']
        
        # 每个GPU的专家分布
        experts_per_gpu = self.config.total_experts // ep_size
        tokens_per_gpu = tokens_per_expert * experts_per_gpu
        
        # 通信量计算
        hidden_size = self.config.hidden_size
        dtype_size = 2  # BF16 = 2 bytes
        
        # All2All通信量
        # 1. 发送tokens到对应专家
        send_volume = tokens_per_gpu * hidden_size * dtype_size  # bytes
        send_volume_gb = send_volume / (1024**3)
        
        # 2. 接收处理后的tokens
        receive_volume = send_volume
        receive_volume_gb = receive_volume / (1024**3)
        
        # 3. 总通信量
        total_volume_gb = send_volume_gb + receive_volume_gb
        
        # 通信模式分析
        # 节点内：NVLink (400 GB/s)
        # 节点间：InfiniBand (50 GB/s)
        intra_node_comm = send_volume_gb * (gpus_per_node - 1) / gpus_per_node
        inter_node_comm = send_volume_gb * (ep_size - gpus_per_node) / ep_size
        
        # 理论通信时间
        intra_node_time = intra_node_comm / self.nvlink_bandwidth
        inter_node_time = inter_node_comm / self.ib_bandwidth
        total_comm_time = max(intra_node_time, inter_node_time)
        
        result = {
            'phase': 'prefill',
            'ep_size': ep_size,
            'nodes': nodes,
            'gpus_per_node': gpus_per_node,
            'experts_per_gpu': experts_per_gpu,
            'tokens_per_gpu': tokens_per_gpu,
            'send_volume_gb': send_volume_gb,
            'receive_volume_gb': receive_volume_gb,
            'total_volume_gb': total_volume_gb,
            'intra_node_comm_gb': intra_node_comm,
            'inter_node_comm_gb': inter_node_comm,
            'intra_node_time_ms': intra_node_time * 1000,
            'inter_node_time_ms': inter_node_time * 1000,
            'total_comm_time_ms': total_comm_time * 1000,
            'token_distribution': token_dist
        }
        
        print(f"🔧 配置信息:")
        print(f"   - 专家并行度: {ep_size}")
        print(f"   - 节点数: {nodes}")
        print(f"   - 每节点GPU数: {gpus_per_node}")
        print(f"   - 每GPU专家数: {experts_per_gpu}")
        
        print(f"\n📈 通信量分析:")
        print(f"   - 每GPU发送量: {send_volume_gb:.2f} GB")
        print(f"   - 每GPU接收量: {receive_volume_gb:.2f} GB")
        print(f"   - 总通信量: {total_volume_gb:.2f} GB")
        print(f"   - 节点内通信: {intra_node_comm:.2f} GB")
        print(f"   - 节点间通信: {inter_node_comm:.2f} GB")
        
        print(f"\n⏱️ 通信时间分析:")
        print(f"   - 节点内通信时间: {intra_node_time*1000:.2f} ms")
        print(f"   - 节点间通信时间: {inter_node_time*1000:.2f} ms")
        print(f"   - 总通信时间: {total_comm_time*1000:.2f} ms")
        
        return result
    
    def analyze_decode_communication(self, batch_size: int, seq_len: int = 1) -> Dict:
        """分析Decode阶段的通信量"""
        print("\n" + "=" * 80)
        print("📊 Decode阶段通信量分析")
        print("=" * 80)
        
        # 基础参数
        ep_size = self.config.decode_ep_size
        nodes = self.config.decode_nodes
        gpus_per_node = self.config.decode_gpus_per_node
        experts_per_gpu = self.config.decode_experts_per_gpu
        
        # Token分布 (decode阶段seq_len=1)
        token_dist = self.calculate_token_distribution(batch_size, seq_len)
        tokens_per_expert = token_dist['tokens_per_expert']
        
        # 每个GPU的专家分布
        experts_per_gpu = self.config.total_experts // ep_size
        tokens_per_gpu = tokens_per_expert * experts_per_gpu
        
        # 通信量计算
        hidden_size = self.config.hidden_size
        dtype_size = 2  # BF16 = 2 bytes
        
        # All2All通信量
        send_volume = tokens_per_gpu * hidden_size * dtype_size
        send_volume_gb = send_volume / (1024**3)
        receive_volume_gb = send_volume_gb
        total_volume_gb = send_volume_gb + receive_volume_gb
        
        # 通信模式分析
        intra_node_comm = send_volume_gb * (gpus_per_node - 1) / gpus_per_node
        inter_node_comm = send_volume_gb * (ep_size - gpus_per_node) / ep_size
        
        # 理论通信时间
        intra_node_time = intra_node_comm / self.nvlink_bandwidth
        inter_node_time = inter_node_comm / self.ib_bandwidth
        total_comm_time = max(intra_node_time, inter_node_time)
        
        result = {
            'phase': 'decode',
            'ep_size': ep_size,
            'nodes': nodes,
            'gpus_per_node': gpus_per_node,
            'experts_per_gpu': experts_per_gpu,
            'tokens_per_gpu': tokens_per_gpu,
            'send_volume_gb': send_volume_gb,
            'receive_volume_gb': receive_volume_gb,
            'total_volume_gb': total_volume_gb,
            'intra_node_comm_gb': intra_node_comm,
            'inter_node_comm_gb': inter_node_comm,
            'intra_node_time_ms': intra_node_time * 1000,
            'inter_node_time_ms': inter_node_time * 1000,
            'total_comm_time_ms': total_comm_time * 1000,
            'token_distribution': token_dist
        }
        
        print(f"🔧 配置信息:")
        print(f"   - 专家并行度: {ep_size}")
        print(f"   - 节点数: {nodes}")
        print(f"   - 每节点GPU数: {gpus_per_node}")
        print(f"   - 每GPU专家数: {experts_per_gpu}")
        
        print(f"\n📈 通信量分析:")
        print(f"   - 每GPU发送量: {send_volume_gb:.4f} GB")
        print(f"   - 每GPU接收量: {receive_volume_gb:.4f} GB")
        print(f"   - 总通信量: {total_volume_gb:.4f} GB")
        print(f"   - 节点内通信: {intra_node_comm:.4f} GB")
        print(f"   - 节点间通信: {inter_node_comm:.4f} GB")
        
        print(f"\n⏱️ 通信时间分析:")
        print(f"   - 节点内通信时间: {intra_node_time*1000:.4f} ms")
        print(f"   - 节点间通信时间: {inter_node_time*1000:.4f} ms")
        print(f"   - 总通信时间: {total_comm_time*1000:.4f} ms")
        
        return result

class All2AllVisualizer:
    """All2All通信过程可视化器"""
    
    def __init__(self, config: EPConfig):
        self.config = config
        
    def visualize_prefill_all2all(self, batch_size: int, seq_len: int):
        """可视化Prefill阶段的All2All过程"""
        print("\n" + "=" * 80)
        print("🎨 Prefill阶段 All2All 可视化")
        print("=" * 80)
        
        ep_size = self.config.prefill_ep_size
        nodes = self.config.prefill_nodes
        gpus_per_node = self.config.prefill_gpus_per_node
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 1. 专家分布图
        self._plot_expert_distribution(ax1, ep_size, nodes, gpus_per_node, "Prefill")
        
        # 2. All2All通信流程图
        self._plot_all2all_communication(ax2, ep_size, nodes, gpus_per_node, "Prefill")
        
        plt.tight_layout()
        plt.savefig('prefill_all2all_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_decode_all2all(self, batch_size: int):
        """可视化Decode阶段的All2All过程"""
        print("\n" + "=" * 80)
        print("🎨 Decode阶段 All2All 可视化")
        print("=" * 80)
        
        ep_size = self.config.decode_ep_size
        nodes = self.config.decode_nodes
        gpus_per_node = self.config.decode_gpus_per_node
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 1. 专家分布图
        self._plot_expert_distribution(ax1, ep_size, nodes, gpus_per_node, "Decode")
        
        # 2. All2All通信流程图
        self._plot_all2all_communication(ax2, ep_size, nodes, gpus_per_node, "Decode")
        
        plt.tight_layout()
        plt.savefig('decode_all2all_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_expert_distribution(self, ax, ep_size: int, nodes: int, gpus_per_node: int, phase: str):
        """绘制专家分布图"""
        total_gpus = ep_size
        experts_per_gpu = self.config.total_experts // ep_size
        
        # 创建GPU网格
        gpu_positions = []
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                if node * gpus_per_node + gpu < total_gpus:
                    gpu_positions.append((node, gpu))
        
        # 绘制GPU和专家
        colors = plt.cm.Set3(np.linspace(0, 1, experts_per_gpu))
        
        for i, (node, gpu) in enumerate(gpu_positions):
            # GPU位置
            x = node * 3
            y = gpu * 2
            
            # 绘制GPU框
            rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, f'GPU\n{node}-{gpu}', ha='center', va='center', fontsize=8)
            
            # 绘制专家
            for j in range(experts_per_gpu):
                expert_x = x - 0.3 + j * 0.15
                expert_y = y + 0.6
                circle = patches.Circle((expert_x, expert_y), 0.05, 
                                      facecolor=colors[j], edgecolor='black', linewidth=1)
                ax.add_patch(circle)
                ax.text(expert_x, expert_y, f'E{i*experts_per_gpu+j}', 
                       ha='center', va='center', fontsize=6)
        
        ax.set_xlim(-1, nodes * 3 - 1)
        ax.set_ylim(-1, gpus_per_node * 2 - 1)
        ax.set_xlabel('节点')
        ax.set_ylabel('GPU')
        ax.set_title(f'{phase}阶段专家分布 (EP{ep_size}, {nodes}节点)')
        ax.grid(True, alpha=0.3)
        
    def _plot_all2all_communication(self, ax, ep_size: int, nodes: int, gpus_per_node: int, phase: str):
        """绘制All2All通信流程图"""
        total_gpus = ep_size
        
        # 创建GPU位置
        gpu_positions = []
        for node in range(nodes):
            for gpu in range(gpus_per_node):
                if node * gpus_per_node + gpu < total_gpus:
                    gpu_positions.append((node, gpu))
        
        # 绘制GPU
        for i, (node, gpu) in enumerate(gpu_positions):
            x = i * 2
            y = 0
            
            # GPU框
            rect = patches.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor='lightgreen', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, f'GPU{node}-{gpu}', ha='center', va='center', fontsize=8)
        
        # 绘制All2All通信
        for i in range(total_gpus):
            for j in range(total_gpus):
                if i != j:
                    x1 = i * 2
                    x2 = j * 2
                    y1 = 0.6
                    y2 = 0.6
                    
                    # 节点内通信用实线，节点间通信用虚线
                    node_i = i // gpus_per_node
                    node_j = j // gpus_per_node
                    
                    if node_i == node_j:
                        # 节点内通信 (NVLink)
                        ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=1)
                    else:
                        # 节点间通信 (InfiniBand)
                        ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.3, linewidth=1)
        
        ax.set_xlim(-1, total_gpus * 2 - 1)
        ax.set_ylim(-1, 2)
        ax.set_xlabel('GPU ID')
        ax.set_ylabel('通信层')
        ax.set_title(f'{phase}阶段 All2All 通信模式')
        
        # 添加图例
        ax.plot([], [], 'b-', label='节点内通信 (NVLink)', linewidth=2)
        ax.plot([], [], 'r--', label='节点间通信 (InfiniBand)', linewidth=2)
        ax.legend(loc='upper right')

def simulate_moe_communication():
    """模拟MoE通信过程"""
    print("🚀 DeepSeek V3 MoE 通信量分析与可视化")
    print("=" * 80)
    
    # 创建配置
    config = EPConfig()
    
    # 创建分析器
    analyzer = MoECommunicationAnalyzer(config)
    visualizer = All2AllVisualizer(config)
    
    # 分析Prefill阶段
    batch_size = 1024
    seq_len = 2048
    prefill_result = analyzer.analyze_prefill_communication(batch_size, seq_len)
    
    # 分析Decode阶段
    decode_result = analyzer.analyze_decode_communication(batch_size, seq_len=1)
    
    # 可视化
    visualizer.visualize_prefill_all2all(batch_size, seq_len)
    visualizer.visualize_decode_all2all(batch_size)
    
    # 对比分析
    print("\n" + "=" * 80)
    print("📊 Prefill vs Decode 对比分析")
    print("=" * 80)
    
    print(f"🔧 并行度对比:")
    print(f"   - Prefill EP: {prefill_result['ep_size']} vs Decode EP: {decode_result['ep_size']}")
    print(f"   - Prefill 节点: {prefill_result['nodes']} vs Decode 节点: {decode_result['nodes']}")
    print(f"   - Prefill 每GPU专家: {prefill_result['experts_per_gpu']} vs Decode 每GPU专家: {decode_result['experts_per_gpu']}")
    
    print(f"\n📈 通信量对比:")
    print(f"   - Prefill 总通信量: {prefill_result['total_volume_gb']:.2f} GB")
    print(f"   - Decode 总通信量: {decode_result['total_volume_gb']:.4f} GB")
    print(f"   - 通信量比例: {prefill_result['total_volume_gb']/decode_result['total_volume_gb']:.0f}:1")
    
    print(f"\n⏱️ 通信时间对比:")
    print(f"   - Prefill 通信时间: {prefill_result['total_comm_time_ms']:.2f} ms")
    print(f"   - Decode 通信时间: {decode_result['total_comm_time_ms']:.4f} ms")
    print(f"   - 时间比例: {prefill_result['total_comm_time_ms']/decode_result['total_comm_time_ms']:.0f}:1")
    
    return prefill_result, decode_result

if __name__ == "__main__":
    prefill_result, decode_result = simulate_moe_communication() 