"""
DeepSeek V3 MoE All2All 通信过程详细可视化

展示All2All通信的三个阶段：
1. 发送阶段：tokens分发到对应专家
2. 计算阶段：专家并行处理
3. 接收阶段：结果聚合回原位置
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class All2AllVisualizer:
    """All2All通信过程可视化器"""
    
    def __init__(self):
        self.fig = None
        self.axes = None
        
    def create_communication_diagram(self, phase="prefill"):
        """创建通信流程图"""
        if phase == "prefill":
            ep_size = 32
            nodes = 4
            gpus_per_node = 8
            experts_per_gpu = 8
        else:
            ep_size = 144
            nodes = 18
            gpus_per_node = 8
            experts_per_gpu = 1
            
        # 创建图形
        self.fig, self.axes = plt.subplots(2, 2, figsize=(20, 16))
        self.fig.suptitle(f'DeepSeek V3 MoE All2All 通信过程 - {phase.upper()}阶段', fontsize=16)
        
        # 1. 初始状态：token分布
        self._plot_initial_state(self.axes[0, 0], ep_size, nodes, gpus_per_node, experts_per_gpu)
        
        # 2. 发送阶段：All2All通信
        self._plot_send_phase(self.axes[0, 1], ep_size, nodes, gpus_per_node, experts_per_gpu)
        
        # 3. 计算阶段：专家处理
        self._plot_compute_phase(self.axes[1, 0], ep_size, nodes, gpus_per_node, experts_per_gpu)
        
        # 4. 接收阶段：结果聚合
        self._plot_receive_phase(self.axes[1, 1], ep_size, nodes, gpus_per_node, experts_per_gpu)
        
        plt.tight_layout()
        return self.fig
    
    def _plot_initial_state(self, ax, ep_size, nodes, gpus_per_node, experts_per_gpu):
        """绘制初始状态"""
        ax.set_title('1. 初始状态：Token分布', fontsize=14)
        
        # 创建GPU网格
        for node in range(min(nodes, 4)):  # 限制显示节点数
            for gpu in range(gpus_per_node):
                x = node * 3
                y = gpu * 2
                
                # GPU框
                rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor='lightblue', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x, y, f'{node}-{gpu}', ha='center', va='center', fontsize=8)
                
                # Token分布 (用小圆点表示)
                num_tokens = 100  # 简化显示
                for i in range(min(num_tokens, 20)):  # 只显示前20个token
                    tx = x - 0.3 + (i % 10) * 0.06
                    ty = y + 0.6 + (i // 10) * 0.1
                    circle = patches.Circle((tx, ty), 0.02, 
                                          facecolor='yellow', edgecolor='black', alpha=0.6)
                    ax.add_patch(circle)
        
        ax.set_xlim(-1, min(nodes, 4) * 3 - 1)
        ax.set_ylim(-1, gpus_per_node * 2 - 1)
        ax.set_xlabel('节点')
        ax.set_ylabel('GPU')
        ax.grid(True, alpha=0.3)
        
    def _plot_send_phase(self, ax, ep_size, nodes, gpus_per_node, experts_per_gpu):
        """绘制发送阶段"""
        ax.set_title('2. 发送阶段：All2All通信', fontsize=14)
        
        # 创建GPU网格
        for node in range(min(nodes, 4)):
            for gpu in range(gpus_per_node):
                x = node * 3
                y = gpu * 2
                
                # GPU框
                rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor='lightgreen', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x, y, f'{node}-{gpu}', ha='center', va='center', fontsize=8)
                
                # 专家分布
                for j in range(experts_per_gpu):
                    ex = x - 0.3 + j * 0.15
                    ey = y + 0.6
                    circle = patches.Circle((ex, ey), 0.05, 
                                          facecolor='orange', edgecolor='black')
                    ax.add_patch(circle)
                    ax.text(ex, ey, f'E{j}', ha='center', va='center', fontsize=6)
        
        # 绘制通信箭头
        for i in range(min(ep_size, 16)):  # 限制箭头数量
            for j in range(min(ep_size, 16)):
                if i != j:
                    node_i = i // gpus_per_node
                    gpu_i = i % gpus_per_node
                    node_j = j // gpus_per_node
                    gpu_j = j % gpus_per_node
                    
                    if node_i < 4 and node_j < 4:  # 只显示前4个节点
                        x1 = node_i * 3
                        y1 = gpu_i * 2
                        x2 = node_j * 3
                        y2 = gpu_j * 2
                        
                        # 节点内通信用蓝色，节点间通信用红色
                        if node_i == node_j:
                            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                      arrowprops=dict(arrowstyle='->', color='blue', alpha=0.3, lw=1))
                        else:
                            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                      arrowprops=dict(arrowstyle='->', color='red', alpha=0.3, lw=1))
        
        ax.set_xlim(-1, min(nodes, 4) * 3 - 1)
        ax.set_ylim(-1, gpus_per_node * 2 - 1)
        ax.set_xlabel('节点')
        ax.set_ylabel('GPU')
        ax.grid(True, alpha=0.3)
        
    def _plot_compute_phase(self, ax, ep_size, nodes, gpus_per_node, experts_per_gpu):
        """绘制计算阶段"""
        ax.set_title('3. 计算阶段：专家并行处理', fontsize=14)
        
        # 创建GPU网格
        for node in range(min(nodes, 4)):
            for gpu in range(gpus_per_node):
                x = node * 3
                y = gpu * 2
                
                # GPU框 (计算中状态)
                rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor='red', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x, y, f'{node}-{gpu}', ha='center', va='center', fontsize=8)
                
                # 专家计算状态
                for j in range(experts_per_gpu):
                    ex = x - 0.3 + j * 0.15
                    ey = y + 0.6
                    circle = patches.Circle((ex, ey), 0.05, 
                                          facecolor='purple', edgecolor='black')
                    ax.add_patch(circle)
                    ax.text(ex, ey, f'E{j}', ha='center', va='center', fontsize=6)
                    
                    # 添加计算指示器
                    ax.text(ex, ey + 0.15, '⚡', ha='center', va='center', fontsize=8)
        
        ax.set_xlim(-1, min(nodes, 4) * 3 - 1)
        ax.set_ylim(-1, gpus_per_node * 2 - 1)
        ax.set_xlabel('节点')
        ax.set_ylabel('GPU')
        ax.grid(True, alpha=0.3)
        
    def _plot_receive_phase(self, ax, ep_size, nodes, gpus_per_node, experts_per_gpu):
        """绘制接收阶段"""
        ax.set_title('4. 接收阶段：结果聚合', fontsize=14)
        
        # 创建GPU网格
        for node in range(min(nodes, 4)):
            for gpu in range(gpus_per_node):
                x = node * 3
                y = gpu * 2
                
                # GPU框 (完成状态)
                rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor='lightblue', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x, y, f'{node}-{gpu}', ha='center', va='center', fontsize=8)
                
                # 处理后的结果
                for i in range(5):  # 显示5个结果
                    rx = x - 0.2 + i * 0.1
                    ry = y + 0.6
                    circle = patches.Circle((rx, ry), 0.03, 
                                          facecolor='green', edgecolor='black', alpha=0.8)
                    ax.add_patch(circle)
        
        ax.set_xlim(-1, min(nodes, 4) * 3 - 1)
        ax.set_ylim(-1, gpus_per_node * 2 - 1)
        ax.set_xlabel('节点')
        ax.set_ylabel('GPU')
        ax.grid(True, alpha=0.3)

def create_communication_timeline():
    """创建通信时间线图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # 时间线数据
    phases = ['Token分发', 'All2All发送', '专家计算', 'All2All接收', '结果聚合']
    prefill_times = [0.1, 1.25, 5.0, 1.25, 0.1]  # ms
    decode_times = [0.001, 0.0025, 0.1, 0.0025, 0.001]  # ms
    
    x = np.arange(len(phases))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, prefill_times, width, label='Prefill阶段', 
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, decode_times, width, label='Decode阶段', 
                   color='lightcoral', alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('通信阶段')
    ax.set_ylabel('时间 (ms)')
    ax.set_title('DeepSeek V3 MoE 通信时间线')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moe_communication_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_bandwidth_analysis():
    """创建带宽分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 带宽利用率分析
    bandwidths = ['NVLink', 'InfiniBand']
    prefill_utilization = [87.5, 12.5]  # 百分比
    decode_utilization = [87.5, 12.5]   # 百分比
    
    x = np.arange(len(bandwidths))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, prefill_utilization, width, label='Prefill', 
                    color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, decode_utilization, width, label='Decode', 
                    color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('通信类型')
    ax1.set_ylabel('带宽利用率 (%)')
    ax1.set_title('通信带宽利用率分析')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bandwidths)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 通信量对比
    phases = ['Prefill', 'Decode']
    volumes = [14.0, 0.0009]  # GB
    
    bars = ax2.bar(phases, volumes, color=['skyblue', 'lightcoral'], alpha=0.8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.1,
                f'{height:.4f} GB', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('通信量 (GB)')
    ax2.set_title('Prefill vs Decode 通信量对比')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('moe_bandwidth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🎨 DeepSeek V3 MoE All2All 通信过程可视化")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = All2AllVisualizer()
    
    # 生成Prefill阶段可视化
    print("📊 生成Prefill阶段All2All可视化...")
    fig1 = visualizer.create_communication_diagram("prefill")
    plt.savefig('prefill_all2all_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 生成Decode阶段可视化
    print("📊 生成Decode阶段All2All可视化...")
    fig2 = visualizer.create_communication_diagram("decode")
    plt.savefig('decode_all2all_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建通信时间线
    print("📈 生成通信时间线...")
    create_communication_timeline()
    
    # 创建带宽分析
    print("📡 生成带宽分析...")
    create_bandwidth_analysis()
    
    print("✅ 所有可视化图生成完成!")

if __name__ == "__main__":
    main() 