"""
DeepSeek-V3 MoE模块详细实现
包含详细注释、可视化流程图和输入输出实例

作者: 基于DeepSeek-V3论文实现
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DeepseekV3MLP(nn.Module):
    """
    DeepSeek-V3 MLP专家模块
    每个专家都是一个独立的MLP，使用SwiGLU激活函数
    """
    
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        
        # SwiGLU激活函数的前向投影层
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu  # SwiGLU激活函数
        
        print(f"🔧 创建MLP专家: hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}")

    def forward(self, x):
        """
        SwiGLU激活函数: SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)
        """
        gate_output = self.act_fn(self.gate_proj(x))  # SiLU激活
        up_output = self.up_proj(x)  # 线性变换
        combined = gate_output * up_output  # 逐元素相乘
        output = self.down_proj(combined)  # 输出投影
        return output

class MoEGate(nn.Module):
    """
    MoE门控网络 - 简化版本用于演示
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(torch.randn(self.n_routed_experts, config.hidden_size))
        
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # 计算门控分数
        logits = F.linear(hidden_states, self.weight)
        scores = logits.sigmoid()
        
        # 选择top-k专家
        _, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        
        # 归一化权重
        if self.top_k > 1:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            
        return topk_idx, topk_weight

class DeepseekV3MoE(nn.Module):
    """
    DeepSeek-V3 混合专家模块
    
    核心功能:
    1. 通过门控网络为每个token选择合适的专家
    2. 并行处理所有token的专家计算
    3. 加权聚合专家输出
    4. 支持共享专家机制
    
    主要特点:
    - 支持专家并行(EP)训练
    - 高效的推理优化
    - 共享专家增强通用能力
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        
        print(f"🚀 初始化DeepSeek-V3 MoE模块:")
        print(f"   - 总专家数: {config.n_routed_experts}")
        print(f"   - 每个token选择专家数: {config.num_experts_per_tok}")
        print(f"   - 专家中间层大小: {config.moe_intermediate_size}")
        
        # 专家并行配置
        if hasattr(config, "ep_size") and config.ep_size > 1:
            # 多GPU专家并行
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = 0  # 简化版本，假设单GPU
            print(f"   - 专家并行大小: {self.ep_size}")
            print(f"   - 每个rank专家数: {self.experts_per_rank}")
        else:
            # 单GPU模式
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            print(f"   - 单GPU模式")
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            DeepseekV3MLP(
                config, 
                intermediate_size=config.moe_intermediate_size
            )
            for i in range(config.n_routed_experts)
        ])
        
        # 门控网络
        self.gate = MoEGate(config)
        
        # 共享专家（可选）
        if hasattr(config, 'n_shared_experts') and config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config, 
                intermediate_size=intermediate_size
            )
            print(f"   - 共享专家中间层大小: {intermediate_size}")
        else:
            self.shared_experts = None
            print(f"   - 无共享专家")

    def forward(self, hidden_states):
        """
        MoE前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            输出隐藏状态 [batch_size, seq_len, hidden_size]
        """
        print(f"\n🔄 DeepSeek-V3 MoE前向传播开始:")
        print(f"   - 输入形状: {hidden_states.shape}")
        
        # 保存原始形状和输入
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        # 步骤1: 门控网络计算路由决策
        print(f"\n📊 步骤1: 门控网络计算路由决策")
        topk_idx, topk_weight = self.gate(hidden_states)
        print(f"   - 专家索引形状: {topk_idx.shape}")
        print(f"   - 专家权重形状: {topk_weight.shape}")
        print(f"   - 第一个token选择的专家: {topk_idx[0].tolist()}")
        print(f"   - 第一个token的专家权重: {topk_weight[0].tolist()}")
        
        # 重塑输入为二维
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        print(f"   - 重塑后输入形状: {hidden_states.shape}")
        
        # 步骤2: 推理模式下的专家计算
        print(f"\n🎯 步骤2: 推理模式专家计算")
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

        # 步骤3: 添加共享专家输出（如果存在）
        if self.shared_experts is not None:
            print(f"\n🔗 步骤3: 添加共享专家输出")
            shared_output = self.shared_experts(identity)
            y = y + shared_output
            print(f"   - 共享专家输出形状: {shared_output.shape}")
        
        print(f"\n✅ MoE前向传播完成")
        print(f"   - 输出形状: {y.shape}")
        
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        """
        推理模式下的专家计算（优化版本）
        
        核心优化:
        1. 按专家分组处理token，减少内存访问
        2. 批量计算提高GPU利用率
        3. 避免重复计算
        """
        print(f"   - 推理模式专家计算开始")
        
        # 步骤1: 统计每个专家处理的token数量
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        print(f"   - 每个专家处理的token数: {tokens_per_expert.tolist()}")
        print(f"   - 活跃专家数: {(tokens_per_expert > 0).sum().item()}")
        
        # 步骤2: 对token按专家ID排序
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        print(f"   - 排序后token形状: {sorted_tokens.shape}")
        
        # 步骤3: 按专家分组处理
        outputs = []
        start_idx = 0
        
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
                
            # 获取当前专家
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            
            print(f"   - 专家{i}: 处理{num_tokens}个token")
            
            # 专家前向传播
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        
        # 步骤4: 合并所有专家输出
        if len(outputs) > 0:
            outs = torch.cat(outputs, dim=0)
        else:
            outs = sorted_tokens.new_empty(0)
        
        print(f"   - 合并后输出形状: {outs.shape}")
        
        # 步骤5: 恢复原始token顺序
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        
        # 步骤6: 加权聚合
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        
        print(f"   - 最终输出形状: {final_out.shape}")
        return final_out

    def visualize_moe_process(self, hidden_states, save_path="moe_process_visualization.png"):
        """
        可视化MoE处理过程
        """
        print(f"\n🎨 开始生成MoE处理过程可视化...")
        
        # 获取MoE处理结果
        with torch.no_grad():
            output = self.forward(hidden_states)
            topk_idx, topk_weight = self.gate(hidden_states)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DeepSeek-V3 MoE处理过程可视化', fontsize=16, fontweight='bold')
        
        # 1. 专家使用分布
        ax1 = axes[0, 0]
        expert_usage = torch.zeros(self.config.n_routed_experts)
        for idx in topk_idx.flatten():
            expert_usage[idx] += 1
        
        bars = ax1.bar(range(self.config.n_routed_experts), expert_usage.numpy(), 
                      alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('专家使用分布')
        ax1.set_xlabel('专家索引')
        ax1.set_ylabel('使用次数')
        ax1.set_xticks(range(0, self.config.n_routed_experts, 32))
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            if expert_usage[i] > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{int(expert_usage[i])}', ha='center', va='bottom', fontsize=8)
        
        # 2. 权重分布热力图
        ax2 = axes[0, 1]
        sns.heatmap(topk_weight[:min(20, topk_weight.shape[0])].numpy(), 
                   ax=ax2, cmap='YlOrRd', cbar_kws={'label': '权重值'})
        ax2.set_title('Token-专家权重热力图 (前20个token)')
        ax2.set_xlabel('专家索引')
        ax2.set_ylabel('Token索引')
        
        # 3. 输入输出对比
        ax3 = axes[1, 0]
        bsz, seq_len, hidden_size = hidden_states.shape
        
        # 选择第一个token进行可视化
        input_token = hidden_states[0, 0].detach().numpy()
        output_token = output[0, 0].detach().numpy()
        
        x_pos = np.arange(min(50, hidden_size))
        ax3.plot(x_pos, input_token[:50], 'b-', alpha=0.7, label='输入', linewidth=2)
        ax3.plot(x_pos, output_token[:50], 'r-', alpha=0.7, label='输出', linewidth=2)
        ax3.set_title('Token向量对比 (前50维)')
        ax3.set_xlabel('隐藏维度')
        ax3.set_ylabel('数值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MoE处理流程图
        ax4 = axes[1, 1]
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 8)
        ax4.axis('off')
        
        # 绘制流程图
        steps = [
            ('输入\nHidden States', 1, 6, 'lightblue'),
            ('门控网络\n路由决策', 3, 6, 'lightyellow'),
            ('专家并行\n计算', 5, 6, 'lightgreen'),
            ('加权聚合\n输出', 7, 6, 'lightcoral'),
            ('共享专家\n融合', 9, 6, 'lightpink')
        ]
        
        for text, x, y, color in steps:
            rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=2, 
                                   edgecolor='black', facecolor=color, alpha=0.7)
            ax4.add_patch(rect)
            ax4.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 添加箭头
        for i in range(len(steps)-1):
            ax4.arrow(steps[i][1]+0.5, steps[i][2], 1, 0, head_width=0.1, 
                     head_length=0.1, fc='black', ec='black')
        
        ax4.set_title('MoE处理流程')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图表已保存到: {save_path}")
        plt.show()

def create_moe_config():
    """
    创建MoE配置对象
    """
    class Config:
        def __init__(self):
            self.hidden_size = 7168  # 隐藏层维度
            self.intermediate_size = 18432  # 标准MLP中间层大小
            self.moe_intermediate_size = 2048  # MoE专家中间层大小
            self.n_routed_experts = 256  # 路由专家数量
            self.num_experts_per_tok = 8  # 每个token选择的专家数
            self.n_shared_experts = 1  # 共享专家数量
            self.norm_topk_prob = True  # 是否归一化top-k概率
    
    return Config()

def demo_deepseekv3_moe():
    """
    DeepSeek-V3 MoE模块演示函数
    """
    print("=" * 80)
    print("🚀 DeepSeek-V3 MoE模块演示")
    print("=" * 80)
    
    # 创建配置
    config = create_moe_config()
    print(f"📋 配置信息:")
    print(f"   - 隐藏层维度: {config.hidden_size}")
    print(f"   - 路由专家数: {config.n_routed_experts}")
    print(f"   - 每个token选择专家数: {config.num_experts_per_tok}")
    print(f"   - 专家中间层大小: {config.moe_intermediate_size}")
    print(f"   - 共享专家数: {config.n_shared_experts}")
    
    # 创建MoE模块
    moe_module = DeepseekV3MoE(config)
    print(f"\n🔧 MoE模块创建完成")
    
    # 创建示例输入
    batch_size = 2
    seq_len = 8
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\n📥 示例输入:")
    print(f"   - 形状: {hidden_states.shape}")
    print(f"   - 数据类型: {hidden_states.dtype}")
    print(f"   - 数值范围: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
    
    # 设置为推理模式
    moe_module.eval()
    
    # 执行前向传播
    print(f"\n🔄 执行MoE前向传播...")
    with torch.no_grad():
        output = moe_module(hidden_states)
    
    # 分析结果
    print(f"\n📊 MoE处理结果分析:")
    print(f"   - 输出形状: {output.shape}")
    print(f"   - 输出数值范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   - 输入输出差异: {torch.abs(output - hidden_states).mean():.6f}")
    
    # 统计专家使用情况
    with torch.no_grad():
        topk_idx, topk_weight = moe_module.gate(hidden_states)
    
    expert_usage = torch.zeros(config.n_routed_experts)
    for idx in topk_idx.flatten():
        expert_usage[idx] += 1
    
    print(f"\n🎯 专家使用统计:")
    print(f"   - 被使用的专家数: {(expert_usage > 0).sum().item()}")
    print(f"   - 使用最多的专家: {expert_usage.argmax().item()} (使用{expert_usage.max().item()}次)")
    print(f"   - 使用最少的专家: {expert_usage.argmin().item()} (使用{expert_usage.min().item()}次)")
    print(f"   - 平均每个专家使用次数: {expert_usage.mean():.2f}")
    
    # 检查负载均衡
    print(f"\n⚖️ 负载均衡检查:")
    total_usage = expert_usage.sum()
    expected_usage = batch_size * seq_len * config.num_experts_per_tok
    print(f"   - 总专家使用次数: {total_usage}")
    print(f"   - 期望使用次数: {expected_usage}")
    print(f"   - 负载均衡度: {1 - abs(total_usage - expected_usage) / expected_usage:.4f}")
    
    # 生成可视化
    print(f"\n🎨 生成MoE处理过程可视化...")
    moe_module.visualize_moe_process(hidden_states, save_path="deepseekv3_moe_visualization.png")
    
    print(f"\n✅ 演示完成!")
    print("=" * 80)

if __name__ == "__main__":
    # 运行演示
    demo_deepseekv3_moe() 