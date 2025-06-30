"""
DeepSeek-V3 MoE门控机制详细实现
包含详细注释、可视化流程图和输入输出实例

作者: 基于DeepSeek-V3论文实现
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Tuple, Optional
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MoEGate(nn.Module):
    """
    DeepSeek-V3 MoE门控机制实现
    
    核心功能:
    1. 计算每个token对每个专家的亲和度分数
    2. 基于节点受限路由策略选择top-k专家
    3. 实现无辅助损失的负载均衡
    
    主要创新:
    - 无辅助损失负载均衡策略
    - 节点受限路由(每个token最多路由到4个节点)
    - 无token丢弃机制
    """
    
    def __init__(self, config):
        """
        初始化MoE门控模块
        
        Args:
            config: 配置对象，包含以下关键参数:
                - num_experts_per_tok: 每个token路由到的专家数量 (默认8)
                - n_routed_experts: 总专家数量 (默认64)
                - routed_scaling_factor: 路由缩放因子 (默认1.0)
                - scoring_func: 评分函数类型 ("sigmoid")
                - topk_method: top-k选择方法 ("noaux_tc")
                - n_group: 专家分组数量 (默认16)
                - topk_group: 每个token最多路由到的节点数 (默认4)
                - norm_topk_prob: 是否归一化top-k概率
                - hidden_size: 隐藏层维度
        """
        super().__init__()
        self.config = config
        
        # 核心参数
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数: 8
        self.n_routed_experts = config.n_routed_experts  # 总专家数: 64
        self.routed_scaling_factor = config.routed_scaling_factor  # 路由缩放因子
        self.scoring_func = config.scoring_func  # 评分函数: "sigmoid"
        self.topk_method = config.topk_method  # top-k方法: "noaux_tc"
        self.n_group = config.n_group  # 专家分组数: 16
        self.topk_group = config.topk_group  # 最多路由节点数: 4
        
        # 算法参数
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化概率
        self.gating_dim = config.hidden_size  # 门控维度
        
        # 门控权重矩阵: [n_routed_experts, hidden_size]
        # 用于计算每个token对每个专家的亲和度分数
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        
        # 专家分数修正偏置 (仅用于noaux_tc方法)
        # 用于实现无辅助损失的负载均衡
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        
        # 初始化参数
        self.reset_parameters()
        
        print(f"🔧 MoE门控初始化完成:")
        print(f"   - 总专家数: {self.n_routed_experts}")
        print(f"   - 每个token选择专家数: {self.top_k}")
        print(f"   - 专家分组数: {self.n_group}")
        print(f"   - 最多路由节点数: {self.topk_group}")
        print(f"   - 门控维度: {self.gating_dim}")

    def reset_parameters(self) -> None:
        """
        初始化模型参数
        
        使用Kaiming均匀初始化门控权重，确保训练稳定性
        """
        import torch.nn.init as init
        
        # Kaiming均匀初始化门控权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 初始化专家分数修正偏置
        if self.topk_method == "noaux_tc":
            nn.init.zeros_(self.e_score_correction_bias)
        
        print("✅ 参数初始化完成")

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: 计算路由决策
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            topk_idx: 选择的专家索引 [batch_size * seq_len, top_k]
            topk_weight: 对应的权重 [batch_size * seq_len, top_k]
        """
        bsz, seq_len, h = hidden_states.shape
        
        print(f"\n🚀 MoE门控前向传播开始:")
        print(f"   - 输入形状: {hidden_states.shape}")
        print(f"   - 批次大小: {bsz}, 序列长度: {seq_len}, 隐藏维度: {h}")
        
        # ==================== 步骤1: 计算门控分数 ====================
        print(f"\n📊 步骤1: 计算门控分数")
        
        # 重塑输入: [batch_size, seq_len, hidden_size] -> [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)
        print(f"   - 重塑后形状: {hidden_states.shape}")
        
        # 计算logits: [batch_size * seq_len, n_routed_experts]
        # 使用线性变换计算每个token对每个专家的亲和度
        logits = F.linear(
            hidden_states.type(torch.float32), 
            self.weight.type(torch.float32), 
            None
        )
        print(f"   - Logits形状: {logits.shape}")
        print(f"   - Logits统计: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        
        # 应用激活函数得到最终分数
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"不支持的评分函数: {self.scoring_func}"
            )
        print(f"   - Sigmoid分数统计: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        # ==================== 步骤2: 选择top-k专家 ====================
        print(f"\n🎯 步骤2: 选择top-k专家 (方法: {self.topk_method})")
        
        if self.topk_method == "noaux_tc":
            # 确保在推理模式下运行
            assert not self.training, "noaux_tc方法仅支持推理模式"
            
            # 应用专家分数修正偏置 (负载均衡)
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            print(f"   - 修正后分数形状: {scores_for_choice.shape}")
            print(f"   - 修正偏置统计: min={self.e_score_correction_bias.min():.4f}, max={self.e_score_correction_bias.max():.4f}")
            
            # 计算每个节点的组分数
            # 将专家按节点分组，计算每个节点上最高K_r/M个专家的分数之和
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1)
                .topk(2, dim=-1)[0]  # 选择每个组内前2个最高分数
                .sum(dim=-1)  # 求和得到组分数
            )  # [batch_size * seq_len, n_group]
            print(f"   - 组分数形状: {group_scores.shape}")
            print(f"   - 组分数统计: min={group_scores.min():.4f}, max={group_scores.max():.4f}")
            
            # 选择top-k_group个节点
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]  # [batch_size * seq_len, topk_group]
            print(f"   - 选择的节点索引形状: {group_idx.shape}")
            print(f"   - 节点索引范围: {group_idx.min()} - {group_idx.max()}")
            
            # 创建节点掩码
            group_mask = torch.zeros_like(group_scores)  # [batch_size * seq_len, n_group]
            group_mask.scatter_(1, group_idx, 1)  # 将选中的节点标记为1
            print(f"   - 节点掩码形状: {group_mask.shape}")
            print(f"   - 选中节点数: {group_mask.sum().item()}")
            
            # 创建专家掩码
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [batch_size * seq_len, n_routed_experts]
            print(f"   - 专家掩码形状: {score_mask.shape}")
            print(f"   - 可用专家数: {score_mask.sum().item()}")
            
            # 应用掩码，将未选中节点的专家分数设为负无穷
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
            print(f"   - 掩码后分数统计: min={tmp_scores.min():.4f}, max={tmp_scores.max():.4f}")
            
            # 选择top-k个专家
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
            print(f"   - 选择的专家索引形状: {topk_idx.shape}")
            print(f"   - 专家权重形状: {topk_weight.shape}")
            
        else:
            raise NotImplementedError(
                f"不支持的TopK方法: {self.topk_method}"
            )
        
        # ==================== 步骤3: 权重归一化 ====================
        print(f"\n⚖️ 步骤3: 权重归一化")
        
        # 如果选择多个专家且需要归一化，则进行概率归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            print(f"   - 归一化后权重统计: min={topk_weight.min():.4f}, max={topk_weight.max():.4f}")
        
        # 应用路由缩放因子
        topk_weight = topk_weight * self.routed_scaling_factor
        print(f"   - 缩放后权重统计: min={topk_weight.min():.4f}, max={topk_weight.max():.4f}")
        
        print(f"\n✅ MoE门控前向传播完成")
        print(f"   - 输出专家索引形状: {topk_idx.shape}")
        print(f"   - 输出权重形状: {topk_weight.shape}")
        
        return topk_idx, topk_weight

    def visualize_routing_process(self, hidden_states: torch.Tensor, save_path: str = "moe_routing.png"):
        """
        可视化MoE路由过程
        
        Args:
            hidden_states: 输入隐藏状态
            save_path: 保存路径
        """
        print(f"\n🎨 开始生成路由过程可视化...")
        
        # 获取路由结果
        with torch.no_grad():
            topk_idx, topk_weight = self.forward(hidden_states)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DeepSeek-V3 MoE路由过程可视化', fontsize=16, fontweight='bold')
        
        # 1. 专家分布热力图
        ax1 = axes[0, 0]
        expert_usage = torch.zeros(self.n_routed_experts)
        for idx in topk_idx.flatten():
            expert_usage[idx] += 1
        
        sns.heatmap(
            expert_usage.view(self.n_group, -1).numpy(),
            ax=ax1,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': '使用次数'}
        )
        ax1.set_title('专家使用分布热力图')
        ax1.set_xlabel('组内专家索引')
        ax1.set_ylabel('节点组索引')
        
        # 2. 权重分布直方图
        ax2 = axes[0, 1]
        ax2.hist(topk_weight.flatten().numpy(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('专家权重分布')
        ax2.set_xlabel('权重值')
        ax2.set_ylabel('频次')
        ax2.grid(True, alpha=0.3)
        
        # 3. 节点选择统计
        ax3 = axes[1, 0]
        bsz, seq_len, _ = hidden_states.shape
        n_tokens = bsz * seq_len
        
        # 计算每个token选择的节点分布
        with torch.no_grad():
            scores = F.linear(
                hidden_states.view(-1, self.gating_dim).type(torch.float32),
                self.weight.type(torch.float32),
                None
            ).sigmoid()
            
            scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(n_tokens, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            _, group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)
        
        # 统计每个节点被选择的次数
        node_usage = torch.zeros(self.n_group)
        for idx in group_idx.flatten():
            node_usage[idx] += 1
        
        bars = ax3.bar(range(self.n_group), node_usage.numpy(), color='lightgreen', alpha=0.7)
        ax3.set_title('节点选择统计')
        ax3.set_xlabel('节点索引')
        ax3.set_ylabel('被选择次数')
        ax3.set_xticks(range(self.n_group))
        
        # 添加数值标签
        for bar, value in zip(bars, node_usage.numpy()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(value)}', ha='center', va='bottom')
        
        # 4. 路由流程图
        ax4 = axes[1, 1]
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 8)
        ax4.axis('off')
        
        # 绘制流程图
        steps = [
            ('输入\nHidden States', 1, 6, 'lightblue'),
            ('计算\n门控分数', 3, 6, 'lightyellow'),
            ('节点受限\n路由', 5, 6, 'lightgreen'),
            ('专家选择\nTop-K', 7, 6, 'lightcoral'),
            ('权重归一化', 9, 6, 'lightpink')
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
        
        ax4.set_title('MoE路由流程')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图表已保存到: {save_path}")
        plt.show()

def create_moe_config():
    """
    创建MoE配置对象
    
    Returns:
        配置对象
    """
    class Config:
        def __init__(self):
            self.num_experts_per_tok = 8  # 每个token选择的专家数
            self.n_routed_experts = 256    # 总专家数
            self.routed_scaling_factor = 1.0  # 路由缩放因子
            self.scoring_func = "sigmoid"  # 评分函数
            self.topk_method = "noaux_tc"  # top-k方法
            self.n_group = 8  # 专家分组数 (节点数)
            self.topk_group = 4  # 每个token最多路由到的节点数
            self.norm_topk_prob = True  # 是否归一化概率
            self.hidden_size = 7168  # 隐藏层维度
    
    return Config()

def demo_moe_gate():
    """
    MoE门控机制演示函数
    """
    print("=" * 80)
    print("🚀 DeepSeek-V3 MoE门控机制演示")
    print("=" * 80)
    
    # 创建配置
    config = create_moe_config()
    print(f"📋 配置信息:")
    print(f"   - 总专家数: {config.n_routed_experts}")
    print(f"   - 每个token选择专家数: {config.num_experts_per_tok}")
    print(f"   - 专家分组数: {config.n_group}")
    print(f"   - 最多路由节点数: {config.topk_group}")
    print(f"   - 隐藏层维度: {config.hidden_size}")
    
    # 创建MoE门控模块
    moe_gate = MoEGate(config)
    print(f"\n🔧 MoE门控模块创建完成")
    
    # 创建示例输入
    batch_size = 4
    seq_len = 64
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\n📥 示例输入:")
    print(f"   - 形状: {hidden_states.shape}")
    print(f"   - 数据类型: {hidden_states.dtype}")
    print(f"   - 数值范围: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
    
    # 设置为推理模式
    moe_gate.eval()
    
    # 执行前向传播
    print(f"\n🔄 执行前向传播...")
    with torch.no_grad():
        topk_idx, topk_weight = moe_gate(hidden_states)
    
    # 分析结果
    print(f"\n📊 路由结果分析:")
    print(f"   - 选择的专家索引形状: {topk_idx.shape}")
    print(f"   - 专家权重形状: {topk_weight.shape}")
    
    # 统计专家使用情况
    expert_usage = torch.zeros(config.n_routed_experts)
    for idx in topk_idx.flatten():
        expert_usage[idx] += 1
    
    print(f"\n🎯 专家使用统计:")
    print(f"   - 被使用的专家数: {(expert_usage > 0).sum().item()}")
    print(f"   - 使用最多的专家: {expert_usage.argmax().item()} (使用{expert_usage.max().item()}次)")
    print(f"   - 使用最少的专家: {expert_usage.argmin().item()} (使用{expert_usage.min().item()}次)")
    
    # 统计权重分布
    print(f"\n⚖️ 权重分布统计:")
    print(f"   - 权重最小值: {topk_weight.min():.4f}")
    print(f"   - 权重最大值: {topk_weight.max():.4f}")
    print(f"   - 权重平均值: {topk_weight.mean():.4f}")
    print(f"   - 权重标准差: {topk_weight.std():.4f}")
    
    # 检查负载均衡
    print(f"\n⚖️ 负载均衡检查:")
    total_usage = expert_usage.sum()
    expected_usage = batch_size * seq_len * config.num_experts_per_tok
    print(f"   - 总专家使用次数: {total_usage}")
    print(f"   - 期望使用次数: {expected_usage}")
    print(f"   - 负载均衡度: {1 - abs(total_usage - expected_usage) / expected_usage:.4f}")
    
    # 生成可视化
    print(f"\n🎨 生成路由过程可视化...")
    moe_gate.visualize_routing_process(hidden_states, save_path="moe_routing_visualization.png")
    
    print(f"\n✅ 演示完成!")
    print("=" * 80)

if __name__ == "__main__":
    # 运行演示
    demo_moe_gate() 