import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import copy

# -----------------------------
# 参数与初始状态
# -----------------------------
N = 4  # 进程数

def init_states():
    """
    初始化每个进程的向量，每个向量长度为 N，
    每个元素初始为进程编号+1。例如：P0 -> [1, 1, 1, 1]
    """
    return {i: [i+1 for _ in range(N)] for i in range(N)}

# frames 用于记录每一步的状态，方便动画展示
frames = []  # 每一帧记录：{'phase': 阶段, 'step': 步骤, 'desc': 描述, 'states': {进程: [向量]}}

def record_frame(phase, step, desc, states):
    # 深拷贝当前状态，确保后续更新不影响历史记录
    state_copy = {pid: list(vec) for pid, vec in states.items()}
    frames.append({
        'phase': phase,
        'step': step,
        'desc': desc,
        'states': state_copy
    })

# 初始化所有进程状态
states = init_states()
record_frame("初始状态", 0, "各进程初始数据", states)

# -----------------------------
# 1. Reduce-Scatter 阶段
# -----------------------------
# 模拟过程中，每个进程在每一步从其左邻居接收一个 segment，并将其累加到本地对应位置上。
for s in range(1, N):
    new_states = copy.deepcopy(states)
    for i in range(N):
        sender = (i - 1) % N  # 左邻居
        # 当前 step 对应的 segment 下标，采用 (i - s) mod N
        seg_index = (i - s) % N
        received_value = states[sender][seg_index]
        new_states[i][seg_index] += received_value
        record_frame("Reduce-Scatter", s,
                     f"进程 {sender} -> 进程 {i}: segment {seg_index} 的值 {received_value} 累加",
                     new_states)
    states = new_states

# -----------------------------
# 2. Allgather 阶段
# -----------------------------
# 模拟过程中，每个进程将规约好的局部 segment 传递给其他进程，
# 使得所有进程最终都拥有完整的全局规约结果向量。
for s in range(1, N):
    new_states = copy.deepcopy(states)
    for i in range(N):
        sender = (i - 1) % N  # 左邻居
        # 传递的 segment 下标同样采用 (i - s) mod N
        seg_index = (i - s) % N
        # 进程 i 接收到 sender 对应 segment 的结果，更新本地该位置（模拟广播）
        new_states[i][seg_index] = states[sender][seg_index]
        record_frame("Allgather", s,
                     f"进程 {sender} -> 进程 {i}: segment {seg_index} 的值 {states[sender][seg_index]} 广播",
                     new_states)
    states = new_states

record_frame("完成", 0, "最终 AllReduce 结果", states)

# -----------------------------
# 3. 可视化过程
# -----------------------------
# 使用 networkx 构造一个环形网络图，每个节点代表一个进程
G = nx.DiGraph()
nodes = list(range(N))
G.add_nodes_from(nodes)
# 构造环形拓扑：节点 i 指向 (i+1)%N
for i in nodes:
    G.add_edge(i, (i+1) % N)

# 采用圆形布局使节点均匀分布
pos = nx.circular_layout(G)

fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    # 绘制图中边（箭头表示数据传递方向）
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=20)
    # 每个节点上显示进程编号及当前向量状态
    current_states = frame['states']
    labels = {i: f"P{i}\n{current_states[i]}" for i in current_states}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)
    # 标题显示当前阶段、步骤及通信描述
    ax.set_title(f"Stage：{frame['phase']} Process：{frame['step']}\n{frame['desc']}")
    ax.axis('off')

# 创建动画，每帧间隔 1000 毫秒（1 秒）
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
plt.show()
