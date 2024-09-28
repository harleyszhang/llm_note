## 摘要

`Softmax` 函数在机器学习中应用广泛，此前的多项研究提出了更快的替代方案。本文针对提升原始 Softmax 性能，提出了一种通过**减少内存访问**来计算经典 Softmax 的方法，并推测这种减少内存访问的方法可以提升 Softmax 在实际硬件上的性能。基准测试也证实了这一推测：Softmax 的速度最多提升 `1.3` 倍，且 Softmax 和 `TopK` 结合并融合后的速度提升最多可达 `5` 倍。

## 原来的 Softmax

Softmax 函数是一种常用于机器学习，特别是多分类问题中的激活函数。它的作用是将一个任意实数向量转换为一个概率分布，并确保输出的概率和为 1。

给定输入向量 $\mathbf{x} = [x_1, x_2, \dots, x_n]$，$Softmax(x)$ 函数的输出定义为：

$$y_i = \frac{e^{x_i}}{\sum_j^{n} e^{x_j} }$$

其中，$x,y\in  \mathbb{R}^{n}$。朴素的 Softmax [算法 1] 实现需要对 $\mathbf{x}$ 进行二次内存访问，一次计算归一化项 $d_n$，另一次计算输出值 $y_i$，加上写输出结果 $y_i$，即**每个向量元素都需要进行三次内存访问：两次读取和一次写入**。

$\text{算法 1 朴素 softmax} \\
\begin{aligned}
1: & \quad d_0 \leftarrow 0 \\
2: & \quad \textbf{for} \ j \leftarrow 1, n \ \textbf{do} \\
3: & \quad \quad d_j \leftarrow d_{j-1} + e^{x_j} \\
4: & \quad \textbf{end for} \\
5: & \quad \textbf{for} \ i \leftarrow 1, n \ \textbf{do} \\
6: & \quad \quad y_i \leftarrow \frac{e^{x_i}}{d_n} \\
7: & \quad \textbf{end for}
\end{aligned}
$

在实际硬件上，由于表示对数字范围有限，算法 1 的第 3 行可能会因指数运算而发生溢出或下溢，因此目前通用的 Softmax 实现（更安全的形式）中为了防止数值溢出还需要再额外减掉一个 `max` 最大值：

$$m = \text{max}_{k}^{n} x_k\\
y_i = \frac{e^{(x_i - m)}}{\sum_j^{n} e^{(x_j -m)}}
$$

大部分深度学习框架都是采用这个更安全的朴素实现，算法流程见 [算法2]。但安全 Softmax 对输入向量进行了三次遍历：第一次计算最大值 $m_n$，第二次计算归一化项 $d_n$，第三次计算最终值 $y_i$；这导致每个向量元素总共需要 4 次内存访问。之前的 Softmax 算法的内存访问（`MAC`）偏大，作者希望对此进行改进。

$\text{算法 2 安全 Softmax} \\
\begin{aligned}
1: & \quad m_0 \leftarrow -\infty \\
2: & \quad \textbf{for} \ k \leftarrow 1, n \ \textbf{do} \\
3: & \quad \quad m_k \leftarrow \max(m_{k-1}, x_k) \\
4: & \quad \textbf{end for} \\
5: & \quad d_0 \leftarrow 0 \\
6: & \quad \textbf{for} \ j \leftarrow 1, n \ \textbf{do} \\
7: & \quad \quad d_j \leftarrow d_{j-1} + e^{x_j - m_n} \\
8: & \quad \textbf{end for} \\
9: & \quad \textbf{for} \ i \leftarrow 1, n \ \textbf{do} \\
10: & \quad \quad y_i \leftarrow \frac{e^{x_i - m_n}}{d_n} \\
11: & \quad \textbf{end for}
\end{aligned}$

## Online normalizer calculation
## 参考资料

- [Flash Attention (零) - Online Softmax](https://zhuanlan.zhihu.com/p/672664395)