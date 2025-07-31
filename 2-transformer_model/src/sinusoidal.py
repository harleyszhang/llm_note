import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 定义 θ(t) 函数，增加参数 d
def theta(t, d):
    return (1/d) ** t

# 定义 f(x) 函数，增加参数 d
def f(x, d):
    # 使用 lambda 表示积分变量 t
    integrand = lambda t: np.exp(1j * x * theta(t, d))
    # 使用 quad 进行积分，返回实部
    integral, error = quad(lambda t: np.real(integrand(t)), 0, 1)
    return integral

# 定义 x 的范围
x_values = np.linspace(-128, 128, 1000)

# 定义不同的 d 值
d_values = [100, 1000, 10000, 100000]

# 创建图像
plt.figure(figsize=(10, 6))

# 计算并绘制每个 d 对应的 f(x)
for d in d_values:
    f_values = [f(x, d) for x in x_values]
    plt.plot(x_values, f_values, label=f'd = {d}')

# 设置图像的标签和标题
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) for Different d Values')
plt.legend()
plt.grid(True)
plt.show()