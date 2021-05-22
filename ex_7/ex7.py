import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

# 创建插值点
X = np.arange(0, 360, 1)

ang = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
       150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
       270, 280, 290, 300, 310, 320, 330, 340, 350, 360]

pwr = [1.6, 1.615, 1.632, 1.563, 1.653, 1.688, 1.73, 1.76, 1.799,
       1.82, 1.81, 1.756, 1.736, 1.733, 1.732, 1.705, 1.671, 1.637,
       1.604, 1.585, 1.602, 1.621, 1.645, 1.669, 1.675, 1.705, 1.741,
       1.786, 1.82, 1.822, 1.793, 1.76, 1.696, 1.645, 1.677, 1.572]

ipo2 = spi.splrep(ang, pwr, k=2)  # 样本点导入，生成参数
iy2 = spi.splev(X, ipo2)  # 根据观测点和样条参数，生成插值

# 极坐标
ax = plt.subplot(111, projection='polar')

theta = np.pi / 180 * X  # 生成角度
# 显示网格
plt.grid(True)

# 设置半径
ax.set_rmax(3)

# 显示数据点并不画线
plt.plot(ang, pwr, '*')

plt.plot(theta, iy2)

plt.savefig("图二.jpeg")
plt.show()
