import math as math
import scipy as scipy
import matplotlib.pyplot as plt
import numpy as np
import sympy as sympy
import xlrd as xlrd
from scipy.stats import norm

# 打开excel
wb = xlrd.open_workbook("data.xls")
# 列出所有表
names = wb.sheet_names()
# 输出所有的表名，以列表的形式
print(names)
# 通过sheet索引获得sheet对象
worksheet = wb.sheet_by_index(0)
print(worksheet)

# 通过sheet名获得sheet对象
worksheet = wb.sheet_by_name("Sheet1")

'''对sheet进行操作'''
# 获取该表总行数
nrows = worksheet.nrows
# 获取该表总列数
ncols = worksheet.ncols
'''
# 循环打印每一行
for i in range(nrows):
    print(worksheet.row_values(i))  # 以列表形式读出，列表中的每一项是str类型
'''
# 获取第一列的内容
col_data = worksheet.col_values(0)
col_data = np.array(col_data)
col_data = col_data[:, np.newaxis]

# 获取第二列的内容
pwr = worksheet.col_values(1)
pwr = np.array(pwr)
# 升维
pwr = pwr[:, np.newaxis]

'''最小二乘拟合'''
#
# x = np.arange(0, 207, 0.1)
h = int(input('请输入拟合阶数'))
n = len(col_data)
N = h + 1
G = np.zeros((n, N))
j: int
for j in range(0, N):
    for i in range(0, n):
        G[i][j] = math.pow(col_data[i], j)

G = np.append(G, pwr, axis=1)

'''绘图'''
# 显示中文横纵坐标
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设定横纵坐标
plt.xlabel("采样点")
plt.ylabel("功率/W")
# 设定网格
plt.grid(True)
# 绘制原始数据图
# plt.plot(col_data, pwr)
# plt.savefig("图一.jpeg")
# plt.show()

w = np.zeros((1, n))

for k in range(0, N):
    sigma = 0
    Gm = 0
    for i in range(k, n):
        Gm = Gm + math.pow(G[i][k], 2)
    sigma = -1 * np.sign(G[k][k]) * np.sqrt(np.sum(Gm))
    w[0, k] = G[k][k] - sigma

    for j in range(k + 1, n):
        w[0, j] = G[j][k]

    B = sigma * w[0, k]

    G[k][k] = sigma

    for j in range(k + 1, N + 1):
        wg = 0
        for i in range(k, n):
            wg = wg + w[0, i] * G[i][j]
        t = wg / B

        for i in range(k, n):
            G[i][j] = G[i][j] + t * w[0, i]
'''解三角方程'''

x = np.zeros((1, N))

x[0, N - 1] = G[N - 1][N] / G[N - 1][N - 1]
for i in range(N - 1, -1, -1):
    gx = 0
    for j in range(i + 1, N):
        gx = gx + G[i][j] * x[0, j]
    x[0, i] = (G[i][N] - gx) / G[i][i]

L = np.size(x)
Y = np.zeros((1, 207))

for i in range(0, L):
    for j in range(0, 207):
        Y[0, j] = Y[0, j] + x[0, i] * math.pow(col_data[j], i)
'''求导数'''
# 设定形参
X = sympy.Symbol("X")
# 组成函数
fun = 0
for i in range(0, L):
    fun = fun + x[0, i] * X ** i
# 求导
dif = sympy.diff(fun, X)
dif_ans = np.zeros((1, 207))
# 赋值
for j in range(0, 207):
    dif_ans[0, j] = dif.subs(X, j)
# 取行值画图
col_data = col_data.T[0]
dif_ans = dif_ans[0]

# 平滑曲线
f = scipy.interpolate.interp1d(col_data, dif_ans, kind='cubic')
ndif_ans = f(col_data)
mu =np.mean(dif_ans)
sigma1=np.std(dif_ans)
pdf = norm.pdf(dif_ans, mu, sigma1)


# 绘图

plt.plot(col_data, pdf,'r--')
plt.close()
plt.plot(col_data, ndif_ans)
plt.savefig('图二.jpeg')
plt.show()
plt.close()


# 显示网格
plt.grid(True)
# 显示中文横纵坐标
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("点")

plt.ylabel("功率/W")
# 提出第一维

Y = Y[0]
# 显示数据点并不画线
plt.plot(col_data, Y)
plt.savefig('图一.jpeg')
plt.show()
