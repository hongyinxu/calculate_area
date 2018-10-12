# # -*- coding：utf-8 -*-
from matplotlib.tri import (Triangulation, UniformTriRefiner, CubicTriInterpolator)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import *


def dipole_potential(x, y):
     """生成每个网格点上的 z 值"""
     r_sq = x**2 + y**2
     theta = np.arctan2(y, x)
     z = np.cos(theta)/r_sq
     # print('z:', z)a
     return (np.max(z) - z) / (np.max(z) - np.min(z))


def cal_area(a):
    """定义一个计算三角元面积的函数，返回面积

    a 接收一个数组，数组为三个点的坐标值
    如:
    [[2, 3],
    [4, 5],
    [6, 7]]a
    """
    one = np.ones(3)
    """
    注意，插入一列时，列长度应与数组的行数相等；插入一行时，行长度应与数组的列数相等
    数组a加上数组one， len(a[0])的值为2，代表第2行或第2列， axis=0时，代表行， axis=1时代表列
    """
    det1 = np.insert(a, len(a[0]), values=one, axis=1)
    """三角元的面积等于行列式的值 * 0.5
        行列式的格式为:
        [[2, 3, 1],
         [4, 5, 1],
         [6, 7, 1]
        ]
    面积计算参考网址：https://wenku.baidu.com/view/4dc519202f60ddccda38a09d.html
    """
    area = 0.5 * det(det1)
    return area


n_angles = 3    # 控制角度数组长度： 3 个
n_radii = 2     # 控制半径数组长度： 2 个
min_radius = 0.2
radii = np.linspace(min_radius, 0.6, n_radii)   # [0.2, 0.6]
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)    # [0, 2.09, 4.18, ]
# x_t_0 = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)   # [0, 2.09, 4.18, ]

"""np.newaxis增加一个维度

x_t 数据:
   [[[0.       ]
  [0.       ]
  [0.       ]
  [0.       ]]

 [[2.0943951]
  [2.0943951]
  [2.0943951]
  [2.0943951]]

 [[4.1887902]
  [4.1887902]
  [4.1887902]
  [4.1887902]]]     
"""
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
# x_t = np.repeat(angles[..., np.newaxis], n_radii, axis=1)       # tuple(3, 4, 1)

# print('angels', angles)
# print(angles[:, 1::2])      # [[0.       ] [2.0943951] [4.1887902]]
angles[:, 1::2] += np.pi / n_angles
# print(angles[:, 1::2])      # [[1.04719755] [3.14159265] [5.23598776]]

""" # flatten降低矩阵维度 """
x = (radii*np.cos(angles)).flatten()    # x: [ 0.2  0.3 -0.1 -0.6 -0.1  0.3]
y = (radii*np.sin(angles)).flatten()    # # flatten降低矩阵维度
# print(y)    # [ 0.00000000e+00  5.19615242e-01  1.73205081e-01  7.34788079e-17  -1.73205081e-01 -5.19615242e-01]

z = dipole_potential(x, y)

triang = Triangulation(x, y)

# Mask off unwanted triangles.
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)) < min_radius)

refiner = UniformTriRefiner(triang)
tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)

# -----------------------------------------------------------------------------
# Computes the electrical field (Ex, Ey) as gradient of electrical potential
# -----------------------------------------------------------------------------
# tci = CubicTriInterpolator(triang, -z)
# # Gradient requested here at the mesh nodes but could be anywhere else:
# (Ex, Ey) = tci.gradient(triang.x, triang.y)
# E_norm = np.sqrt(Ex**2 + Ey**2)

"""hxu添加的代码"""
# 取出坐标x, y 的值
X = triang.x
Y = triang.y
# print(X)
total_area, total_zeta = [], []
# triangles数据格式：[[1, 2, 4], [4, 5, 10], ...]每个数组中的三个数代表nodes节点编号
for data in triang.triangles:
    """根据节点编号找到对应的x, y"""
    da1 = [X[data[0]], Y[data[0]]]
    da2 = [X[data[1]], Y[data[1]]]
    da3 = [X[data[2]], Y[data[2]]]
    """
    将标量场看成是线性分布的，average = (data1 + data2 + data3)/3
    矢量时，采用最小二乘，利用周围的4个三角单元上的速度矢量进行计算。
    """
    zeta = (z[data[0]] + z[data[1]] + z[data[2]])/3.0
    total_zeta.append(zeta)
    """生成cal_area()函数接收的数组"""
    arr = np.array([da1, da2, da3])
    area = cal_area(arr)
    total_area.append(area)

# sum1 = np.sum(total_area)
"""根据节点上的 z 值筛选面积"""
data_select = []
# area_select = []
[data_select.append(total_area[i]) for i in range(len(total_zeta)) if total_zeta[i] >= 0.5]   # 筛选出面积大于0.05的三角元
# [area_select.append(total_zeta[i]) for i in range(len(total_zeta)) if total_zeta[i] >= 0.5]

"""----------------------------"""
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

ax.triplot(triang, color='0.8')
plt.show()