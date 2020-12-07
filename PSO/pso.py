import time
import numpy as np
import matplotlib.pyplot as plt


def op_func(x):
    return (x - 2) ** 2 - 6 * x + np.sin(x) * 12 / x - 9 * np.cos(x) * x


class PSO(object):
    def __init__(self, left, right, particle_num, iter_num):
        # 约束范围
        self._left, self._right = left, right
        # 迭代次数
        self._iter_num = iter_num
        # 粒子数量
        self._particle_num = particle_num
        # 保存的数据
        self._data = np.zeros(self._particle_num)
        # 常量
        self._c1, self._c2 = 1, 1
        # 权重
        self._w = 0.5
        # 初始化位置，受到约束条件的限制
        self._x = np.random.uniform(self._left, self._right, self._particle_num)
        self._data = np.vstack((self._data, self._x))
        # 初始化速度，[0, 2] 之间取值
        self._v = np.random.uniform(0, 2, self._particle_num)
        # 当前最优
        self._pbest = 0
        # 全局最优
        self._gbest = 1000000
        # 每个点的性能
        self._fitness = np.zeros(self._particle_num)

    def update(self):
        return self._update()

    def _update(self):
        for i in range(self._iter_num):
            # 计算每个点的 fitness
            self._fitness = op_func(self._x)
            # 当前最优
            self._pbest = self._x[np.argmin(self._fitness)]
            # 全局最优
            if i == 0:
                self._gbest = self._pbest
            elif op_func(self._pbest) < op_func(self._gbest):
                self._gbest = self._pbest
            # 速度更新
            r1, r2 = np.random.rand(), np.random.rand()
            self._v = self._w * self._v + \
                r1 * self._c1 * (self._pbest - self._x) + r2 * self._c2 * (self._gbest - self._x) 
            # 位置更新
            self._x += self._v
            # 越界后强制到边界点
            self._x = np.clip(self._x, self._left, self._right)
            # 保存
            self._data = np.vstack((self._data, self._x))

    def result(self):
        return self._result()
    
    def _result(self):
        print('x = {}, value = {}'.format(round(self._gbest, 2), round(op_func(self._gbest), 2)))
        return self._data

left = -4
right = 4
particle_num = 10
iter_num = 15

a = PSO(right=right, left=left, particle_num=particle_num, iter_num=iter_num)
a.update()
data = a.result()

fig, ax = plt.subplots()
plt.style.use('ggplot')
ax.set_xlim(left-1, right+1)
ax.set_ylim(-30, 50)
print(data.shape)

x = np.linspace(left, right, 100)
for i in range(1, 2+iter_num):
    ax.cla()   # 清除键
    ax.plot(x, op_func(x))
    ax.scatter(data[i], op_func(data[i]))
    ax.legend()
    plt.pause(0.2)
plt.show()