import numpy as np
from scipy.signal import argrelextrema
# 加载数据
data = np.load('data.npy')
# 求极大值
maxn = argrelextrema(data, np.greater)
# 求极小值
minn = argrelextrema(data, np.less)
# 打印极大值的索引
print(maxn)
# 打印极小值的索引
print(minn)