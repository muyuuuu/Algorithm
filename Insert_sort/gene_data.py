###
 # @Author         : lanling
 # @Date           : 2020-04-13 14:21:26
 # @LastEditTime: 2020-04-13 14:24:54
 # @FilePath       : \Insert_sort\gene_data.py
 # @Github         : https://github.com/muyuuuu
 # @Description    : 生成随机数据用于排序
 # @佛祖保佑，永无BUG
###

import numpy as np
# 第一个参数是左区间 a，第二个参数是右区间 b，第三个参数是个数 c
# 即会产生 c 个 [a, b) 区间内的整数
a = np.random.randint(0, 300000, 300000)
# fmt 表示取消科学计数法的保存形式
np.savetxt('Three_hundred_thousand.txt', a, fmt='%d')
