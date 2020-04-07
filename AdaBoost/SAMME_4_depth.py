import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_hastie_10_2

# Construct dataset
# 创建8000数据 4个特征 2个类
X, Y = make_hastie_10_2(n_samples=12000, random_state=1)

n_split = 2000

# 分类器的个数
num_estimators = [1, 5, 20, 50, 100]

X_train, X_test = X[n_split:], X[:n_split]
Y_train, Y_test = Y[n_split:], Y[:n_split]

for num in num_estimators:
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                            algorithm="SAMME",
                            n_estimators=num)
    
    # 记录算法的训练时间
    start = time.time()
    bdt.fit(X_train, Y_train)
    end = time.time()

    score = bdt.score(X_test, Y_test)

    print('Estimators ===>', num, ' Accuracy ===>', score, 'Time ===>', round(end - start, 2), 's')

print(bdt.base_estimator_)