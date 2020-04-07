import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn import datasets
import matplotlib.pyplot as plt

# Construct dataset
# 创建8000数据 4个特征 2个类
X, Y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)


n_split = 2000

X_train, X_test = X[n_split:], X[:n_split]
Y_train, Y_test = Y[n_split:], Y[:n_split]

train_errors = []
test_errors = []

num = 350

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                        algorithm="SAMME",
                        n_estimators=num)

bdt.fit(X_train, Y_train)

for train_error, test_error in zip(bdt.staged_predict(X_train), bdt.staged_predict(X_test)):
    train_errors.append(zero_one_loss(train_error, Y_train))
    test_errors.append(zero_one_loss(test_error, Y_test))

x = np.linspace(1, num, num)
plt.style.use('ggplot')
plt.plot(x, train_errors, 'r', label='SAMME Train Error')
plt.plot(x, test_errors, 'b', label='SAMME Test Error')
plt.legend(loc='upper right', fancybox=True)
plt.xlabel('Eestimator Numbers')
plt.ylabel('Error Rate')
# plt.show()
plt.savefig('error_rate.png', dpi=250)