from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_hastie_10_2
import time

# 创建8000数据 4个特征 2个类
X, Y = make_hastie_10_2(n_samples=12000, random_state=1)

n_split = 2000

# 分类器的个数
num_estimators = [1, 5, 20, 50, 100, 200]

X_train, X_test = X[n_split:], X[:n_split]
Y_train, Y_test = Y[n_split:], Y[:n_split]

for num in num_estimators:
    clf = AdaBoostClassifier(n_estimators=num,
                             random_state=0,
                             algorithm='SAMME')
    
    # 记录算法的训练时间
    start = time.time()
    clf.fit(X_train, Y_train)
    end = time.time()

    score = clf.score(X_train, Y_train)

    print('Estimators ===>', num, ' Accuracy ===>', score, 'Time ===>', round(end - start, 2), 's')

    if (num == 5):
        print(clf.base_estimator_)
        print('Errors ', clf.estimator_errors_)
        print('Weights ', clf.estimator_weights_)

print(clf.base_estimator_)