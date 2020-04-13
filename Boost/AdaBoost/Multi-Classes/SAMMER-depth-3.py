# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 分类的编号为 0 1 2 3 
X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=4, random_state=1)
                        
n_split = 3000

X_train, X_test = X[n_split:], X[:n_split]
Y_train, Y_test = y[n_split:], y[:n_split]

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                        algorithm="SAMME.R",
                        n_estimators=600)

clf.fit(X_train, Y_train)

train_errors = []
test_errors = []

for train_error, test_error in zip(clf.staged_predict(X_train), clf.staged_predict(X_test)):
    train_errors.append(1 - accuracy_score(train_error, Y_train))
    test_errors.append(1 - accuracy_score(test_error, Y_test))

x = np.linspace(1, 600, 600)
plt.style.use('ggplot')
plt.plot(x, train_errors, 'r', label='SAMME.R Train Error')
plt.plot(x, test_errors, 'b', label='SAMME.R Test Error')
plt.legend(loc='upper right', fancybox=True)
plt.xlabel('Eestimator Numbers')
plt.ylabel('Error Rate')
# plt.show()
plt.savefig('error_rate.png', dpi=250)

# https://www.zybuluo.com/yxd/note/614495
print(clf.feature_importances_)

score = clf.score(X_test, Y_test)

print(score)
