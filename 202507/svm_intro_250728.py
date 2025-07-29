import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)

plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
            c='tomato', marker='o',
            label='class0')

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='royalblue', marker='s',
            label='class1')

plt.grid(True)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend(loc='best')

plt.tight_layout()
# plt.show()

from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1, gamma=0.1, random_state=0)
# rbf는 방사 기저 함수의 약어로 비선형 결정 경계를 다룰 때 주효
# xor은 비선형 경계로만 나눌 수 있어 linear를 쓰면 안 됨
svm.fit(X_xor, y_xor)

from iris_perceptron_250723 import plot_decision_regions

plot_decision_regions(X_xor, y_xor, model=svm)
plt.legend(loc='best')
plt.tight_layout()
plt.show()