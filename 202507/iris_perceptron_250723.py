import numpy as np
import os

class Perceptron():
    def __init__(self, eta=0.01, n_iter=50, random_state=0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        inputs = np.random.default_rng(self.random_state)
        self.w = inputs.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = float(0)
        self.error = []

        for _ in range(self.n_iter):
            error = 0

            # X와 y의 길이가 다르면 zip이 짧은 쪽에 맞춰서 반복합니다.
            # 이 때문에 학습이 제대로 안 될 수 있습니다.
            for xi, target in zip(X, y):
                # predict 내부에서 self.w와 self.b가 사용됨
                self.w += self.eta * xi * (target - self.predict(xi))
                self.b += self.eta * (target - self.predict(xi))
                error += int(self.eta * (target - self.predict(xi)) != 0.0)
                # 원리적으로 다른 코드와 통일하기 위해 이렇게 썼지만
                # self.eta ~ 항이 너무 길기 때문에 다른 이름의 변수로 미리 지정해 놓고 써도 ok

            self.error.append(error)

        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

X = np.array([[1, 3],
              [5, 7]])
y = np.array([0, 1])

model = Perceptron(eta=0.01, n_iter=20)
model.fit(X, y)
predictions = model.predict(X)

import pandas as pd
# iris 데이터 다운로드
df = pd.read_excel('iris_data.xlsx', header=None)

import matplotlib.pyplot as plt
# df에서 원하는 자료를 추출해 np.ndarray로 변형
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)  # setosa면 0, 다른 종은 1로 변환
X = df.iloc[:, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='dodgerblue', marker='o', label='versi')
plt.xlabel('length of sepal')
plt.ylabel('length of petal')
plt.legend(loc='best')
# plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, model, resolution=0.01) :
    markers = ['o', 'x', '*', 's', '^']
    colors = ['blue', 'red', 'gray', 'green', 'magenta']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.2, cmap=cmap)

    plt.xlim(x1_min-1, x1_max+1)
    plt.ylim(x2_min-1, x2_max+1)

    for idx, cl in enumerate(np.unique(y)) :
        plt.scatter(x=[X[y==cl, 0]], y=[X[y==cl, 1]],
                    alpha=0.8, colors=colors[idx],
                    marker=markers[idx], edgecolors='black',
                    label=f'class {cl}')
        plt.show()

plot_decision_regions(X, y, model=Perceptron(eta=0.01, n_iter=30))
plt.xlabel('length of sepal')
plt.ylabel('length of petal')
plt.legend(loc='best')
plt.show()