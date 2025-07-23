import numpy as np
import os

class Perceptron() :
    # 사전 설정 함수 __init__ 정의
    def __init__(self, eta=0.01, n_iter=50, random_state=0) :
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y) :
        inputs = np.random.default_rng(self.random_state)
        self.w = inputs.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = float(0)
        self.error = []

        for _ in range(self.n_iter) :
            error = 0

            for xi, target in zip(X, y) :
                update = - self.eta * (self.predict(xi) - target)
                self.w += update * xi
                self.b += update
                error += int(update != 0.0)

            self.error.append(error)

        return self

    def net_input(self, X) :
        return np.dot(X, self.w) + self.b

    def predict(self, X) :
        return np.where(self.net_input(X) >= 0, 1, 0)

X = np.array([[1, 3],
              [5, 7]])
y = np.array([0, 1, 0, 1])

model = Perceptron()
model.fit(X, y)
predictions = model.predict(X)

import pandas as pd

df = pd.read_excel('iris_data.xlsx', header=None)

import matplotlib.pyplot as plt

y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
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