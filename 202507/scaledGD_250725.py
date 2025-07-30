import numpy as np

class AdalineGD() :
    def __init__(self, eta=0.01, n_iters=30, random_state=0) :
        self.eta= eta
        self.n_iters = n_iters
        self.random_state = random_state

    def fit(self, X, y) :
        default = np.random.default_rng(0)
        self.w = default.normal(0, 0.1, size=X.shape[1])
        self.b = np.float32(0)
        self.loss = []

        for i in range(self.n_iters) :
            result = self.matcal(X)
            output = self.activation(result)
            error = y - output

            self.w += self.eta * 2 * (X.T @ error) / X.shape[0]
            self.b += self.eta * 2 * error.mean()

            mse = (error ** 2).mean()

            self.loss.append(mse)

        return self

    def matcal(self, X) :
        return X @ self.w + self.b

    def activation(self, matcal_outcome) :
        return matcal_outcome

    def predict(self, X) :
        return np.where(self.activation(self.matcal(X)) >= 0.5, 1, 0)

from matplotlib.colors import ListedColormap
# 결정 경계를 등고선으로 표현하기
def plot_decision_regions(X, y, model, resolution=0.01) :
    markers = ['o', 'x', '*', 's', '^']
    colors = ['blue', 'red', 'gray', 'green', 'magenta']
    # 사용할 마커와 색상을 사전 정의한 뒤 ListedColormap으로 활용
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.2, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)) :
        plt.scatter(x=[X[y==cl, 0]], y=[X[y==cl, 1]],
                    alpha=0.8, cmap=colors[idx],
                    marker=markers[idx], edgecolors='black',
                    label=f'class {cl}')
        # plt.show()

import pandas as pd
df = pd.read_excel('iris_data.xlsx', header=None)

y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:, [0, 2]].values

# 꽃받침과 꽃잎 길이 표준화
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

import matplotlib.pyplot as plt
# 표준화한 독립변수 데이터와 타깃 값을 학습 모델에 투입
adaline_gd = AdalineGD(eta=0.1, n_iters=30).fit(X_std, y)
# 학습한 모델을 바탕으로 표준화한 데이터와 타깃 값을 활용해 등고선 제작
plot_decision_regions(X_std, y, model=adaline_gd)

plt.title('AdalineGD')
plt.xlabel('Standardized sepal length')
plt.ylabel('Standardized petal length')

plt.tight_layout()
plt.legend(loc='best')
plt.show()

plt.plot(range(1, len(adaline_gd.loss) + 1), adaline_gd.loss, marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()