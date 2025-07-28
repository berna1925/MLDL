import numpy as np

class LogisticRegressionGD() :
    # 하이퍼파라미터 초기화
    def __init__(self, eta, n_iter, random_state):
        self.eta = eta
        self.n_iter =n_iter
        self.random_state = random_state

    # 학습 함수 fit
    def fit(self, X, y) :
        random_factor = np.random.default_rng(self.random_state)
        # 가중치 및 절편 초기화
        self.w = random_factor.normal(0, 0.1, size=X.shape[1])
        self.b = 0.0
        self.losses = []

        # 행렬*벡터 단위로 이뤄지는 로지스틱 메커니즘을 고려해 에폭 단위로만 반복문 제시
        for i in range(self.n_iter) :
            # 연산 결과가 행렬*벡터 단위로 통째로 이뤄지므로 error 벡터도 한 번에 생성
            outputs = self.activation(self.matcal(X))
            errors = y - outputs

            # 에폭 단위 가중치, 절편 업데이트
            self.w += self.eta * 2 * (X.T @ errors) / X.shape[0]
            # 절편은 이미 벡터 단위로 error가 나와 있어 평균 내는 것으로 충분
            self.b += self.eta * 2 * errors.mean()

            loss = self.loss_function(outputs, y)
            self.losses.append(loss.mean())

        return self

    def matcal(self, X) :
        return X @ self.w + self.b

    def activation(self, z) :
        return 1 / (1 + np.exp(-np.clip(z, -1e+4, 1e+4)))

    def loss_function(self, output, target) :
        epsilon = 1e-10
        # log 값에 들어가야 하므로 양수가 되어야 하니 최소값을 미리 지정
        output = np.clip(output, epsilon, 1 - epsilon)

        return -(np.log(output) @ target + np.log(1 - output) @ (1 - target))

    def prediction(self, X) :
        # 시그모이드 함수의 계산 범위 (0, 1)에 따라 0.5 이상은 양성으로 판정되게 지정
        return np.where(self.activation(self.matcal(X)) >= 0.5, 1, 0)

from sklearn import datasets
# datasets.load~ 로 원하는 데이터셋 호출
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

X_train_subset = X_train_scaled[(y_train == 0) | (y_train == 1)]
y_train_subset = y_train[(y_train == 0) | (y_train == 1)]

from iris_perceptron_250723 import plot_decision_regions

X_combined_std = np.vstack([X_train_scaled, X_test_scaled])
y_combined = np.hstack([y_train, y_test])

lrgd = LogisticRegressionGD(eta=0.1, n_iter=30, random_state=0)
lrgd.fit(X_train_scaled, y_train)

print(plot_decision_regions(X=X_combined_std, y=y_combined, model=lrgd))