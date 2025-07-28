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
            # 퍼셉트론 모델은 개별 훈련 샘플마다 평가한 뒤 오차 계산
            # → update가 일차방정식 한 줄 단위로 순차적 경사하강법의 방식(예측이 맞으면 업데이트 없는 구조)
            # 아달린 모델은 행렬 단위 연산을 한 번에 진행한 뒤 오차 계산
            # → update가 전체 연립방정식 단위로 이뤄지는 배치 경사하강법의 방식(예측 값을 맞혀도 거의 대부분 업데이트 발생)
            output = self.activation(result)
            # 연산 결과를 선형 활성화 함수에 한 번 통과시킨 뒤
            error = y - output
            # 타깃과 예측의 차이를 비교해 error를 산출
            # 이것을 제곱해서 평균한 것이 MSE로 아달린이 회귀 모델의 오차 산출 방법을 따름을 알 수 있음

            self.w += self.eta * 2 * (X.T @ error) / X.shape[0]
            # w := w - η (∂L/∂w) = w + n (-∂L/∂w)
            # 업데이트되는 가중치의 값은 학습률에 미분 값을 곱한 수치를 빼는 것
            # MSE의 식을 w_i로 각각 미분하면 -∂L/∂w = 2 * (X.T @ error)[스칼라 값이므로 시그마와 동일] / X.shape[0]
            # 왜 -가 붙냐면 y에서 y_hat을 빼기 때문...

            self.b += self.eta * 2 * error.mean()
            # 업데이트되는 절편은 가중치는 ∂L/∂b를 계산하면 전체 오차합에 2를 곱한 뒤 평균을 구한 것과 같아짐
            # 이것 역시 y에서 y_hat을 빼는 연산으로 미분하면 -가 튀어나와 부호가 반대가 된다

            mse = (error ** 2).mean()
            # 매 epoch마다 MSE 계산 후 저장

            self.loss.append(mse)

        return self

    # 행렬 * 벡터 단위 계산
    def matcal(self, X) :
        return X @ self.w + self.b

    # 활성화 함수
    def activation(self, matcal_outcome) :
        return matcal_outcome

    # 클래스 분류 기준 및 클래스 레이블 제시
    def predict(self, X) :
        return np.where(self.activation(self.matcal(X)) >= 0.5, 1, 0)

X = np.array([[1, -1], [5, 7]])
y = np.array([0, 1])

# agd = AdalineGD(eta=0.05, n_iters=30)
# agd.fit(X, y)
# print(agd.predict(X))

import matplotlib.pyplot as plt
# fig, ax로 저수준 API 접근
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# 클래스에 인자를 담아 객체로 만든 뒤 바로 fit을 진행
data1 = AdalineGD(eta=0.1, n_iters=20).fit(X, y)
ax[0].plot(range(1, len(data1.loss) + 1), np.log10(data1.loss), marker='o')  # 선형 그래프의 x, y축 값 지정
ax[0].set_xlabel('Epochs')   # ax 객체에 일일이 접근해 속성 지정
ax[0].set_ylabel('log MSE')
ax[0].set_title('MSE by epochs with eta=0.1, n_iters=20')

data2 = AdalineGD(eta=0.001, n_iters=20).fit(X,y)
ax[1].plot(range(1, len(data2.loss) + 1), np.log10(data2.loss), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_title('MSE by epochs with eta=0.001, n_iters=20')
# set_ylabel 없이 진행해 축 레이블 공유

plt.show()
