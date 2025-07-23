import numpy as np

# 학습률과 가중치, 입력 값 등을 놓고 연산하는 퍼셉트론 모델 구현
class Perceptron() :
    # 사전 설정 함수 __init__ 정의
    def __init__(self, eta=0.01, n_iter=50, random_state=0) :
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    # __init__ 함수에서 클래스로 객체 호출 시 학습률과 학습 회수를 초기화하도록 설정
    # numpy의 랜덤성을 잡기 위해 random_state까지 사전 설정

    # 학습 함수 fit 정의
    def fit(self, X, y) :
        inputs = np.random.default_rng(self.random_state)
        self.w = inputs.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = float(0)
        self.error = []

        # 가중치 벡터의 사이즈를 1차원적으로 설정하면 np.dot으로 순서 상관없이 내적 계산 가능
        # np.random.default_rng로 특정 변수만 랜덤성 제어
        # 초기 가중치를 평균 0, 표준편차 0.01인 입력변수의 특성 개수만큼의 차원으로 초기화

        for _ in range(self.n_iter) :
            error = 0
            # error 초기화

            for xi, target in zip(X, y) :
                update = self.eta * (target - self.predict(xi))
                self.w += update * xi
                self.b += update
                error += int(update != 0.0)
            self.error.append(error)
        return self

    def net_input(self, X) :
        return X @ self.w + self.b

    def predict(self, X) :
        return np.where(self.net_input(X) >= 0, 1, 0)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])

pct = Perceptron(eta=0.01, n_iter=20)
pct.fit(X,y)

print(pct.predict(X))