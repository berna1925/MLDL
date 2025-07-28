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

    # 학습 함수 fit 정의, 최종 연산에 사용할 원소 정의
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
                update = - self.eta * (self.predict(xi) - target)
                self.w += update * xi
                # w := w - η(y_hat - y)x
                self.b += update
                # b := b - η(y_hat - y)
                error += int(update != 0.0)
                # update가 0이 아니라면 1이 추가되도록 bool 형태를 사용

            self.error.append(error)
            # 미리 준비해 둔 깡통 리스트에 error를 반환해서 반출

        return self

    # 행렬-벡터 단위 연산 방식 선언
    def net_input(self, X) :
        return X @ self.w + self.b

    # 최종 연산 결과를 기준으로 어떠한 형태의 값을 사용자에게 반환할지 정의
    # 함수 이름은 예측이지만 실질적으로는 모델 예측 연산이 아닌 예측 값에 레이블링을 하는 것
    def predict(self, X) :
        return np.where(self.net_input(X) >= 0, 1, 0)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])

# 클래스 호출
# 인자 없는 깡통 클래스를 호출하면 self를 달라고 요구하니 실행이 안 된다!
pct = Perceptron(eta=0.01, n_iter=20)

# rfc.fit처럼 클래스를 인스턴스로 만들어 그 부속 함수인 fit으로 학습 진행
pct.fit(X,y)

# fit을 통해 진행한 연산이 net_input 연산을 거쳐 predict로 진행해 결과 반환
print(pct.predict(X))

# 선형대수 계산 예제
v1 = np.array([1, 2, 3])
v2 = np.array([3, 0, -2])
print(np.degrees(np.arccos(v1@v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))))
      # 라디안을 도로 # 코사인각 # 벡터 내적 # v1의 길이         # v2의 길이