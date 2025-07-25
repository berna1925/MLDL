import numpy as np

# 확률적 경사 하강법 모델의 구성
class AdalineSGD() :
    def __init__(self, eta=0.01, n_iters=30, random_state=None, shuffle=True) :
        # 학습 모델의 학습률, 학습 회수, 랜덤 스테이트 초기화
        self.eta = eta
        self.n_iters = n_iters
        # SGD는 배치 단위 계산의 연산량 부담을 줄이기 위해 한 샘플 단위로 계산해 가중치를 업뎃하게 바꾼 구조
        # 랜덤하게 샘플들을 뽑아 한 줄씩 계산해 과적합을 막으므로 랜덤성 규제를 random_state=None으로 해제
        self.random_state = random_state
        # 매 에폭마다 샘플들의 순서가 바뀌어 랜덤하게 데이터를 선택할 수 있도록 shuffle=True를 사전에 부여
        self.shuffle = shuffle

        # 부분 학습을 별도로 진행하는 경우를 대비해 초기 가중치 작업이 아직 안 됐다는 것을 알려줌
        self.w_init = False

    # 가중치 초기화 함수 제작
    def init_w(self, m) :
        # 랜덤하게 변수가 정해질 수 있도록 default_rng로 랜덤성 부여
        self.v = np.random.default_rng(self.random_state)
        # 가중치 초기화 벡터 구현
        self.w = self.v.normal(0, 0.1, size=m)
        # 초기 절편 값 부여
        self.b = np.float32(0)
        # 가중치 초기화 작업을 완료했음을 선언
        self.w_init = True

    # 데이터 순서에 랜덤성 부여
    def shuffle(self, X, y) :
        # np.random.permutation으로 인덱스에 랜덤성 부여
        random_order = self.v.permutation(X.shape[0])

        # X와 r이 매번 랜덤한 순서로 데이터를 가져오도록 명령
        return X[random_order], y[random_order]

    # 학습 모델 구성
    def fit(self, X, y) :
        # init_w 함수를 호출해 초기 가중치가 담긴 벡터와 절편을 소환
        self.init_w(X.shape[1])
        # 총 MSE를 담아놓을 그릇을 미리 마련
        self.loss = []

        # 주어진 n_iters만큼 epoch을 진행
        for i in range(self.n_iters) :
            # 사용자가 특수한 목적으로 shuffle=False를 하지 않는 이상 데이터가 랜덤하게 선별되도록 유도
            if self.shuffle :
                # 랜덤한 순서로 데이터 셔플
                X, y = self.shuffle(X, y)

            # 한 에폭 내에서 발생할 샘플들의 오차 제곱항을 담아놓을 저장소 리스트 확보
            losses = []

            # 랜덤히 선택된 독립변수-종속변수 쌍들에 대해
            for xi, target in zip(X, y) :
                # 계산 결과 오차 제곱항을 전달
                losses.append(self.update_weights(xi, target))

            # 모든 데이터에 대한 계산이 한 바퀴 끝났으면 한 에폭의 평균오차제곱(MSE)을 저장
            avg_loss = np.mean(losses)

            # 각 에폭의 MSE를 self.loss에 저장해 비교에 활용
            self.loss.append(avg_loss)

        return self

    # fit 학습에 따른 가중치 업데이트 방식 규정
    def update_weights(self, xi, target):
        # 1줄 단위로 계산이 진행돼 prediction, target, error가 매번 값이 바뀌게 됨
        # 행렬x벡터 계산과 활성화함수를 거친 예측치가 output임을 선언
        # self.matcal 함수를 받아 가중치 업데이트가 행렬*벡터 연산 바로 뒤에 이어지도록 설계
        output = self.activation(self.matcal(xi))
        # error는 타깃 - 예측치
        error = target - output

        # 가중치 업데이트 방식(경사하강법의 고전적인 공식 활용)
        self.w += self.eta * 2 * error * xi
        self.b += self.eta * 2 * error

        # MSE 계산을 위해 오차 제곱을 반환
        loss = error ** 2

        return loss

    # 행렬*벡터 간 연산 방식 규정
    def matcal(self, X) :
        return X @ self.w + self.b

    # 연산 결과가 통과할 활성화 함수 규정
    def activation(self, X) :
        return X

    # 부분 학습이 필요할 때 사용할 함수 제작
    def partial_fit(self, X, y) :
        # 가중치 초기화가 안 된 상태라면 self.init_w로 함수로 가중치 초기화 진행
        if not self.w_init :
            self.init_w(X.shape[1])

        # 이미 가중치 초기화 작업이 끝났다면 학습 진행
        if y.ravel().shape[0] > 1 :
            for xi, target in zip(X, y) :
                self.update_weights(xi, target)
        # 만약 데이터가 1개밖에 없다면 데이터를 통째로 넣고 학습
        else :
            self.update_weights(X, y)

    # 학습 모델의 예측 결과 반환
    def predict(self, X) :
        return np.where(self.activation(self.matcal(X)) >= 0.5, 1, 0)
