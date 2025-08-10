from nbformat.sign import yield_everything
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values
y = y.astype('int').values

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10) :
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img)

# set_xticks([])로 하면 눈금이 아예 사라지는 마술
ax[0].set_yticks([])
ax[0].set_xticks([])
plt.tight_layout()

from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=0, stratify=y_temp)

from neuralnet import NeuralNetMLP
import numpy as np

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def int_to_onehot(y, num_labels) :
    array = np.zeros([y.shape[0], num_labels])

    for i, val in enumerate(y) :
        array[i, val] = 1

    return array

class NeuralNetMLP() :
    def __init__(self, num_features, num_hidden, num_classes, random_seed=0) :
        # num_hidden : 은닉층 노드 개수(keras의 Dense 속에 들어가는 사용자 정의 숫자)
        super().__init__()

        self.num_classes = num_classes

        rng = np.random.default_rng(random_seed)

        # 가중치 행렬 초기화
        # 특성 개수만큼의 열을 가진 은닉층 뉴런(노드) 개수의 행 벡터를 준비해 원래 데이터 행렬과의 곱셈 준비
        self.weight = rng.normal(loc=0, scale=0.1, size=(num_hidden, num_features))
        self.bias = np.zeros(num_hidden)

        self.weight_out = rng.normal(loc=0, scale=0.1, size=(num_hidden, num_classes))
        self.bias_out = np.zeros(num_classes)

    # 다층 신경망 모델의 순전파 과정 = [은닉층 + 출력층] 구성
    def forward(self, x) :
        # 행렬 속 데이터와 가중치 행렬을 곱하기 위해 전치를 해서 모양을 맞추고
        # 곱셈 진행 후 bias 더하기로 결정 함수 값을 반환 → 은닉층 각 뉴런의 가중합 계산
        z_h = x @ self.weight.T + self.bias
        # 시그모이드 함수로 비선형 변환 진행
        # 중간 특징 벡터 형태로 은닉층 뉴런 출력
        a_h = sigmoid(z_h)

        # 은닉층의 연산 결과를 받아 다시 이후 계산에서 활용하기 쉬운 형태로 반환하는 출력층
        # 데이터별로 단일한 최종 확률 값을 반환하기 위해 shape이 m, 1이 되도록
        # 초기화해 둔 weight_out을 곱해주고 편향 더하기
        z_out = a_h @ self.weight_out.T + self.bias_out
        # 한 은닉층+출력층 과정의 최종 확률 반환 값(predict_proba) 반환
        a_out = sigmoid(z_out)

        return a_h, a_out

    # 역전파 과정
    def backward(self, x, y, a_h, a_out) :
        # 종속변수를 원핫 인코딩
        y_oh = int_to_onehot(y, self.num_classes)

        # 출력층에서 각 데이터 별로 손실 값 구하는 과정 도입(MSE 기준) → ∂L/∂w
        # L = (1/m)[Σ(y_oh - a_out)²] 이므로
        # ∂L/∂y = -2*(y_oh - a_out)/m
        d_loss_d_a_out = -2 * (y_oh - a_out) / y.shape[0]
        # ∂y/∂z = a_out(1 - a_out)
        d_a_out_d_z_out = a_out * (1 - a_out)
        # ∂z/∂w = a_h
        d_z_out_dw_out = a_h

        # (∂L/∂y)(∂y/∂z) = ∂L/∂z = 결정 함수 값에 따른 손실 변화 → 출력층 오차
        delta_out = d_loss_d_a_out * d_a_out_d_z_out
        # 기울기 갱신 ∂L/∂w = (∂L/∂y)(∂y/∂z)(∂z/∂w) = (delta_out)(a_h)
        d_loss_dw_out = delta_out.T @ d_z_out_dw_out
        # 출력층 오차 합 = delta_out의 합계
        d_loss_db_out = np.sum(delta_out, axis=1)

        # 다음 은닉층으로 전달되는 오차 w = ∂z/∂w
        d_z_out_a_h = self.weight_out
        # 출력층 오차 행렬에 은닉층 오차 곱해 기울기 갱신 요소 생성
        d_loss_a_h = delta_out @ d_z_out_a_h
        # 은닉층 활성화 함수 시그모이드 미분
        d_a_h_d_z_h = a_h * (1 - a_h)
        # 은닉층 가중치의 기울기
        d_z_h_d_w_h = x
        # 다시 ∂L/∂w 값 구하기
        d_loss_d_w_h = (d_loss_a_h * d_a_h_d_z_h).T @ d_z_h_d_w_h
        # 은닉층 편향의 기울기
        d_loss_d_b_h = np.sum(d_loss_a_h * d_a_h_d_z_h, axis=0)

        return (d_loss_dw_out, d_loss_db_out, d_loss_d_w_h, d_loss_d_b_h)
