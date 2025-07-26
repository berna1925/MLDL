# 라이트하게, 주어진 sklearn API로 시작
from sklearn import datasets
import numpy as np
# datasets.load~ 로 원하는 데이터셋 호출
iris = datasets.load_iris()

# data, target 속성으로 독립/종속변수를 객체에 저장
X = iris.data[:, [2, 3]]
y = iris.target

# 레이블 종류 파악
# print(np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# np.bincount로 레이블링된 종속변수의 클래스별 개수 확인
# print(np.bincount(y), np.bincount(y_train), np.bincount(y_test))

# 표준화 진행
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# sc.fit(X_train)... X_train_scaled = sc.transform(X_train)으로 분리해서 사용도 가능
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# linear_model 산하 Perceptron 모델 호출
from sklearn.linear_model import Perceptron
pct = Perceptron(eta0=0.1, random_state=0)

# 모델 학습 후 예측 실시
pct.fit(X_train_scaled, y_train)
y_pred = pct.predict(X_test_scaled)

# 실제 레이블과 예측 레이블에 차이가 있는 데이터의 개수 확인
# print((y_test != y_pred).sum())

# 평가 모델 호출
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
# print(f'accuracy of this model : {accuracy : .2f}')

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# 퍼셉트론 모델의 판단 결과를 등고선에 반영
def plot_decision_regions(X, y, model, test_idx=None, resolution=0.01) :
    markers = ['o', 'x', '*', 's', '^']
    colors = ['blue', 'red', 'gray', 'green', 'magenta']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() , X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.arange(x1_min-1, x1_max+1, resolution),
                           np.arange(x2_min-1, x2_max+1, resolution))

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




    # if test_idx :
    #     X_test, y_test = X[test_idx, :], y[test_idx]
    #
    #     plt.scatter(X_test[:, 0], X_test[:, 1],
    #                 c='none', edgecolors='black', alpha=0.1,
    #                 linewidths=1, marker='o',
    #                 s=100, label='Test set')
    #
    plt.show()
    #

# print(plot_decision_regions(X_test_scaled, y_pred, pct, y_pred))

X_combined_std = np.vstack([X_train_scaled, X_test_scaled])
y_combined = np.hstack([y_train, y_test])

#==================================================================================

# 로지스틱 회귀와 시그모이드 함수
def sigmoid(z) :
    # 시그모이드 함수 정의에 따라 함수 규정
    return 1 / (1 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)
plt.plot(z, sigma_z)

# 그림에 수직 방향 보조선 긋기
plt.axvline(0, color='k')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.yticks([0, 0.5, 1])

ax = plt.gca()
ax.yaxis.grid(True)
# 전체 그래프에 눈금을 전부 그리려면 plt.grid(True)로 하면 그만
# 특정 축에만 접근하고 싶다면 ax 단위로 접근해야 함
# ax로 x축, y축 단위로 접근한 뒤 yaxis로 y축을 선택해 grid 속성에 True를 부여해 y축만 눈금 활성화
# plt.grid(True, axis='y')로 써도 OK

plt.tight_layout()
# plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_scaled, y_train)

plot_decision_regions(X_combined_std, y_combined, model=lr)