# Linear Discriminant Analysis
# 선형 분할이 가능한 데이터 그룹이 있을 때 그룹 내 분산은 최소화, 그룹 간 분산은 최대화 목표
import pandas as pd

df_wine = pd.read_csv('wine_data.csv')
X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]

df_wine.columns = ['label', 'alcohol', 'malic acid', 'ash',
                   'alcalinity of ash', 'magnesium','total phenol',
                   'flavanoid', 'nonflavanoid phenol',
                   'proanthocyanin', 'color intensity',
                   'hue', 'diluted', 'proline']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

#===============================================================================
# 클래스 내부 분산 계산
import numpy as np
np.set_printoptions(precision=3)
mean_vectors = []

for label in range(1, 4) :
    # 레이블이 같은 행 벡터들을 모아 인덱스가 같은 원소들의 평균을 구해 평균 벡터 계산
    # 클래스별 평균 벡터를 차례로 mean_vectors에 저장
    mean_vectors.append(np.mean(X_train_scaled[y_train == label], axis=0))

d = X_train_scaled.shape[1]
sw = np.zeros((d, d))

# 클래스별로 데이터 수가 같은 경우
# 스케일 조정 없이 레이블 기준으로 묶어 원래 행 벡터에서 평균 벡터를 빼는 방식으로 산포 정도 적립
for label, mv in zip(range(1, 4), mean_vectors) :
    class_scatter = np.zeros((d, d))

    for row in X_train_scaled[y_train == label] :
        # 각 레이블을 기준으로 묶은 뒤 실제 데이터 벡터에서 평균 벡터를 빼는 과정이 MSE 구하는 과정(y-y_hat)과 닮아 있다
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        # d*d 꼴의 개별 산포 행렬을 만들기 위해 원 데이터와 평균 벡터의 shape을 d,1로 바꾸고 전치와 곱하기
        # 행렬 벡터 계산에서 x @ x^T 꼴의 계산은 외적을 구하는 것과 같으며 이는 데이터 분포의 형상을 부피감 있게 나타내기 위한 것
        # row와 mv가 (d, 1) 벡터가 됐지만 차이 벡터의 외적을 구하면 의미 있는 벡터가 차이 벡터 1개뿐이므로 rank-1에 해당
        class_scatter += (row - mv) @ (row - mv).T

    # 그룹 내부 데이터 분산도 누적
    sw += class_scatter

# 클래스별로 데이터 수가 다른 경우
# 산포 정도가 쌓이는 수준이 클래스마다 달라질 것이므로 데이터 수로 산포 누적도를 나누는 것이 타당
# 산포도를 클래스 데이터 수인 n_i로 나누면 이는 공분산 행렬과 논리적으로 같아짐
for label, mv in zip(range(1, 4), mean_vectors) :
    class_scatter = np.cov(X_train_scaled[y_train == label].T)
    sw += class_scatter

#===================================================================================================
# 클래스 간 분산도 계산
# 전체 평균 벡터 값 저장
mean_overall = np.mean(X_train_scaled, axis=0).reshape(d, 1)
sb = np.zeros((d, d))

# 클래스별로 돌며 차이 벡터 계산
for label, mv in zip(range(1, 4), mean_vectors):
    # 클래스별로 데이터 수 계산
    n = X_train_scaled[y_train == label, :].shape[0]
    mv = mv.reshape(d, 1)
    # Sw를 구할 땐 샘플 속 개별 단위 벡터에서 클래스 평균을 빼고
    # Sb를 구할 땐 클래스 그룹 평균을 개별 단위로 보고 그룹별 평균에서 전체 평균을 빼는 구조
    # 클래스별 샘플 크기를 분산 반영 정도에 가중치로 실어 클래스별 영향력 차이를 강조
    sb += n * (mv - mean_overall) @ (mv - mean_overall).T

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(sw) @ sb)

eigen_pairs = [(np.abs(eigen_vals[i]), np.abs(eigen_vecs[i])) for i in range(len(eigen_vecs))]
eigen_pairs = sorted(eigen_pairs, key=lambda x : x[0], reverse=True)

# for eigen_val in eigen_pairs :
#     print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i /tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

import matplotlib.pyplot as plt
plt.bar(range(1, 14), discr, label='individual')
plt.step(range(1, 14), cum_discr, where='mid', label='cumulative')

plt.xlabel('discriminants')
plt.ylabel('discriminability')

plt.ylim([0, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# 고유값이 가장 큰 두 개의 변수로 모델 설명 가능

w = np.hstack([eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real])
# print(w)

#==========================================================================================
# 새 공간으로 샘플 투영
X_train_lda = X_train_scaled @ w

colors = ['r', 'g', 'b']
markers = ['x', 'o', 's']

for l, c, m in zip(np.unique(y_train), colors, markers) :
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1],
                c=c, label=f'class {l}', marker=m)
    # 고유값 2개로 분산을 100% 설명한 것으로 나와 2차원 공간에 각 변수의 분포를 재현 가능

plt.xlabel('LD #1')
plt.ylabel('LD #2')

plt.legend(loc='best')
plt.tight_layout()
plt.show()

#===============================================================================================
# sklearn에 구현돼 있는 LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train_scaled, y_train)

from matplotlib.colors import ListedColormap
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

from sklearn.linear_model import LogisticRegression as LR
lr = LR(C=1, random_state=0, multi_class='ovr', solver='lbfgs')
lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, model=lr)
plt.show()