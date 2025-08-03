import pandas as pd
from networkx.algorithms.threshold import eigenvalues

df_wine = pd.read_csv('wine_data.csv')
X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]

df_wine.columns = ['label', 'alcohol', 'malic acid', 'ash',
                   'alcalinity of ash', 'magnesium','total phenol',
                   'flavanoid', 'nonflavanoid phenol',
                   'proanthocyanin', 'color intensity',
                   'hue', 'diluted', 'proline']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, model, test_idx=None, resolution=0.01) :
    markers = ['o', 's', '^', 'v', '<']
    colors = ['red', 'green', 'blue', 'gray', 'cyan']
    # ListedColormap으로 사용자 정의 팔레트를 미리 구성
    # 레이블의 가지 수만큼만 쓰도록 len(np.unique)) 코드 사용
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # x축, y축으로 쓸 값의 최저~최대값을 미리 추출
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # np.meshgrid로 최소치와 최대치로 범위를 지정해 놓은 뒤 계산 간격을 미리 정해둔 resolution으로 규정
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # np.meshgrid([x축으로 쓸 배열 값], [y축으로 쓸 배열 값])

    # predict는 m*k shape 행렬을 받아야 하므로 ravel로 펼쳐 놓은 두 독립변수는 다시 전치해 m행으로 맞춰줘야
    # T를 쓰지 않으면 ravel이 걸려 m열이 된 상태가 되니 계산이 안 됨
    label = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    label = label.reshape(xx1.shape)

    plt.contourf(xx1, xx2, label, alpha=0.2, cmap=cmap)

    plt.xlim([xx1.min(), xx1.max()])
    plt.ylim([xx2.min(), xx2.max()])

    for idx, l in enumerate(np.unique(y)) :
        plt.scatter(x=X[y == l, 0], y=X[y == l, 1],
                    alpha=0.5, c=colors[idx], marker=markers[idx],
                    label=f'class {l}',
                    edgecolors='black')

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1, random_state=0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_test_pca, y_test, model=lr)
plt.xlabel('Principal Component #1')
plt.ylabel('Principal Component #2')

plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# PCA(n_components=None)은 참고로 아무 요인도 배제하지 않고 그냥 계산하겠다는 뜻
# n_components에 (0, 1) 범위의 실수를 넣으면 그만큼의 누적 분산 비율을 가질 때까지만 요인 배열을 보여줌
pca_ = PCA(n_components=0.9)
X_train_pca = pca_.fit_transform(X_train_scaled)
print(pca_.explained_variance_ratio_)
# cumsum을 쓴 뒤 plt.step으로 그 값을 시각화하면 프레젠테이션에 활용 가능

cov_mat = np.cov(X_train_scaled.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# 선형대수적 개념에 의하면 데이터의 부분 열 공간은 eig_vecs * eig_vals
# 그러나 eig_vals는 열 벡터 값 분산의 집합이므로 열 특성에 따라 정규화한 뒤에도 분산 차이가 클 수 있음

# 고유값 eig_vals는 각 변수의 분산으로 이걸 제곱근 취하면 각 변수의 표준편차가 됨
# 표준편차는 표준기저직교벡터와 고유값의 곱으로 만들어질 부분 열 공간 QΛ를 QΛ^½으로 스케일링한 것
#
# chunks = eig_vecs * np.sqrt(eig_vals)
modified_chunks = pca_.components_.T * np.sqrt(pca_.explained_variance_)

fig, ax = plt.subplots()
ax.bar(range(13), modified_chunks[:, 0])

ax.set_ylabel('PC#1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)

plt.ylim([-1, 1])
plt.tight_layout()
plt.show()