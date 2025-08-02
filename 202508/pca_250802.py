import pandas as pd
from networkx.algorithms.threshold import eigenvalues

df_wine = pd.read_csv('wine_data.csv')
X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

import numpy as np
# m*n 행렬에서 공분산을 구하려고 np.cov를 사용
# cov의 기본 작용은 열 벡터의 내적을 구하는 것이므로 n*n 꼴의 공분산 행렬을 만들려면 np.cov(M.T) 형태로 써야 한다
cov_mat = np.cov(X_train_scaled.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
# print(eigen_values)
# print()
# print(eigen_vectors)

from sklearn.decomposition import PCA
pca = PCA(n_components=5)

total = sum(eigen_values)
# 고유값의 합을 구한 다음 고유값 목록을 내림차순한 것을 순서대로 뽑아내 최고 주성분부터 새 공간에서 차지하는 비중을 구함
var_exp = [i / total for i in sorted(eigen_values, reverse=True)]
# 추이를 보기 위해 누적합을 np.cumsum으로 계산
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
# plt.bar(range(1, 14), var_exp, align='center', label='individual explained variance')
# cumsum과 궁합이 잘 맞는 스텝 플랏으로 영향력이 큰 축 순서대로 분산 값 시각화
# plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')

plt.xlabel('principal component index')
plt.ylabel('explained variance ratio')
plt.legend(loc='best')

# plt.tight_layout()
# plt.show()

eigen_pairs = [(np.abs(eigen_values[i]), np.abs(eigen_vectors[i]))
               for i in range(len(eigen_values))]
eigen_pairs.sort(key=lambda x : x[0], reverse=True)

# 첫 두 고유값에 해당하는 고유 벡터를 배열 형태로 추출
# 원래 (3, )인 shape을 축을 하나 더 만들어 공간을 한 차원 높이는 np,newaxis로 (3, 1)로 증강
w = np.hstack([eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]])

X_train_pca = X_train_scaled @ w

colors = ['r', 'g', 'b']
markers = ['x', 'o', 's']

for l, c, m in zip(np.unique(y_train), colors, markers) :
    plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1],
                c=c, label=f'class {l}', marker=m)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')

plt.tight_layout()
plt.show()