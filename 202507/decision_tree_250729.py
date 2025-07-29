import numpy as np
import matplotlib.pyplot as plt


def entropy(p) :
    return - (p * np.log2(p) + (1-p) * np.log2(1-p))
# 이진 분류의 엔트로피 지수는 로지스틱 회귀 손실함수식과 동일하다
# (딥러닝의 binary_cross_entropy와 동일한 메커니즘)

x = np.arange(0, 1, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
# p = 0에 대해서만 예외 처리
# plt.plot(x, ent)

# plt.xlabel('class-membership probs')
# plt.ylabel('entropy')
# plt.show()

# 지니 계수
def gini(p) :
    return p * (1-p) + (1-p) * (1 - (1-p))
# p(1-p) 곱하고 p에 (1-p) 대입해서 곱한 것을 더하고
# 간략히 2p(1-p)로 쓰곤 합니다

# 잘못 예측했을 때 줄 페널티를 에러로 규정
# 최대 0.5
def error(p) :
    return 1 - np.max([p, 1-p])

scaled_ent = [p * 0.5 if p else None for p in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot()


# for i, label, ls, c in zip([ent, scaled_ent, gini(x), err],
#                            ['Entropy', 'Entropy(scaled)', 'Gini impurity', 'Misclassification error'],
#                            ['-', '-', '--', '-.'],
#                            ['black', 'lightgray', 'red', 'green', 'cyan']) :

# line = ax.plot(x, i, label=label, linestyle=ls, lw=2, color=c)
# (0, 1) 구간에서 엔트로피 지수, 지니 지수, 분류 오차 지수를 비교

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, shadow=False)
# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1, linewidth=1, color='k', linestyle='--')

# plt.ylim([0, 1.1])
# plt.xlabel('prob')
# plt.ylabel('impurity index')
# plt.show()

import pandas as pd
# iris 데이터 다운로드
df = pd.read_excel('iris_data.xlsx', header=None)

import matplotlib.pyplot as plt
# df에서 원하는 자료를 추출해 np.ndarray로 변형
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)  # setosa면 0, 다른 종은 1로 변환
X = df.iloc[:, [0, 2]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# sc.fit(X_train)... X_train_scaled = sc.transform(X_train)으로 분리해서 사용도 가능
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5)
dtc.fit(X_train_scaled, y_train)

X_combined = np.vstack([X_train, X_test])
y_combined = np.hstack([y_train, y_test])

from iris_perceptron_250723 import plot_decision_regions
# plot_decision_regions(X_combined, y_combined, model=dtc)
# plt.xlim([-5, 5])
# plt.show()

from sklearn import tree
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

# sklearn의 tree 모듈에서 feature_names에 분류 기준 인자를 설정해 나무 모형 시각화
# tree.plot_tree(dtc, feature_names=feature_names, filled=True)
# plt.show()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0, max_depth=3, n_estimators=20, ccp_alpha=0.01)
rfc.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, rfc)
plt.tight_layout()
plt.show()
