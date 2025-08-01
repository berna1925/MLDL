# L1 규제의 활용
import pandas as pd
# df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# df_wine.to_csv('wine_data.csv', index=False)
df_wine = pd.read_csv('wine_data.csv')
df_wine.columns = ['label', 'alcohol', 'malic acid', 'ash',
                   'alcalinity of ash', 'magnesium','total phenol',
                   'flavanoid', 'nonflavanoid phenol',
                   'proanthocyanin', 'color intensity',
                   'hue', 'diluted', 'proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0, stratify=y)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', penalty='l1', C=1, multi_class='ovr')
lr.fit(X_train_scaled, y_train)

# print(lr.score(X_train_scaled, y_train))
# print(lr.score(X_test_scaled, y_test))

# print(lr.intercept_,'\n',lr.coef_)

import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(7.5, 5))
# ax = plt.subplot()
#
# colors = ['blue', 'green', 'red', 'cyan',
#           'magenta', 'yellow', 'black', 'pink',
#           'lightgreen', 'lightblue', 'gray',
#           'indigo', 'orange']
#
# weights, params = [], []

import numpy as np
# for c in range(-4, 6) :
# C의 크기를 달리 함에 따라 회귀계수와 절편 값이 어떻게 변하는지 추적
#     lr = LogisticRegression(penalty='l1', C=10 ** c,
#                             solver='liblinear', multi_class='ovr', random_state=0)
#     lr.fit(X_train_scaled, y_train)
#
#     weights.append(lr.coef_[1])
#     params.append(10**c)
#
# weights = np.array(weights)

# 특성에 따라 값들이 분리될 수 있도록 여러 행의 값을 한 번에 y축에 투입
# for column, color in zip(range(weights.shape[1]), colors) :
#     plt.plot(params, weights[:, column],
#              label=df_wine.columns[column + 1], color=color)
#
# plt.xlabel('C(inverse regularization strength)')
# plt.ylabel('weight coef')
#
# plt.axhline(y=0, color='black', linestyle='--')
# plt.xlim([10**(-5), 10**5])
# plt.xscale('log')
#
# ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1), ncol=1, fancybox=True)
# plt.tight_layout()
# plt.show()

#===============================================================================

# 순차 특성 선택 알고리즘
## 특성 공간을 부분적으로 축소하는 그리디 서치 알고리즘

from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import *

class SequentialBackwardSelection() :
    def __init__(self, estimator, n_features,
                 scoring=accuracy_score, test_size=0.2, random_state=0) :
        # 학습 모델의 초매개변수 정보만 가져오고 학습 내용을 초기화하는 clone 메서드
        self.estimator = clone(estimator)
        # 사용자가 임의로 고르는 숫자만큼 특성을 선택하도록 설정
        self.n_features = n_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y) :
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        # x의 특성 개수가 곧 학습 차원(일반적으로는 그러하다)
        dim = X_train.shape[1]
        # 초기 self.index 값은 (0, 1, ... , n-1)
        self.index_ = tuple(range(dim))
        # 초기 self.subset 값은 전체 차원이 들어간 인덱스 튜플
        self.subset_ = [self.index_]

        # self.index를 바탕으로 추후에 추가되는 튜플 속 인덱스 조합에 따라
        # 그 순서에 맞는 특성만 선택해 모델을 학습하고 점수로 평가하는 구조
        # 처음에는 독립변수 전체 특성을 골라 계산하고 다음에는 평가 점수에 가장 큰 기여를 한 특성만 선택에 투입
        score = self.calc_score(X_train, X_test, y_train, y_test, self.index_)
        self.scores = [score]

        # 데이터 차원을 조금씩 줄여가면서 사용자가 지정한 차원이 될 때까지 연산
        while dim > self.n_features :
            scores = []
            subsets = []

            # combinations(iterator, r)은 iterator 내부 원소 n개 중 r개를 뽑아 반환
            # nCn-1 꼴이므로 기존 index_ 개체에서 원소를 한 개씩 뺀 모든 원소 조합을 대상으로 반복문 실행
            # 원소 조합 역시 인덱스의 조합이므로 학습과 평가에 유동적으로 활용 가능
            for p in combinations(self.index_, r=dim-1) :
                # 원래 인덱스 집합보다 크기가 1 작은 부분집합들의 평가 지표를 돌아가면서 계산
                score = self.calc_score(X_train, X_test, y_train, y_test, p)
                scores.append(score)
                subsets.append(p)

            # 가장 점수가 높은 부분집합의 인덱스를 반환
            best = np.argmax(scores)
            # 최고점을 찍은 부분집합의 인덱스 반환
            self.index_ = subsets[best]
            # self.subset_에 각 단계별 최고점을 찍은 부분집합의 인덱스를 저장
            self.subset_.append(self.index_)

            # 차원을 하나 줄여서 while문으로 다시 돌아가 크기가 줄어든 부분집합 대상으로
            # 다시 부분집합으로 분해해 최고점을 찍을 인덱스 조합을 검색
            dim -= 1
            self.scores.append(scores[best])
            self.k_score = self.scores[-1]

        return self

    # fit에서 학습을 완료하면 사용자가 transform 함수를 호출했을 때
    # 다른 데이터(=테스트용)를 대상으로도 같은 인덱스를 가진 데이터를 추출하도록 명령
    def transform(self, X) :
        return X[:, self.index_]

    # 데이터 파츠를 받아 영향력이 큰 칼럼 인덱스만 추출해 fit, predict 진행
    # 이후 메트릭스 기반 평가까지 진행
    def calc_score(self, X_train, X_test, y_train, y_test, index) :
        self.estimator.fit(X_train[:, index], y_train)
        prediction = self.estimator.predict(X_test[:, index])
        score = self.scoring(y_test, prediction)

        return score

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5)
# 원하는 모델을 호출한 뒤 estimator 인자에 투입
sbs = SequentialBackwardSelection(knc, n_features=3)
sbs.fit(X_train_scaled, y_train)
# subset_에 저장된 리스트 속 개별 부분집합 튜플의 개수를 세어 리스트로 나열(n, n-1, n-2, ...)
k_features = [len(k) for k in sbs.subset_]

plt.plot(k_features, sbs.scores, marker='o')
plt.ylim([0.7, 1.1])

plt.xlabel('# of features')
plt.ylabel('accuracy')
plt.grid(True)
plt.tight_layout()

plt.show()