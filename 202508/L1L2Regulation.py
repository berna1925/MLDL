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
fig = plt.figure(figsize=(7.5, 5))
ax = plt.subplot()

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black', 'pink',
          'lightgreen', 'lightblue', 'gray',
          'indigo', 'orange']

weights, params = [], []

import numpy as np
for c in range(-4, 6) :
    # C의 크기를 달리 함에 따라 회귀계수와 절편 값이 어떻게 변하는지 추적
    lr = LogisticRegression(penalty='l1', C=10 ** c,
                            solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_scaled, y_train)

    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

# 특성에 따라 값들이 분리될 수 있도록 여러 행의 값을 한 번에 y축에 투입
for column, color in zip(range(weights.shape[1]), colors) :
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1], color=color)

plt.xlabel('C(inverse regularization strength)')
plt.ylabel('weight coef')

plt.axhline(y=0, color='black', linestyle='--')
plt.xlim([10**(-5), 10**5])
plt.xscale('log')

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1), ncol=1, fancybox=True)
plt.tight_layout()
plt.show()

#===============================================================================

# 순차 특성 선택 알고리즘

