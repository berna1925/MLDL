import pandas as pd
df_wine = pd.read_csv('wine_data.csv')
df_wine.columns = ['label', 'alcohol', 'malic acid', 'ash',
                   'alcalinity of ash', 'magnesium','total phenol',
                   'flavanoid', 'nonflavanoid phenol',
                   'proanthocyanin', 'color intensity',
                   'hue', 'diluted', 'proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
features = df_wine.columns[1:]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0, n_estimators=300)
rfc.fit(X_train, y_train)

# fit을 하면 확인할 수 있는 독립변수별 영향력 feature_importances_
# print(rfc.feature_importances_)

import numpy as np
weights = rfc.feature_importances_
# iterator를 오름차순으로 정렬해 원소 순서대로 그대로 인덱스를 반환하는 np.argsort
# [::-1] 인덱싱으로 내림차순 정렬로 변경해 가장 중요한 변수부터 오도록 재조정
indices = np.argsort(weights)[::-1]

# for f in range(X_train.shape[1]) :
    # 0부터 오는 반복문의 특성을 고려해 미리 내림차순으로 정렬한 인덱스 순서로 중요도 전개
    # print(f'{f+1:2d}) {features[indices[f]]:<30} {weights[indices[f]]:.3f}')

import matplotlib.pyplot as plt
plt.title('Feature Importances')

plt.bar(range(X_train.shape[1]), weights[indices])
plt.xticks(ticks=range(X_train.shape[1]), labels=features[indices], rotation=90)
# plt.bar에서 x축에 숫자를 달아놓고 plt.xticks에서 그 범위를 ticks 인자에 전달한 뒤
# labels에 레이블명을 매칭하는 방식으로 마킹

plt.xlabel('features')
plt.ylabel('importances')

# plt.tight_layout()
# plt.show()

# feature importance가 낮은 변수를 걸러주는 필터인 SelectFromModel
# 모든 변수의 중요도를 구한 다음 기준점을 넘기는 것만 쳐내기
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(rfc, threshold=0.1, prefit=True)
# 이미 학습을 완료한 모델을 인자로 받아 자료 내용만 바꾸는 것이므로 fit_transform이 아닌 transform 사용
# 만약 rfc가 fit이 되지 않은 상태라면 fit_transform을 써야 함
X_selected = sfm.transform(X_train)

for f in range(X_selected.shape[1]) :
    print(f'{f + 1:2d}) {features[indices[f]]:<30} {weights[indices[f]]:.3f}')
print()

# feature importance가 가장 낮은 변수를 하나씩 단계적으로 제거하는 RFE
# 사용자가 지정한 n_feature_to_select 인자 수에 이를 때까지 1개씩 삭제 작업 반복
# 매번 학습을 반복하는 매커니즘이 있어 머신러닝 모델들처럼 인자와 함께 호출하고 fit 사용이 필수
from sklearn.feature_selection import RFE
rfe = RFE(rfc, n_features_to_select=5)
rfe.fit(X_train, y_train)

# 변수들의 중요도 순위 반환
print(rfe.ranking_)
print()

importances = rfe.estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

# for i in range(len(importances)) :
#     print(f'{i+1}) {features[indices[i]]:<30} {importances[indices[i]]:.3f}')

print(rfe.estimator_)