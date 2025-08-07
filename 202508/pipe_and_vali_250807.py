import pandas as pd
df = pd.read_csv('breast_cancer.csv')

# 전처리기 호출 등을 통한 파이프라인 구축 실습

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

y = le.fit_transform(y)
# print(le.classes_)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

pl = Pipeline([
    # make_pipeline()을 쓰면 대괄호와 별칭 기입 없이 활용 가능
    ('std scaling', StandardScaler()),
    ('pca decomp', PCA(n_components=2)),
    ('lr', LogisticRegression(C=1, random_state=0))
])

pl.fit(X_train, y_train)
y_pred = pl.predict(X_test)

#======================================================================================================

# 각종 분할 검증 방법을 통한 과적합 견제 도구 사용 실습

# print(pl.score(X_test, y_test))
import numpy as np
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True).split(X_train, y_train)
scores = []

# StratifiedKFold는 단독으로 쓰일 땐 인덱스 묶음을 반환한다!
# 그래서 매 회차에 다른 분할 기준을 적용한다는 의미로 X_train과 y_train에 인덱싱을 적용해 슬라이싱하는 것
for k, (train_idx, test_idx) in enumerate(kfold) :
    pl.fit(X_train[train_idx], y_train[train_idx])
    score = pl.score(X_train[test_idx], y_train[test_idx])
    # 매 분할 기준에 따른 훈련용 데이터의 유효성을 검증
    scores.append(score)

    print(f'{k+1 : 2d}th fold : accuracy = {score : .4f}')

print()
print(f'{np.mean(scores) : .3f}')
print()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# StratifiedKFold를 단독으로 쓸 때보다 성능을 압축적으로 확인 가능
# cross_val_score(학습_모델명, 독립변수, 종속변수, cv)
cvs = cross_val_score(pl, X_train, y_train, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0))
print(cvs)
print()

# cross_val_score와 달리 scoring 기준을 precision으로도 바꿀 수 있는 cross_validate
cv = cross_validate(pl, X_train, y_train, scoring='precision', cv=StratifiedKFold(n_splits=10, shuffle=True))
print(cv['test_score'])
print()

from sklearn.model_selection import cross_val_predict
cvp = cross_val_predict(pl, X_train, y_train,
                        method='predict_proba', cv=StratifiedKFold(n_splits=10, shuffle=True))

print(cvp[:5])
print()

#==========================================================================================================

