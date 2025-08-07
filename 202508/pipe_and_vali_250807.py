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

# 검증 과정의 시각화 1 - 학습 곡선 learning_curve

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
pl = Pipeline([
    ('ss', StandardScaler()),
    ('logisticregression', LogisticRegression())
])

train_sizes, train_scores, test_scores = learning_curve(pl, X_train, y_train,
                                                        cv=StratifiedKFold(n_splits=10),
                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                        n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# plt.plot(train_sizes, train_mean, marker='o', color='royalblue', label='training data accuracy')
# plt.plot(train_sizes, test_mean, marker='s', color='green', label='validating data accuracy')
#
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.2, color='blue')
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.2, color='green')

plt.xlabel('train_sizes')
plt.ylabel('accuracy for data type')
plt.legend(loc='best')
plt.ylim([0.85, 1.05])

# plt.tight_layout()
# plt.show()


# 하이퍼파라미터 범위 단위를 x축으로 삼아 하이퍼파라미터 변화에 따른 검증 수치 변화를 추적하는 validation_score
from sklearn.model_selection import validation_curve
c_range = [0.001, 0.01, 0.1, 1, 10, 100]

# 샘플 사이즈 대신 하이퍼파라미터가 들어가므로 train_sizes 변수가 삭제
train_scores, test_scores = validation_curve(pl, X_train, y_train,
                                             # 대신  params_name으로 하이퍼파라미터 이름을,
                                             param_name='logisticregression__C',
                                             # params_range로 하이퍼파라미터 범위를 지정
                                             param_range=c_range, cv=10)

# 기타 표현법은 learning_curve와 동일
train_mean_val = np.mean(train_scores, axis=1)
train_std_val = np.std(train_scores, axis=1)
test_mean_val = np.mean(test_scores, axis=1)
test_std_val = np.std(test_scores, axis=1)

plt.plot(c_range, train_mean_val, marker='o', c='royalblue', label='accuracy of train_data')
plt.plot(c_range, test_mean_val, marker='s', c='green', label='accuracy of val_data')

plt.fill_between(c_range, train_mean_val + train_std_val, train_mean_val - train_std_val, alpha=0.2, color='blue')
plt.fill_between(c_range, test_mean_val + test_std_val, test_mean_val - test_std_val, alpha=0.2, color='green')

plt.xscale('log')
plt.ylim([0.7, 1.05])
plt.xlabel('C_range')
plt.ylabel('accuracy')
plt.legend(loc='best')

plt.tight_layout()
plt.show()