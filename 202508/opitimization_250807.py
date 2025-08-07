import pandas as pd
df = pd.read_csv('breast_cancer.csv')

# 최적 하이퍼파라미터 도출 과정

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

y = le.fit_transform(y)
# print(le.classes_)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# 모든 과정을 파이프라인으로 이어
from sklearn.pipeline import Pipeline
# GridSearchCV로 최적 하이퍼파라미터를 먼저 탐색한 다음
from sklearn.model_selection import GridSearchCV
# 검증 데이터에 대한 정확성까지 한 번에 뽑아내는 프로세스
from sklearn.model_selection import cross_val_score

pl = Pipeline([
    ('standardscaler', StandardScaler()),
    ('svc', SVC(random_state=0))
])

param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# 파이프라인으로 GS까지 연결하려면 param_grid 작성시 모델명__인자명 형태로 작성해야 한다!
# 그냥 학습 모델만 독립적으로 GS 객체를 만들 때는 인자명만 달아줘도 OK
param_grid = [
    {'svc__C' : param_range,
     'svc__kernel' : ['linear']},
    {'svc__C' : param_range,
     'svc__gamma' : param_range,
     'svc__kernel' : ['rbf']}
]

gs = GridSearchCV(pl, param_grid, cv=3)
scores = cross_val_score(gs, X_train, y_train, cv=5)

print(scores)
print()

#=================================================================================================

# 오차 행렬의 시각화
# 오차_행렬을 그대로 영어로 옮긴 confusion_matrix
from sklearn.metrics import confusion_matrix
pl.fit(X_train, y_train)
y_pred = pl.predict(X_test)

# 예측 점수 따는 것처럼 f(y_test, y_pred)로 도출
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
print()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(con_mat, cmap='viridis', alpha=0.5)

for i in range(con_mat.shape[0]) :
    for j in range(con_mat.shape[1]) :
        ax.text(x=j, y=i, s=con_mat[i, j], va='center', ha='center')

plt.xlabel('prediction')
plt.ylabel('reality')
plt.show()

# 디스플레이 전용 클래스로 더 간편히 호출 가능
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(pl, X_test, y_test)
plt.show()

# normalize 인자를 설정해 출력 데이터 정규화도 가능