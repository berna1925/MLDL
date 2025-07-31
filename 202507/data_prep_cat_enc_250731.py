# 다양한 자료 인코딩 방식
import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10, 'class2'],
    ['red', 'L', 13, 'class1'],
    ['blue', 'XL', 15, 'class2'],
])

df.columns = ['color', 'size', 'price', 'label']
print(df)
print()

# map을 이용한 직접적인 값 매칭 방식
size_mapping = {'M' : 1, 'L' : 2, 'XL' : 3}
df['size'] = df['size'].map(size_mapping)
print(df)
print()

# 매핑은 귀찮지만 원래 형태로 복귀하기는 쉬운 편이다
# dictionary comprehension으로 key와 value의 위치를 바꿔줄 수 있음
size_inverse_mapping = {v : k for k, v in size_mapping.items()}
df['size'] = df['size'].map(size_inverse_mapping)
print(df)
print()

import numpy as np
# enumerate로 시리즈 내부의 레이블을 label로, 순서를 idx로 삼는 개체를 불러와 순서를 바꿔서 클래스 레이블처럼 활용
# enumerate를 쓰면 무조건 0부터 튜플이 전개되므로 클래스 인덱스로 쓰기 유리함!
# 일일이 레이블을 따서 매핑하는 원시적인 방식에서 벗어나 np.unique를 사용한 중복 제거 추출로 클래스 인덱싱
class_mapping = {label : idx for idx, label in enumerate(np.unique(df['label']))}
print(class_mapping)
print()

df['label'] = df['label'].map(class_mapping)
print(df)
print()

# 순서가 있는 데이터라면 LabelEncoder를 써도 무방
# 회귀 모델을 분석할 때는 레이블인코더의 결과가 수량적 의미를 가지게 될 수 있어 사용을 피해야
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
print(df)
print()


X = df[['color', 'size', 'price']].values
# 레이블인코더는 별 코드가 없는 한 단일 시리즈 단위로밖에 계산 불가 → 종속변수 대상 레이블링
color_le = LabelEncoder()
# fit, transform 계열 함수는 기본적으로 np.ndarray 꼴의 입력을 기대하므로 미리 values로 행렬 형태로 바꾼 것
# values로 바꾸지 않아도 작동은 함
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
print()

# 복수의 시리즈 단위 변환이 가능해 독립변수 대상 레이블링에 쓰이는 ordinalencoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
oe = OrdinalEncoder(dtype=int)
ss = StandardScaler()
cols = ['color', 'size']

ct = ColumnTransformer([
    ('ordinal_encoding', oe, cols),
    ('standard scaling', ss, ['price']),
])

df_transformed = ct.fit_transform(df)

df_transformed = pd.DataFrame(df_transformed, columns = ['color', 'size', 'price'])
df_transformed['label'] = le.fit_transform(df['label'])
print(df_transformed)
print()

# 원핫 인코딩
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

X = df[['color', 'size', 'price']].values
print(ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())
print()

# sklearn 모듈 밖에 있는 인코더들
from category_encoders import BinaryEncoder
from category_encoders import TargetEncoder
# fit_transform 처리를 할 때 반환 형태가 DataFrame인 관계로 np.ndarray를 기대하는 ColumnTransformer와 호환 안 됨
