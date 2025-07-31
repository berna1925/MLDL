import pandas as pd
from io import StringIO
# 문자열을 파일처럼 취급하게 해주는 StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

# pandas의 숨겨진 다양한 기능
## dropna(thresh=k)로 k개 이상의 정상 데이터를 가진 행/열만 추출 가능
print(df.dropna(thresh=4))
print()

# 자료의 형태에 손대는 imputer와 transformer
from sklearn.impute import SimpleImputer
import numpy as np
# 열 단위로 통계적 연산을 진행해 결측치나 특정 값을 대체해주는 SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imputer.fit(df.values)
modified_data = imp.transform(df.values)
# fit_transform을 거쳐 np.array로 변형

modified_df = pd.DataFrame(modified_data, columns=df.columns)
# fillna가 있음에도 이것을 쓰는 것은 함수처럼 구성해 코드의 재사용성을 높일 수 있고
# fit_transform이 갖춰져 있어 파이프라인 처리도 쉽기 때문!
print(modified_df)
print()

# 결측치가 존재할 때 다른 특성들의 데이터로 예측 모델을 만들어 귀납적으로 결측치를 추론하는 IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
iimp = IterativeImputer()
imputed_data_iimp = iimp.fit_transform(df.values)

print(imputed_data_iimp)
print()

# KNN의 알고리즘을 이용해 결측치를 채우는 KNNImputer
from sklearn.impute import KNNImputer
knnimp = KNNImputer()
modified_data_knn = knnimp.fit_transform(df.values)

print(modified_data_knn)

from sklearn.preprocessing import FunctionTransformer
# SimpleImputer와 다르게 행 단위로 연산을 진행할 수 있게 하는 FunctionTransformer
# 기본적으로는 FT도 열 방향 연산을 수행
# 전치를 시켜 계산한 다음 다시 연산 결과를 전치해 행 벡터를 열 벡터로 복구시키듯 연산
ft = FunctionTransformer(lambda x : imputer.fit_transform(x.T).T, validate=False)
imputed_data = ft.fit_transform(df.values)

modified_df_ft = pd.DataFrame(imputed_data, columns=df.columns)
print(modified_df_ft)
print()