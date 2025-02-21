
▩ 예제14.결측치를_다른값으로_채우는_함수

문법:  
df['컬럼명'] = df['컬럼명'].fillna(0)  # 결측치를 0으로 채움 
df['컬럼명'] = df['컬럼명'].fillna(df['컬럼명'].mean()) # 평균값으로 채움
df['컬럼명'] = df['컬럼명'].fillna(df['컬럼명'].median()) # 중앙값으로 채움 
df['컬럼명'] = df['컬럼명'].fillna(df['컬럼명'].mode()[0]) # 최빈값으로 채움 
df['컬럼명'] = df['컬럼명'].fillna(method='ffill') # 결측치 바로 앞의 값으로 채우기 
df['컬럼명'] = df['컬럼명'].fillna(method='bfill')  # 결측치 바로 뒤의 값으로 채우기

문제1.  emp 데이터 프레임의 결측치가 얼마나 있는지 확인하시오 !

import pandas as pd
emp = pd.read_csv("d:\\data\\emp.csv")



empno        0
ename        0
job            0
mgr          1
hiredate     0
sal           0
comm        10  <-- 결측치가 10개가 보입니다. 
deptno       0

문제2. emp 데이터 프레임의 comm 의 결측치를 comm의 평균값으로 채우시오

답: 



문제3. 부동산 허위매물 훈련 데이터를 불러와서 결측치가 있는 컬럼들의 
       데이터중 숫자형 컬럼의 결측치를 해당 컬럼의 평균값으로 채우시오 !

import pandas  as  pd

train = pd.read_csv("d:\\data\\train.csv")
train.isnull().sum()

