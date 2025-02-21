
▩ 예제16.숫자함수2_(trunc)

 trunc 함수는 반올림하는게 아니라 그냥 버리는 함수입니다.
 파이썬에 내장된 함수가 아니어서 math 라는 모듈에서 불러와서
 사용하겠습니다. 

예제: 
import  math 

print( math.trunc( 16.554 ) )  # 16
print( math.trunc( 17.554 ) )  # 17
print( math.trunc( 18.554 ) )  # 18

그냥 소수점 이하를 다 버려버리는 결과를 출력하고 있습니다. 

문제1. 데이콘 부동산 허위매물 훈련 데이터를 불러와서 총층의 결측치를
         총층의 평균값으로 채워넣고 소수점 이하는 다 버리고 출력되게하시오

import pandas  as  pd

train = pd.read_csv("d:\\data\\train.csv")
train.head()

답:



