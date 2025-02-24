■■ 예제58.단순회귀_분석.py


■ 1. 공분산 (Covariance)

 - 개념 및 정의

  공분산은 두 확률변수  X와 Y가 얼마나 함께 변하는지를 나타내는 통계량입니다.

 - 수식:  Cov(X,Y)=E[(X−E[X])(Y−E[Y])]

 - 해석:
     양의 공분산: X가 평균보다 큰 값을 보일 때 Y도 평균보다 큰 경향이 있음 (즉, 두 변수는 같은 방향으로 움직임).
   음의 공분산: X가 평균보다 큰 값을 보일 때 Y는 평균보다 작은 경향이 있음 (즉, 두 변수는 반대 방향으로 움직임).
   0의 공분산: 두 변수 간 선형 관계가 없음을 시사하지만, 반드시 독립임을 의미하지는 않습니다.

 - 중요 속성 및 오해

    - 독립과의 관계:

     * 정확한 설명: 만약  X와  Y가 독립이라면, 공분산은 0입니다.
   * 주의: 반대로, 공분산이 0이라고 해서 X와 Y가 항상 독립이라는 것은 아닙니다.

    - 측정 단위의 영향:

     공분산은 X와 Y의 단위의 곱으로 나타나므로, 측정 단위에 영향을 받습니다. (즉, 단위에 민감함)

    - 부호에 따른 해석:

        공분산이 음수라면, 한 변수의 값이 증가할 때 다른 변수의 값은 감소하는 경향이 있습니다.


■ 2. 회귀분석 (Regression Analysis)

 - 기본 개념

       회귀분석은 한 변수(종속변수)가 다른 변수(독립변수)에 의해 어떻게 영향을 받는지를 분석하는 통계적 방법입니다.

  - 종속변수와 독립변수:

       종속변수: 결과로 나타나는 변수, 보통 관심의 대상이 되는 변수입니다.
       독립변수: 종속변수에 영향을 주는 변수들로, 연속형이나 범주형 모두 가능합니다.

■ 회귀분석의 종류와 가정
 
 1. 선형 회귀 (Linear Regression):

   - 목적: 종속변수와 독립변수 간의 선형 관계를 모형화합니다.

   - 주요 가정: 잔차(오차항)는 정규분포를 따르고, 평균이 0이며 분산이 일정하다 (등분산성).

   - 최소제곱법 (Least Squares Estimation, LSE): 선형 회귀 계수를 LSE로 추정하면, 
                                             가정이 만족될 경우 **불편추정량(unbiased estimator)**의 성질을 가집니다.

2.로지스틱 회귀 (Logistic Regression):

   - 목적: 종속변수가 범주형(예: 이진 분류)인 경우에 사용됩니다.

   - 차이점:
           로지스틱 회귀는 잔차의 정규성을 가정하지 않습니다. (종속변수가 범주형이기 때문에 오차분포가 다르게 모델링됨)
           독립변수의 유형:  독립변수는 연속형뿐만 아니라 범주형 변수도 포함될 수 있으며, 
                      범주형 변수는 더미(dummy) 변수로 변환하여 분석합니다.




문제1.공분산에 대한 설명으로 옳은 것은?
    (2023년 제6회 빅데이터분석기사 필기)

1. X, Y가 독립이면, Cov(X,Y)=0이다
2. Cov(X,Y)=0이면, X와 Y는 항상 독립이다
3. Cov(X,Y)는 측정 단위와 무관하다
4. Cov(X,Y)<0이면, X값이 상승할 때 Y값도 상승한다

 정답: 


문제2. 회귀분석에 대한 설명으로 옳은 것은?
   (2022년 제4회 빅데이터분석기사 필기)

1.종속변수는 범주형이어야 한다
2.독립변수는 연속형이어야 한다
3.선형/로지스틱 회귀 모두 잔차 정규성을 가정한다
4.선형 회귀 계수를 최소제곱량(LSE)으로 추정하면 불편추정량의 특성을 가진다

정답: 


문제3. 두 변수 간의 공분산이 양수일 때의 해석으로 옳은 것은?
      (2021년 제2회 빅데이터분석기사 필기)

1.X값이 증가할 때 Y값은 감소하는 경향이 있다
2.X값이 증가할 때 Y값도 증가하는 경향이 있다
3.X와 Y는 서로 독립적인 관계이다
4.X와 Y는 비선형 관계이다

정답: 


■ 실습1. 탄닌 함유량과 애벌래 성장율 회귀 분석

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. 데이터를 불러옵니다.
df = pd.read_csv("c:\\data\\regression.txt", sep="\s+")
print(df.head())

# 2. 회귀모델을 생성합니다.
# sklearn 방식
X = df[['tannin']]
y = df['growth']
model = LinearRegression().fit(X, y)

# 회귀 계수와 절편 출력
print("회귀 계수:", model.coef_[0])
print("절편:", model.intercept_)
print("결정계수 (R²):", model.score(X, y))

# statsmodels를 이용한 상세 통계 결과
model_stats = ols('growth ~ tannin', data=df).fit()
print(model_stats.summary())

# 3. 생성한 회귀모델로 탄닌 함유량이 10일 때 성장률을 예측합니다.
new_data = pd.DataFrame({'tannin': [10]})
prediction = model.predict(new_data)
print(f"탄닌 함유량이 10일 때 예측 성장률: {prediction[0]:.4f}")

# statsmodels를 이용한 예측 (대안 방법)
prediction_stats = model_stats.predict(pd.DataFrame({'tannin': [10]}))
print(f"statsmodels 이용 예측 성장률: {prediction_stats[0]:.4f}")


😊문제1.  광고비와 매출간의 단순 회귀 분석 모델을 생성하고
           회귀 계수를 출력하시오 ! 

 데이터 있는곳:  https://cafe.daum.net/oracleoracle/Soei/68
 
 데이터: simple_hg.csv
 

답:




# 😊문제2. 미국 우주 왕복선 챌린져호의 폭파 원인을 단순회귀 분석으로 분석하시오!
# 목표: 온도가 O형링 파손에 얼마나 영향을 미치는지 데이터 분석

# 1. 데이터 불러오기
cha = pd.read_csv("c:\\data\\challenger.csv")
print(cha.head())

# 데이터 설명:
# distress_ct: o형링 파손수 
# temperature: 온도
# field_check_pressure: 압력
# flight_num: 비행기 번호


