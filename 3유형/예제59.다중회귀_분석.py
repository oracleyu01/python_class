■■ 예제59.다중회귀_분석.py


■ 1. 다중회귀 분석 이란?

다중회귀분석은 한 개의 종속 변수와 여러 개의 독립 변수 간의 관계를 동시에 분석하는 통계 기법입니다.

--------------------------------
[기본 정의 및 모형]

정의:
  한 종속 변수에 영향을 미치는 여러 독립 변수들의 효과를 동시에 고려하여, 
  각 독립 변수가 종속 변수에 미치는 영향을 평가하는 분석 방법입니다.

모형 식:
  일반적인 다중회귀 모형은 다음과 같이 표현됩니다.

      Y = β0 + β1X1 + β2X2 + ... + βpXp + ε

  여기서,
      Y      : 종속 변수
      X1~Xp : 독립 변수들
      β0     : 상수항 (절편)
      β1~βp : 각 독립 변수의 회귀 계수
      ε      : 오차항 (모형에서 설명되지 않는 변동)

--------------------------------
[목적 및 활용]

- 예측:
  다중회귀분석을 통해 종속 변수의 값을 예측할 수 있습니다.

- 설명:
  각 독립 변수가 종속 변수에 미치는 영향(효과 크기와 방향)을 파악할 수 있습니다.

- 통제:
  여러 변수가 동시에 영향을 미치는 상황에서 특정 변수의 순수한 효과를 분석할 수 있습니다.

--------------------------------
[주요 가정]

다중회귀분석을 적용하기 위해 몇 가지 중요한 가정을 만족해야 합니다.

1. 선형성:
   종속 변수와 독립 변수들 간의 관계가 선형적이어야 합니다.
2. 독립성:
   각 관측치는 서로 독립적이어야 하며, 오차항도 독립적입니다.
3. 등분산성 (Homoscedasticity):
   오차항의 분산이 일정해야 합니다.
4. 정규성:
   오차항이 정규분포를 따라야 합니다.
5. 다중공선성:
   독립 변수들 사이에 지나치게 높은 상관관계가 없어야 합니다.

--------------------------------
[추정 방법]

최소제곱법 (Least Squares Estimation, LSE):
  회귀계수를 추정하는 가장 일반적인 방법으로, 각 데이터 점과 회귀직선 사이의 
  잔차 제곱합을 최소화하는 계수를 구합니다.
  이 방법으로 얻은 추정치는 불편추정량(unbiased estimator)의 성질을 가지며, 
  장기적으로 보면 실제 계수에 근접하게 됩니다.

--------------------------------
[결론]

다중회귀분석은 여러 요인이 동시에 작용하는 현실 세계의 데이터를 효과적으로 
설명하고 예측하는 데 매우 유용한 도구입니다. 이를 통해 변수들 간의 복잡한 관계를 
파악하고, 정책 결정이나 전략 수립 등의 다양한 분야에서 중요한 인사이트를 도출할 수 있습니다.


--------------------------------

■ 2. 결정계수란 ?

[결정계수 (Coefficient of Determination, R²)]

정의:
  회귀모형이 종속 변수의 총 변동 중에서 얼마를 설명하는지를 나타내는 지표입니다.

계산식:
  R² = SSR / SST = 1 - (SSE / SST)
  
  여기서,
      SSR : 회귀제곱합 (모형이 설명한 변동)
      SSE : 오차제곱합 (모형이 설명하지 못한 변동)
      SST : 총제곱합 (전체 변동)
      
해석:
  - R² 값이 1에 가까울수록 모형이 데이터를 잘 설명한다는 의미입니다.
  - 하지만, 독립 변수의 수가 늘어남에 따라 R²는 인위적으로 증가할 수 있으므로,
    모델 선택 시에는 수정 결정계수(Adjusted R²)를 함께 고려합니다.

--------------------------------

■ 3. 팽창계수란? 

[팽창계수 (Variance Inflation Factor, VIF)]

정의:
  각 독립변수가 다른 독립변수들과 얼마나 높은 상관관계를 가지는지를 나타내며,
  다중공선성의 정도를 수치로 표현한 지표입니다.

계산식:
  VIF_j = 1 / (1 - R_j²)
  
  여기서,
      R_j² : j번째 독립변수를 나머지 독립변수들로 회귀분석했을 때의 결정계수
      
해석:
  - VIF 값이 1에 가까우면 다중공선성이 거의 없음을 의미합니다.
  - 일반적으로 VIF 값이 5 이상(또는 경우에 따라 10 이상)이면 다중공선성이 심각할 수 있다고 봅니다.
  - VIF는 회귀계수 추정치의 분산이 다중공선성으로 인해 얼마나 증가하는지를 나타냅니다.

--------------------------------
요약:

  결정계수는 회귀모형의 설명력을 나타내고, 팽창계수는 독립변수 간의 상관관계(다중공선성)를 평가하는 중요한 지표입니다.




문제1. 다중공선성과 VIF에 대한 설명으로 옳은 것은?
     (2024년 제8회 빅데이터분석기사 필기)

  1. VIF가 1보다 작으면 다중공선성이 있다
  2. 다중회귀에서 독립변수 간 선형관계가 있으면 다중공선성이 있다
  3. 다중공선성이 있어도 예측에는 문제가 없다
  4. VIF가 5 이하면 다중공선성이 없다고 본다

정답: 


문제2. 다중회귀분석에서 결정계수(R²)에 대한 설명으로 틀린 것은?
      (2023년 제6회 빅데이터분석기사 필기)

  1. 결정계수는 1에 가까울수록 좋으나, 무조건 크다고 좋은 것은 아니다
  2. 결정계수는 전체 변동 중 회귀식이 설명 불가능한 변동의 비율이다
  3. 결정계수는 SSR/SST로 구할 수 있다
  4. 다중회귀분석에서 모델 선택 시 수정 결정계수도 함께 고려할 필요가 있다

정답: 


문제3. 다중회귀분석에서 독립변수 간 선형관계가 존재하여 회귀식이 오류를 범할 수 있는 것은?
      (2022년 제4회 빅데이터분석기사 필기)

 1. 이상치
 2. 등분산성
 3. 다중공선성
 4. 독립성

정답: 


■ 실습1.

# ➡️ 미국민 의료비를 예측하는 다중 회귀 모델 생성하기
# 
# #1. 데이터를 불러옵니다.
# #2. 데이터를 살펴봅니다.
# #3. 다중회귀 분석 모델을 생성합니다.
# #4. 회귀분석 결과를 해석합니다.
# #5. 회귀분석 모델의 성능을 높입니다. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# #1. 데이터를 불러옵니다.
ins = pd.read_csv("insurance.csv")







# #4. 회귀분석 결과를 해석합니다.
# 
# 1. 나이가 일년씩 더해질 때 마다 평균적으로 의료비가 증가될 것으로 예상됩니다.
# 2. 더미변수인 sex_male이 자동으로 추가되면서 변수값의 상대적 추정을 
#    했을때 결과는 남성은 여성에 비해서 매년 의료비가 적게 든다고 예상합니다.
# 3. 비만지수(bmi)의 단위가 증가할 때 연간 의료비가 증가될 것으로 예상합니다.
# 4. 자녀수가 한명이 추가될때마다 의료비가 추가될 것으로 예상합니다.
# 5. 흡연자(smoker_yes)가 비흡연자보다 매년 평균 의료비가 더 많이 들 것으로 예상합니다.
# 6. region_northeast에 비해 다른 지역들은 의료비가 덜 드는 것으로 예상됩니다.

# #5. 회귀분석 모델의 성능을 높입니다. 
# 파생변수를 추가해서 설명력을 높입니다.

# 파생변수1. 비만 여부 컬럼을 추가합니다.
# 가설:
#  귀무가설: 비만은 의료비에 영향이 없다.
#  대립가설: 비만은 의료비에 영향이 있다. 




# 파생변수2. 비만이면서 흡연하는 경우의 상호작용 변수 추가
# 가설: 
# 귀무가설: 비만이면서 흡연을 하는것은 의료비에 영향이 없다.
# 대립가설: 비만이면서 흡연을 하는것은 의료비에 영향이 있다.



# 다음과 같이 모델을 생성해도 됩니다.
# 상호작용항을 포함한 회귀분석 (R의 formula 방식과 유사)


■ 문제1. 우주 왕복선 데이터를 다중회귀분석하여 o형링 파손에 가장 큰 영향을 주는 독립변수가 무엇인지 출력하시오 



