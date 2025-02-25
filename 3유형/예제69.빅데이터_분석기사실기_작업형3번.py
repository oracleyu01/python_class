
예제69: 빅데이터 분석 기사 실기 작업형 3번 문제 풀이]

문제 개요:
  - 타이타닉호 승객 데이터를 이용하여 생존 여부(Survived)와 관련된 통계적 분석을 수행하는 문제입니다.
  
데이터 설명:
  - 파일: Titanic.csv
  - 주요 변수:
      * Survived: 생존 여부 (1: 생존, 0: 사망)
      * Gender: 성별
      * SibSp: 함께 탑승한 형제/자매 및 배우자 수
      * Parch: 함께 탑승한 부모 및 자녀 수
      * Fare: 요금

세부 문항:
  ① Gender와 Survived 변수 간의 독립성 검정을 위한 카이제곱 통계량을 구하시오.
      - 결과는 소수 셋째 자리까지 반올림하여 출력.
  ② 로지스틱 회귀모형을 이용하여, 독립변수(Gender, SibSp, Parch, Fare)를 사용하고,
      종속변수로 Survived를 대상으로 할 때, Parch 변수의 계수값을 소수 셋째 자리까지 반올림하여 구하시오.
  ③ 위 로지스틱 회귀모형에서 SibSp 변수의 오즈비(Odds ratio)를 소수 셋째 자리까지 반올림하여 구하시오.

--------------------------------
[파이썬 코드 예제]

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm

# 데이터 로드
df = pd.read_csv('data/Titanic.csv')

# ① Gender와 Survived 변수 간의 독립성 검정 (카이제곱 검정)
# 교차표(Contingency Table) 생성
contingency_table = pd.crosstab(df['Gender'], df['Survived'])
# 카이제곱 검정 수행
chi2, p, dof, expected = chi2_contingency(contingency_table)
# 결과 출력 (소수 셋째 자리까지 반올림)
print(f"① 카이제곱 통계량: {round(chi2, 3)}")

# ② 로지스틱 회귀모형: Parch 변수의 계수값 구하기
# 독립변수 설정: Gender, SibSp, Parch, Fare
X = df[['Gender', 'SibSp', 'Parch', 'Fare']]
# Gender 변수가 범주형이면 더미 변수로 변환 (첫 번째 범주 제거)
if X['Gender'].dtype == 'object':
    X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
# 종속변수 설정: Survived
y = df['Survived']
# 상수항 추가
X = sm.add_constant(X)
# 로지스틱 회귀모형 적합 (출력 생략)
logit_model = sm.Logit(y, X).fit(disp=False)
# Parch 변수의 계수 추출 후 반올림
parch_coef = logit_model.params['Parch']
print(f"② Parch 변수의 계수값: {round(parch_coef, 3)}")

# ③ 로지스틱 회귀모형: SibSp 변수의 오즈비(Odds Ratio) 계산
sibsp_coef = logit_model.params['SibSp']
odds_ratio = np.exp(sibsp_coef)
print(f"③ SibSp 변수의 오즈비: {round(odds_ratio, 3)}")

--------------------------------
[파이썬 문제]

문제:
  타이타닉호 승객 데이터를 사용하여 다음의 분석을 수행하시오.
  
  1) Gender와 Survived 변수 간의 독립성을 검정하기 위해
     카이제곱 검정을 수행하고, 카이제곱 통계량을 소수 셋째 자리까지 반올림하여 출력하시오.
     
  2) Gender, SibSp, Parch, Fare 변수를 독립변수로 하고 Survived를 종속변수로 하는 로지스틱 회귀모형을 적합한 후,
     Parch 변수의 계수값을 소수 셋째 자리까지 반올림하여 출력하시오.
     
  3) 동일한 로지스틱 회귀모형에서 SibSp 변수의 오즈비(Odds Ratio)를 소수 셋째 자리까지 반올림하여 출력하시오.
  
  데이터 파일은 'data/Titanic.csv'에 위치하며, 결과는 각 항목별로 출력하시오.

유의사항:
  - 필요한 라이브러리: pandas, numpy, scipy.stats, statsmodels.api
  - 결과는 지정된 반올림 자리수(소수 셋째 자리)로 출력합니다.
  
작성하시오.
