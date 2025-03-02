예제69: 빅데이터 분석 기사 실기 작업형 3번 문제 풀이]

시험환경: https://dataq.goorm.io/exam/3/%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EA%B8%B0%EC%82%AC-%EC%8B%A4%EA%B8%B0-%EC%B2%B4%ED%97%98/quiz/4%3Fembed


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

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import math

# 데이터 불러오기

df = pd.read_csv('data/Titanic.csv')

