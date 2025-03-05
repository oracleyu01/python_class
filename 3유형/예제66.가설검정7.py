■ 예제66. 가설검정7.세 모집단 이상의 모평균 차이 검정 (ANOVA)


가설검정
 ├─ 정규분포를 따르는 경우
 │   ├─ 단일 모집단의 모평균 검정
 │   │    ├─ 분산을 아는 경우  →  Z 검정   
 │   │    └─ 분산을 모르는 경우 →  t 검정 
 │   ├─ 두 모집단의 모평균 차이 검정
 │   │    ├─ 독립 표본         →  t 검정  
 │   │    └─ 대응 표본         →  t 검정
 │   └─ 세 모집단 이상의 모평균 차이 검정
 │        → ANOVA             <---  😊 이 부분을 하고 있어요
 └─ 정규분포를 따르지 않는 경우
      → 비모수적 가설 검정

--------------------------------

--------------------------------
[다변량분산분석(ANOVA) 이론 정리]

1. 정의:
   - ANOVA(분산분석)는 두 개 이상의 모집단(또는 집단) 간의 평균 차이가 통계적으로 유의한지를 검정하는 방법입니다.
   - 다변량분산분석(MANOVA)은 하나 이상의 독립변수를 대상으로 하여 여러 개의 종속변수에 대해 동시에 평균 차이를 검정합니다.
   - 일원배치 분산분석(One-way ANOVA)은 하나의 독립변수(요인)에 따른 한 개의 종속변수의 평균 차이를 검정하는 방법입니다.

2. 주요 특징 및 가정:
   - 다변량분산분석(MANOVA):
       - 독립변수: 1개 이상
       - 종속변수: 여러 개
       - 각 종속변수들이 상호 연관되어 있을 때, 이를 동시에 고려하여 집단 간 차이를 검정합니다.
   - 일원배치 ANOVA:
       - 독립변수와 종속변수가 각각 1개이며, 모집단 분산을 모를 때 사용.
       - 각 그룹의 관측치는 독립적이고, 정규성을 가정하며, 그룹 간 분산은 동질적(동분산성)이라는 가정을 전제로 합니다.

--------------------------------
[기출문제 정리]

● 2024년 제8회 빅데이터분석기사 필기
   - 문제:
         다변량분산분석(ANOVA)에 대한 설명으로 옳은 것은?
         선택지:
           1) 독립변수 1개, 종속변수 여러 개
           2) 독립변수 여러 개, 종속변수 1개
           3) 독립변수 1개 이상, 종속변수 여러 개
           4) 독립변수와 종속변수 모두 여러 개

   - 정답: 
 
● 2022년 제4회 빅데이터분석기사 필기
   - 문제:
         통계에서 평균에 대한 차이검정으로 모집단 3개 이상 사용하는 분석방법으로 가장 알맞은 것은?
         선택지:
           1) t검정
           2) z검정
           3) 분산분석
           4) 상관분석

   - 정답: 




--------------------------------
[파이썬 코드 예제: 일원배치 ANOVA]

# 예제: 세 그룹의 데이터를 이용하여 일원배치 ANOVA를 수행하는 코드
import numpy as np
from scipy import stats

# 예제 데이터: 세 집단의 측정값 (예: 세 가지 다른 처치 방법에 따른 결과)
group1 = np.array([5, 6, 7, 8, 7])
group2 = np.array([8, 9, 7, 10, 8])
group3 = np.array([6, 5, 7, 6, 6])





--------------------------------
[파이썬 문제: 일원배치 ANOVA를 이용한 다이어트 요법 효과 비교]

문제 설명:
  한 연구에서 세 가지 다른 다이어트 요법(A, B, C)이 체중 감량에 미치는 영향을 비교하고자 합니다.
  각 다이어트 그룹에서 10명의 참가자를 선정하여 체중 감량량(kg)을 측정하였습니다.
  
  다이어트 A: [3.2, 2.8, 3.5, 3.0, 3.1, 2.9, 3.3, 3.0, 2.7, 3.4]
  다이어트 B: [2.5, 2.7, 2.8, 2.4, 2.6, 2.5, 2.9, 2.7, 2.8, 2.6]
  다이어트 C: [3.6, 3.7, 3.5, 3.8, 3.6, 3.9, 3.7, 3.8, 3.6, 3.7]

  유의수준 α = 0.05에서, 세 그룹 간의 평균 체중 감량에 유의한 차이가 있는지
  일원배치 분산분석(One-way ANOVA)을 이용하여 검정하시오.
  
--------------------------------




