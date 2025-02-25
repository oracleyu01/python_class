■ 예제65. 가설검정6 .두 모집단의 모평균 차이 검정: 대응표본 (t 검정)


가설검정
 ├─ 정규분포를 따르는 경우
 │   ├─ 단일 모집단의 모평균 검정
 │   │    ├─ 분산을 아는 경우  →  Z 검정   
 │   │    └─ 분산을 모르는 경우 →  t 검정 
 │   ├─ 두 모집단의 모평균 차이 검정
 │   │    ├─ 독립 표본         →  t 검정  
 │   │    └─ 대응 표본         →  t 검정<---  😊 이 부분을 하고 있어요
 │   └─ 세 모집단 이상의 모평균 차이 검정
 │        → ANOVA
 └─ 정규분포를 따르지 않는 경우
      → 비모수적 가설 검정

--------------------------------
[대응표본 t-검정 이론 정리]

1. 정의:
   - 대응표본 t-검정(Paired Sample t-test)은 동일한 대상이나 매칭된 쌍의 데이터를 이용하여,
     두 조건(예: 처치 전후, 두 시점, 두 방법 등)에서의 평균 차이가 통계적으로 유의한지를 검정하는 방법입니다.
   - 주로 동일한 대상에게서 측정된 두 조건의 결과 차이에 대해, 그 차이의 평균이 0과 다른지를 평가합니다.

2. 주요 가정:
   - 각 쌍의 관측치 간 차이가 정규분포를 따른다.
   - 데이터는 쌍으로 짝지어져 있으며, 각 쌍은 서로 독립적이다.
   - 표본의 크기가 작을 경우, 차이의 정규성 가정이 중요하다.

3. 검정통계량:
   - 차이값(d_i)을 각 쌍에서 계산한 후, 그 평균(d̄)과 표준편차(s_d)를 구합니다.
   - 검정통계량 t는 다음과 같이 계산됩니다.
         t = d̄ / (s_d / √n)
     여기서 n은 쌍의 수입니다.
   - 자유도(df)는 n - 1 입니다.

--------------------------------
[기출문제 정리]

● 2024년 제7회 빅데이터분석기사 필기
   - 문제:
         대응표본 t-검정에 대한 설명으로 옳은 것은?
         1) 서로 다른 두 집단의 평균을 비교할 때 사용한다
         2) 같은 집단에서 두 시점이나 두 조건에서 나온 데이터를 비교할 때 사용한다
         3) 정규성 가정이 필요하지 않다
         4) 표본 크기가 항상 30개 이상이어야 한다
   - 정답: 
   - 해설:
         대응표본 t-검정은 동일한 대상이나 매칭된 쌍의 데이터를 비교하여 두 조건 간의 평균 차이를 분석할 때 사용합니다.

● 2023년 제6회 빅데이터분석기사 필기
   - 문제:
         대응표본 t-검정을 사용하는 경우는?
         1) 동일한 대상에게 처치 전후 효과를 측정할 때
         2) 서로 다른 두 집단의 평균을 비교할 때
         3) 세 개 이상 집단의 평균을 비교할 때
         4) 범주형 변수 간의 관계를 분석할 때
   - 정답: 
   - 해설:
         대응표본 t-검정은 같은 대상에게서 처치 전후 데이터를 수집하여, 변화가 유의한지를 검정할 때 사용합니다.

● 2022년 제5회 빅데이터분석기사 필기
   - 문제:
         대응표본 t-검정의 가정으로 옳은 것은?
         1) 두 집단의 표본 크기가 같아야 한다
         2) 두 집단이 서로 독립적이어야 한다
         3) 두 집단의 차이가 정규분포를 따라야 한다
         4) 두 집단의 분산이 같아야 한다
   - 정답: 
   - 해설:
         대응표본 t-검정은 각 쌍의 차이가 정규분포를 따른다는 가정을 전제로 합니다.
         (표본 크기가 충분히 크다면 정규성 가정이 완화될 수 있으나, 소규모 표본의 경우 반드시 확인해야 합니다.)

--------------------------------
[파이썬 코드 예제: 대응표본 t-검정]

# 예제: 동일한 대상의 처치 전후 혈압 데이터를 이용한 대응표본 t-검정

import numpy as np
from scipy import stats

# 예제 데이터: 10명의 환자에 대해 처치 전과 후의 혈압 측정값 (단위: mmHg)
pre_treatment = np.array([130, 128, 135, 132, 129, 131, 134, 127, 133, 130])
post_treatment = np.array([125, 126, 130, 128, 127, 129, 131, 125, 130, 128])


답: 


--------------------------------
[문제: 대응표본 t-검정을 이용한 처치 전후 혈압 비교]

문제 설명:
한 연구에서 10명의 환자에게 처치 전과 처치 후 혈압을 측정하였습니다.
처치 전 혈압 데이터: [130, 128, 135, 132, 129, 131, 134, 127, 133, 130] (mmHg)
처치 후 혈압 데이터: [125, 126, 130, 128, 127, 129, 131, 125, 130, 128] (mmHg)
유의수준 α = 0.05에서, 처치 전과 후의 혈압 차이가 통계적으로 유의미한지
대응표본 t-검정을 수행하여 검정 결과를 확인하시오.

--------------------------------
[파이썬 코드 예제: 대응표본 t-검정]

import numpy as np
from scipy import stats

# 데이터 설정: 동일한 환자에서 측정한 처치 전후 혈압 (단위: mmHg)
pre_treatment = np.array([130, 128, 135, 132, 129, 131, 134, 127, 133, 130])
post_treatment = np.array([125, 126, 130, 128, 127, 129, 131, 125, 130, 128])

답: 
