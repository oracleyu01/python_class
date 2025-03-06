"""
예제70: 빅데이터 분석 기사 9회 시험 전체 복기


필요한 라이브러리 임포트
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

np.random.seed(42)  # 재현성을 위해

# =============================================================================
# ■ 작업형 1-1 (대출액 관련 문제)
# =============================================================================
# [문제 설명]
# - 데이터셋 구성: 지역코드, 성별(1: 남자, 2: 여자), A금액, B금액
# - 파생 컬럼 '대출액' = A금액 + B금액 생성
# - 지역코드별, 성별별 대출액 합계를 구한 후, 남녀 간 차액(절대값)을 계산하고,
#   차액 절대값이 가장 큰 지역코드 번호(정수형)를 구하시오.
#
# [가상의 데이터 생성]

n = 100
region_codes = np.random.choice([4146510700, 4146510800, 4146510900, 4146511000, 4146511100], size=n)
genders = np.random.choice([1, 2], size=n)
A_amount = np.random.randint(100, 1000, size=n)
B_amount = np.random.randint(50, 500, size=n)

df_loan = pd.DataFrame({
    '지역코드': region_codes,
    '성별': genders,
    'A금액': A_amount,
    'B금액': B_amount
})
df_loan['대출액'] = df_loan['A금액'] + df_loan['B금액']

# =============================================================================
# [문제 풀이]
# 여기에 작업형 1-1 문제의 풀이 코드를 작성하세요.
# 예: 지역코드별, 성별별 대출액 집계, 차액 절대값 계산, 최대 차액 지역코드 선택 등.
# =============================================================================
# (답 코드 작성 영역)





# =============================================================================
# ■ 작업형 1-2 (범죄율 관련 문제)
# =============================================================================
# [문제 설명]
# - 데이터셋 구성: 연도, 구분(발생건수/검거건수), 33개 범죄유형(여기선 예제로 3개 컬럼 사용)
# - 각 연도별, 각 범죄유형별 검거율(검거건수 / 발생건수) 계산 후,
#   검거율이 가장 높은 범죄유형의 검거 건수를 모두 합산하여 출력하시오.
#
# [가상의 데이터 생성]

data = {
    '연도': [2018, 2018,2019,2019],
    '구분': ['발생건수', '검거건수','발생건수','검거건수'],
    '범죄1': [15556, 253, 15560, 289],
    '범죄2': [569, 100, 600, 99],
    '범죄3': [300, 50, 200, 60]  # 예시 값
}
df_crime = pd.DataFrame(data)
df_crime

# =============================================================================
# [문제 풀이]
# 여기에 작업형 1-2 문제의 풀이 코드를 작성하세요.
# 예: 발생건수와 검거건수 분리, 검거율 계산, 최대 검거율 범죄유형 선택 후 해당 검거건수 합산.
# =============================================================================
# (답 코드 작성 영역)




# =============================================================================
# ■ 작업형 1-3 (근속연수 관련 문제)
# =============================================================================
# [문제 설명]
# - 데이터셋 구성: 사원ID, 소속부서, 성과등급, 근속연수 등
# - 성과등급이 null인 사원들의 결측치는 소속 부서의 성과등급 평균으로 대체하고,
#   근속연수의 결측치는 부서와 성과등급이 같은 사원의 근속연수 평균으로 대체하시오.
# - 결측치 대체 후, 문제에서 제시한 조건에 따라 a값과 b값을 산출하고, a+b 값을 구하시오.
#
# [가상의 데이터 생성]

n_emp = 50
departments = np.random.choice(['부서1', '부서2', '부서3'], size=n_emp)
performance = np.random.choice(['A', 'B', 'C', None], size=n_emp, p=[0.3, 0.3, 0.3, 0.1])
years = np.random.randint(5, 20, size=n_emp).astype(float)
missing_idx = np.random.choice(n_emp, size=5, replace=False)
years[missing_idx] = np.nan

df_emp = pd.DataFrame({
    '사원ID': range(1, n_emp+1),
    '소속부서': departments,
    '성과등급': performance,
    '근속연수': years
})

# =============================================================================
# [문제 풀이]
# 여기에 작업형 1-3 문제의 결측치 처리 및 a, b 산출, 최종 a+b 계산 코드를 작성하세요.
# 예: 그룹별 평균 계산 후 결측치 대체, a값과 b값 산출, a+b 계산.
# =============================================================================
# (답 코드 작성 영역)






# =============================================================================
# ■ 작업형 2 (다중 분류 문제: 농약 검출 여부 예측)
# =============================================================================
# [문제 설명]
# - 데이터셋 구성: 연도, 작물 유형, 온도, 습도, 지역명, 농약 종류, 농약 사용량, 농약 빈도, 토양유형 등
# - 종속변수: 농약검출여부 (0, 1, 2)
# - 목표: 주어진 데이터를 바탕으로 농약 검출 여부를 예측하는 분류 모델을 구축하시오.
# - 평가지표: macro F1 Score (모델 성능 비교 시 참고)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# [가상의 데이터 생성 (F1 Score 개선)]

n_samples = 2000  # 데이터 크기 유지
np.random.seed(42)  # 재현성을 위한 시드 설정

# 베이스 값 생성 - 농약 검출 여부와의 상관관계를 위한 기초 데이터
base_values = np.random.normal(0, 1, size=n_samples)

years = np.random.choice([2018, 2019, 2020, 2021], size=n_samples)
crop_type = np.random.choice(['과일', '채소', '곡물'], size=n_samples)
temperature = np.random.normal(25, 3, size=n_samples)  # 원본 코드대로 유지
humidity = np.random.uniform(50, 70, size=n_samples)  # 원본 코드대로 유지
region = np.random.choice(['지역1', '지역2', '지역3'], size=n_samples)
pesticide_type = np.random.choice(['유형A', '유형B', '유형C'], size=n_samples)

# 농약 사용량과 빈도를 base_values와 연결하여 예측력 향상
pesticide_usage = 50 + 35 * base_values + np.random.normal(0, 5, size=n_samples)
pesticide_usage = np.clip(pesticide_usage, 10, 90)  # 범위 제한
pesticide_frequency = 6 + np.round(3 * base_values).astype(int)
pesticide_frequency = np.clip(pesticide_frequency, 3, 10).astype(int)  # 범위 제한

soil_type = np.random.choice(['토양1', '토양2'], size=n_samples)

# 농약 검출 여부 생성 (명확한 패턴으로 설정)
# 기준값 생성 (여기서 base_values는 농약 검출 여부와 강한 상관관계를 가짐)
detection_score = 1.2 * base_values + 0.8 * (pesticide_usage/90) + 0.6 * (pesticide_frequency/10)

# 경계를 분명하게 구분하여 모델이 쉽게 학습할 수 있도록 함
pesticide_detect = np.zeros(n_samples, dtype=int)
pesticide_detect[detection_score > 0.3] = 1
pesticide_detect[detection_score > 1.5] = 2

# 특정 조합에 대한 명확한 룰 추가 (모델의 예측력 향상)
for i in range(n_samples):
    if pesticide_type[i] == '유형A' and pesticide_usage[i] > 70:
        pesticide_detect[i] = 2
    elif pesticide_type[i] == '유형C' and pesticide_usage[i] < 30:
        pesticide_detect[i] = 0
    
    # 작물 유형과 지역에 따른 특별한 패턴
    if crop_type[i] == '과일' and region[i] == '지역1':
        pesticide_detect[i] = 2 if pesticide_frequency[i] > 7 else pesticide_detect[i]
    elif crop_type[i] == '채소' and soil_type[i] == '토양1':
        pesticide_detect[i] = 0 if pesticide_frequency[i] < 5 else pesticide_detect[i]

df_agri = pd.DataFrame({
    '연도': years,
    '작물유형': crop_type,
    '온도': temperature,
    '습도': humidity,
    '지역명': region,
    '농약종류': pesticide_type,
    '농약사용량': pesticide_usage,
    '농약빈도': pesticide_frequency,
    '토양유형': soil_type,
    '농약검출여부': pesticide_detect
})

# CSV 파일로 저장
df_agri.to_csv('농약_데이터.csv', index=False)

#답 : 






# =============================================================================
# ■ 작업형 3-1 (다중회귀분석 모델 문제)
# =============================================================================
# [문제 설명]
# - 데이터셋: 총 175행 (예: 'design' 목표 변수와 4개의 독립변수)
# - 학습용: 처음 140행, 평가용: 나머지
#
# 문제 1: 학습용 데이터로 다중회귀 모델 구축 후, p-value가 0.05보다 작은
#          (유의미한) 설명변수(상수 제외)의 개수를 구하시오.
#
# 문제 2: 학습용 데이터에서 독립변수 2개만 사용하여 모델을 구축하고,
#          실제값과 예측값의 피어슨 상관계수를 소수 셋째 자리까지 구하시오.
#
# 문제 3: 문제 2의 모델을 사용하여 평가용 데이터에 대한 예측값과 실제값을 비교해 RMSE를
#          소수 셋째 자리까지 구하시오.
#
# [가상의 데이터 생성]

n_total = 175
df_reg = pd.DataFrame({
    'design': np.random.normal(50, 10, size=n_total),
    '변수1': np.random.normal(5, 2, size=n_total),
    '변수2': np.random.normal(10, 3, size=n_total),
    '변수3': np.random.normal(15, 4, size=n_total),
    '변수4': np.random.normal(20, 5, size=n_total)
})

train_reg = df_reg.iloc[:140].reset_index(drop=True)
test_reg = df_reg.iloc[140:].reset_index(drop=True)

# =============================================================================
# [문제 풀이]
# 여기에 작업형 3-1 문제의 다중회귀분석 모델 구축 및 평가 코드를 작성하세요.
# 문제 1: p-value 기준 유의 변수 개수 구하기
# 문제 2: 독립변수 2개 모델로 피어슨 상관계수 계산
# 문제 3: 평가용 데이터 RMSE 계산
# =============================================================================
# (답 코드 작성 영역)



# =============================================================================
# ■ 작업형 3-2 (로지스틱 회귀분석 문제)
# =============================================================================
# [문제 설명]
# - 가상의 데이터셋 생성 (로지스틱 회귀용)
#
# 문제 1: 로지스틱 회귀모형을 구축 후, 특정 변수(예: '독립변수1')의 p-value를
#          소수 셋째 자리까지 구하시오.
#
# 문제 2: 위 모형에서 다른 변수(예: '판매량')의 오즈비(Odds Ratio)를 소수 셋째 자리까지 구하시오.
#
# 문제 3: 로지스틱 회귀모형으로 예측한 확률 중, 0.3 초과하는 사례 수를 구하시오.
#
# [가상의 데이터 생성]

n_log = 150
df_log = pd.DataFrame({
    '타겟변수': np.random.choice([0, 1], size=n_log, p=[0.6, 0.4]),
    '판매량': np.random.normal(100, 20, size=n_log),
    '독립변수1': np.random.normal(50, 10, size=n_log),
    '독립변수2': np.random.normal(30, 5, size=n_log)
})

# =============================================================================
# [문제 풀이]
# 여기에 작업형 3-2 문제의 로지스틱 회귀분석 모형 구축 및 평가 코드를 작성하세요.
# 문제 1: 특정 변수('독립변수1')의 p-value 계산
# 문제 2: '판매량' 변수의 오즈비 계산
# 문제 3: 예측 확률 중 0.3 초과 사례 수 계산
# =============================================================================
# (답 코드 작성 영역)

