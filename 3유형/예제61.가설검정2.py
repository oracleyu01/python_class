
예제61. 가설검정2.단일 모집단의 모평균 검정: 분산을 아는 경우(Z 검정):

■ 예제61.가설검정2.단일 모집단의 모평균 검정: 분산을 아는 경우(Z 검정)

■ [가설검정 전체 ]

가설검정
 ├─ 정규분포를 따르는 경우
 │   ├─ 단일 모집단의 모평균 검정
 │   │    ├─ 분산을 아는 경우  →  Z 검정  <---  😊 이 부분을 하고 있어요
 │   │    └─ 분산을 모르는 경우 →  t 검정
 │   ├─ 두 모집단의 모평균 차이 검정
 │   │    ├─ 독립 표본         →  t 검정
 │   │    └─ 대응 표본         →  t 검정
 │   └─ 세 모집단 이상의 모평균 차이 검정
 │        → ANOVA
 └─ 정규분포를 따르지 않는 경우
      → 비모수적 가설 검정


■  [Z 검정 (Z-test)]

1. 정의:
   - 모집단의 분산(σ²)을 알고 있을 때 사용하는 검정 방법.
   - 표본의 크기가 큰 경우(n ≥ 30), 중심극한정리에 의해 표본평균이 정규분포를 
     따른다고 볼 수 있음.

2. 특징:
   - 검정통계량은 표준정규분포(Z ~ N(0,1))를 따름.
   - Z 검정에서는 모집단의 분산(또는 표준편차)이 알려져 있어야 함.

3. 예시:
   - 모집단의 분산이 이미 알려진 경우
   - 표본의 크기가 충분히 큰 경우(n ≥ 30)


■  [검정통계량 (Z-test)]

1. 검정통계량 공식:
   Z = ( X̄ - μ₀ ) / ( σ / √n )

   여기서:
   - X̄ : 표본평균
   - μ₀ : 귀무가설에서 제시된 모평균
   - σ  : 모표준편차
   - n  : 표본 크기

2. 가설 설정

   1) 양측검정 (Two-sided test)
      - 귀무가설(H₀): μ = μ₀
      - 대립가설(H₁): μ ≠ μ₀

   2) 우측검정 (Right-tailed test)
      - 귀무가설(H₀): μ ≤ μ₀
      - 대립가설(H₁): μ > μ₀

   3) 좌측검정 (Left-tailed test)
      - 귀무가설(H₀): μ ≥ μ₀
      - 대립가설(H₁): μ < μ₀



문제1. 모평균의 95% 신뢰구간을 구하는 식에서 Z값은? (2022년 제5회 빅데이터분석기사 필기)

1. 1.645
2. 1.96
3. 2.58
4. 3.00

정답:  2

해설설명: https://github.com/oracleyu01/python_class/blob/main/3유형/그림/z값.png

파이썬으로 위의 문제의 답을 확인하고 싶다면  다음과 같이 코딩합니다.

from scipy.stats import norm

# 신뢰수준 95%에서 Z 값 (양측검정)

z_value = norm.ppf(1 - 0.05 / 2)  # 1 - (1 - 0.95) / 2

print('95%의 신뢰구간의 Z 값:', z_value)

95%의 신뢰구간의 Z 값: 1.959963984540054
 
문제1. 신뢰수준 99%에서의 z 값(양측검정) 을 구하시오 

from scipy.stats import norm

# 신뢰수준 99%에서 Z 값 (양측검정)

z_value = norm.ppf(1 - 0.01 / 2)  # 1 - (1 - 0.95) / 2

print('95%의 신뢰구간의 Z 값:', z_value)



문제1. 어떤 공장에서 생산되는 제품의 무게는 평균이 500g이고 표준편차가 
        10g인 정규분포를 따른다고 한다. 

        새로운 생산 방식을 도입한 후 임의로 36개의 제품을 추출하여 측정한 결과 평균이 503g이 나왔다. 
        새로운 생산 방식이 제품의 평균 무게를 변화시켰다고 할 수 있는가? (α = 0.05)


귀무가설:  새로운 생산 방식이 제품의 평균무게를 변화시키지 않았다.

 H0(μ) = 500

대립가설: 새로운 생산 방식이 제품의 평균 무게를 변화시켰다

 H0(μ) != 500


답:

# 귀무가설:  새로운 생산 방식이 제품의 평균무게를 변화시키지 않았다.

#  H0(μ) = 500

# 대립가설: 새로운 생산 방식이 제품의 평균 무게를 변화시켰다

#  H0(μ) != 500

import  numpy  as  np
import  scipy.stats  as  stats
import  matplotlib.pyplot  as plt

#한글 폰트 설정 
plt.rc('font', family='Malgun Gothic')  # 윈도우 사용자이 한글깨짐 방지 
#plt.rc('font', family='Apple Gothic')  #맥 사용자
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지 

# 데이터 설정
# 문제1. 어떤 공장에서 생산되는 제품의 무게는 평균이 500g이고 표준편차가 
# 10g인 정규분포를 따른다고 한다. 

# 새로운 생산 방식을 도입한 후 임의로 36개의 제품을 추출하여 측정한 결과
# 평균이 503g이 나왔다. 
# 새로운 생산 방식이 제품의 평균 무게를 변화시켰다고 할 수 있는가? (α = 0.05)

n = 36  #표본의 갯수 
mu_sample=503  # 표본평균
sigma = 10  # 모집단의 표준편차 
mu_pop = 500  # 모집단의 평균 

# 정규분포를 따르는 표본 생성 
import numpy as  np
np.random.seed(123)
sample_data = np.random.normal(loc=mu_sample, scale=sigma, size= n)
sample_data = mu_sample + (sample_data - sample_data.mean()) / sample_data.std() * sigma
sample_data  # 36개의 표본 

# Z검정 통계량과 p_value 구하기 
#    Z = ( X̄ - μ₀ ) / ( σ / √n )

z_stat =( np.mean(sample_data) - mu_pop ) / ( sigma / np.sqrt(n) )
p_value = 2 * ( 1- stats.norm.cdf(abs(z_stat))) #양측검정

# 임계값 계산 
alpha = 0.05
z_crit = stats.norm.ppf( 1 - alpha / 2 )

# 결과 해석
print('z 검정 통계량' , round(z_stat, 5))
print('임계값 (z_crit) ±', round(z_crit,5))
print('p_value ', round(p_value,5))

if abs(z_stat) > z_crit:
    print('귀무가설을 기각합니다')
    print('새로운 생산방식이 제품의 평균무게를 변화시켰다고 할 수 있습니다')
else:
    print('귀무가설을 기각할 수 없습니다')

# 시각화 
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

plt.figure(figsize=(4,3))
plt.plot(x,y, label='표준정규분포',color='black')
plt.axvline(z_crit, color='red', linestyle='dashed', label=f"임계값 ({z_crit:.2f})")
plt.axvline(-z_crit, color='red', linestyle='dashed', label=f"임계값 (-{z_crit:.2f})")
plt.axvline(z_stat, color='blue', linestyle='solid', label=f"검정통계량 ({z_stat:.2f})")

# x축에 텍스트 추가 (검정통계량과 임계값 추가)
plt.text(z_crit,-0.1,f"{z_crit:.2f}", color='red', ha='center', fontsize=10)
plt.text(-z_crit,-0.1,f"-{z_crit:.2f}", color='red', ha='center', fontsize=10)
plt.text(z_stat,-0.13,f"{z_stat:.2f}", color='blue', ha='center', fontsize=10)

plt.legend()
plt.title('z-검정 결과 시각화')
plt.xlabel('z값')
plt.ylabel('확률밀도함수')
plt.show()

문제2. 한 제약회사에서 생산하는 진통제의 유효성분 함량은 평균이 50mg이고 
         표준편차가 2mg인  정규분포를 따른다.
         품질 관리자가 무작위로 49개의 진통제를 선택하여 검사한 결과 평균 함량이 
         49.5mg으로 나타났다. 
        유의수준 1%에서 이 진통제의 유효성분 함량이 감소했다고 할 수 있는가?

# 1. 가설설정

#귀무가설 (H0): 진통제의 유효성분 함량은 평균이 50mg 이다.(변화없음)
#  H0:  μ = 50

#대립가설 (H1): 진통제의 유효성분 평균 함량이 감소했다.
#  H1:  μ < 50 ( 단측검정, 좌측검정)


# 2. 문제에서 주어진 값 설정 

n = 49  # 표본크기
mu_sample = 49.5      #표본평균
sigma =  2      # 모집단 표준편차 
mu_pop =  50   # 모집단 평균 
alpha  =   0.01    # 유의수준

# 3. z 검정 수행( 좌측 검정)
#    Z = ( X̄ - μ₀ ) / ( σ / √n )

z_stat =( np.mean(mu_sample) - mu_pop ) / ( sigma / np.sqrt(n) )
p_value = stats.norm.cdf(z_stat)  # 좌측검정이므로 누적분포함수(cdf) 사용

# 3. z 검정 수행( 좌측 검정)
#    Z = ( X̄ - μ₀ ) / ( σ / √n )

z_stat =( np.mean(mu_sample) - mu_pop ) / ( sigma / np.sqrt(n) )
p_value = stats.norm.cdf(z_stat)  # 좌측검정이므로 누적분포함수(cdf) 사용

# 임계값 계산 
alpha = 0.01
z_crit = stats.norm.ppf(alpha)
z_crit


# 4. 결과 해석
print('z 검정 통계량' , round(z_stat, 5))
print('임계값 (z_crit)', round(z_crit,5))
print('p_value ', round(p_value,5))

if    z_stat < z_crit:
    print('귀무가설을 기각합니다')
    print('유효성분함량이 감소했다고 볼 수 있음')
else:
    print('귀무가설 채택- 유효성분함량이 감소했다고 보기 어려움')


# 시각화 
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

plt.figure(figsize=(4,3))
plt.plot(x,y, label='표준정규분포',color='black')
plt.axvline(z_crit, color='red', linestyle='dashed', label=f"임계값 ({z_crit:.2f})")
#plt.axvline(-z_crit, color='red', linestyle='dashed', label=f"임계값 (-{z_crit:.2f})")
plt.axvline(z_stat, color='blue', linestyle='solid', label=f"검정통계량 ({z_stat:.2f})")

# x축에 텍스트 추가 (검정통계량과 임계값 추가)
plt.text(z_crit,-0.1,f"{z_crit:.2f}", color='red', ha='center', fontsize=10)
#plt.text(-z_crit,-0.1,f"-{z_crit:.2f}", color='red', ha='center', fontsize=10)
plt.text(z_stat,-0.13,f"{z_stat:.2f}", color='blue', ha='center', fontsize=10)

plt.legend()
plt.title('z-검정 결과 시각화')
plt.xlabel('z값')
plt.ylabel('확률밀도함수')
plt.show()

