

■ 예제62.가설검정3.단일 모집단의 모평균 검정: 분산을 모르는 경우 (t 검정)


가설검정
 ├─ 정규분포를 따르는 경우
 │   ├─ 단일 모집단의 모평균 검정
 │   │    ├─ 분산을 아는 경우  →  Z 검정   
 │   │    └─ 분산을 모르는 경우 →  t 검정 <---  😊 이 부분을 하고 있어요.
 │   ├─ 두 모집단의 모평균 차이 검정
 │   │    ├─ 독립 표본         →  t 검정
 │   │    └─ 대응 표본         →  t 검정
 │   └─ 세 모집단 이상의 모평균 차이 검정
 │        → ANOVA
 └─ 정규분포를 따르지 않는 경우
      → 비모수적 가설 검정

--------------------------------
[단일 모집단 t-검정 이론 정리]

1. 정의:
   - 단일 모집단 t-검정은 하나의 모집단에 대한 모평균(μ)에 대한 가설을 검정하는 통계적 방법입니다.
   - 주로 모분산(σ²)이 알려지지 않았거나, 표본의 크기가 작을 때(n < 30) 사용됩니다.

2. 사용 상황:
   - 모집단이 정규분포를 따른다는 가정 하에,
   - 표본의 크기가 작거나 모분산 정보를 모를 때,
   - 단일 모집단의 평균이 특정 값과 다른지를 검정하고자 할 때 사용됩니다.

3. 주요 가정 (Assumptions):
   - 모집단이 정규분포를 따른다.
   - 데이터는 연속형이며, 표본은 독립적으로 추출된다.
   - (참고: 표본의 크기가 충분히 크면 중심극한정리에 의해 정규성을 근사할 수 있으나, 
        단일 모집단 t-검정은 보통 모분산이 알려지지 않은 경우에 적용됩니다.)

4. 검정통계량:
     t = ( X̄ - μ₀ ) / ( s / √n )

     여기서,
       X̄ : 표본평균,
       μ₀ : 귀무가설에서 제시한 모평균,
       s  : 표본표준편차,
       n  : 표본의 크기.

5. 해석:
   - 계산된 t 값을 자유도(n-1)를 가진 t 분포와 비교하여, 
      p-value를 산출하고 귀무가설(H₀)을 기각할지 결정합니다.
   
  - 유의수준(α)에 따라 p-value가 작으면(예: p < α) 
    귀무가설을 기각하고, 그렇지 않으면 채택합니다.

--------------------------------
[기출문제 정리]

● 2023년 제7회 빅데이터분석기사 필기

   - 문제1: 단일 표본 t-검정의 가정으로 옳은 것은?
         1) 표본의 크기가 30개 이상이어야 한다
         2) 모집단이 정규분포를 따라야 한다
         3) 두 집단의 분산이 같아야 한다
         4) 데이터가 연속형이 아니어야 한다

   - 정답:  2

   - 해설: 단일 모집단 t-검정은 주로 표본의 크기가 작고(보통 n < 30), 
            모분산을 모르는 경우에 사용되며, 
            이때 모집단이 정규분포를 따른다는 가정이 필수적입니다.

● 2022년 제5회 빅데이터분석기사 필기

   - 문제2: 표본의 크기가 작은 경우(n < 30) 단일 모집단의 평균 검정에서 사용하는 검정은?
         1) z-검정
         2) t-검정
         3) 카이제곱 검정
         4) F-검정


   - 정답:  2

   - 해설: 표본의 크기가 작고 모분산을 모르는 경우, t-검정이 적절한 검정 방법입니다.

● 2021년 제2회 빅데이터분석기사 필기

   - 문제3: 단일 표본 t-검정에 대한 설명으로 옳은 것은?

         1) 모분산을 알고 있을 때 사용한다
         2) 표본이 정규분포를 따른다고 가정한다
         3) 표본 크기가 클 때는 사용할 수 없다
         4) 두 집단 간의 차이를 검정할 때 사용한다

   - 정답:  2

   - 해설: 단일 모집단 t-검정은 모집단이 정규분포를 따른다는 가정 하에 수행되므로, 
            해당 설명이 옳습니다.

문제1. (좌측검정)
# 한 학급의 수학 성적이 평균 70점 이상이라고 알려져있다.
# 이를 검증하기 위해 이 학급에서 16명을 무작위로 선발하여 시험을 보았더니 
# 평균이 68점, 표준편차가 5점이었다. 유의수준 5%에서 
# 수학성적 평균이 70점 이상이 아니라고 하는 주장이 옳다고 할 수 있는가?

# 답:

# 1. 가설설정

#귀무가설 (H0) (원래 알려진 정보): 학급의 수학성적 평균은 70점이상이다. 
#  H0:  μ >= 70

#대립가설 (H1) (검증하려는 주장): 학급의 수학성적 평균이 70점보다 미만이다.
#  H1:  μ < 70 ( 단측검정, 좌측검정)

#검정방향: 좌측검정(표본 통계량이 기준값보다 작은 방향으로 유의한지를 검정)


#2. 문제에서 주어진 값 설정

n = 16  # 표본크기
mu_sample = 68      #표본평균
sigma =  5      # 표본의 표준편차 
mu_pop =  70   # 모집단 평균 
alpha  =   0.05    # 유의수준
df = n - 1   #  자유도 

# 3. t 검정 수행( 좌측 검정)
#    t = ( X̄ - μ₀ ) / ( s / √n )  # s는 표본의 표준편차 
from  scipy.stats  import   t 

t_stat =( mu_sample - mu_pop ) / ( sigma / np.sqrt(n) )
p_value = t.cdf(t_stat, df )  # 좌측검정이므로 누적분포함수(cdf) 사용


# 4. 결과 해석
print('t 검정 통계량' , round(t_stat, 5))
print('임계값 (t_crit)', round(t_crit,5))
print('p_value ', round(p_value,5))

if    t_stat < t_crit:
    print('귀무가설을 기각하고 대립가설이 채택')
    print('학습의 수학성적 평균이 70점보다 낮다고 볼 수 있음')
else:
    print('귀무가설 채택- 학습의 수학성적 평균이 70점보다 낮다고 보기 어려움')


# 시각화 
from  scipy.stats  import   t 
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df)

plt.figure(figsize=(4,3))
plt.plot(x,y, label='t-분포',color='black')
plt.axvline(t_crit, color='red', linestyle='dashed', label=f"임계값 ({t_crit:.2f})")
plt.axvline(t_stat, color='blue', linestyle='solid', label=f"검정통계량 ({t_stat:.2f})")

#기각역 채우기 
fill_x = np.linspace(-4, t_crit, 100)
fill_x
fill_y = t.pdf(fill_x, df)
plt.fill_between(fill_x, fill_y, alpha=0.3, color='red')


# x축에 텍스트 추가 (검정통계량과 임계값 추가)
plt.text(t_crit,-0.1,f"{t_crit:.2f}", color='red', ha='center', fontsize=10)
plt.text(t_stat,-0.13,f"{t_stat:.2f}", color='blue', ha='center', fontsize=10)

plt.legend()
plt.title('t-검정 결과 시각화')
plt.xlabel('t값')
plt.ylabel('확률밀도함수')
plt.show()



문제2. (우측 검정)
한 학급의 수학 성적이 평균 70점 이하라고 알려져있다. 
이를 검증하기 위해 이 학급에서 16명을 무작위로 선발하여 시험을 보았더니  
평균이 72점, 표준편차가 5점이었다.  
유의수준 5%에서  수학성적 평균이 70점보다 높다고 하는 주장이 옳다고 할 수 있는가 ?



# 답:

# 1. 가설설정

#귀무가설 (H0) (원래 알려진 정보): 학급의 수학성적 평균은 70점이하이다. 
#  H0:  μ <= 70

#대립가설 (H1) (검증하려는 주장): 학급의 수학성적 평균이 70점보다 높다.
#  H1:  μ > 70 ( 단측검정, 우측검정)

#검정방향: 우측검정(표본 통계량이 기준값보다 큰 방향으로 유의한지를 검정)


#2. 문제에서 주어진 값 설정

n = 16  # 표본크기
mu_sample = 72     #표본평균
sigma =  5      # 표본의 표준편차 
mu_pop =  70   # 모집단 평균 
alpha  =   0.05    # 유의수준
df = n - 1   #  자유도 

# 3. t 검정 수행( 좌측 검정)
#    t = ( X̄ - μ₀ ) / ( s / √n )  # s는 표본의 표준편차 
from  scipy.stats  import   t 

t_stat =( mu_sample - mu_pop ) / ( sigma / np.sqrt(n) )
p_value = 1- t.cdf(t_stat, df )  # 우측검정이므로 누적분포함수(cdf) 사용

# 임계치 계산 
t_crit= t.ppf(1-alpha, df)


# 4. 결과 해석
print('t 검정 통계량' , round(t_stat, 5))
print('임계값 (t_crit)', round(t_crit,5))
print('p_value ', round(p_value,5))

if    t_stat > t_crit:
    print('귀무가설을 기각하고 대립가설이 채택')
    print('학습의 수학성적 평균이 70점보다 높다고 볼 수 있음')
else:
    print('귀무가설 채택- 학습의 수학성적 평균이 70점보다 높다고 보기 어려움')


# 시각화 
from  scipy.stats  import   t 
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df)

plt.figure(figsize=(4,3))
plt.plot(x,y, label='t-분포',color='black')
plt.axvline(t_crit, color='red', linestyle='dashed', label=f"임계값 ({t_crit:.2f})")
plt.axvline(t_stat, color='blue', linestyle='solid', label=f"검정통계량 ({t_stat:.2f})")

#기각역 채우기 
fill_x = np.linspace( t_crit, 4, 100)
fill_y = t.pdf(fill_x, df)
plt.fill_between(fill_x, fill_y, alpha=0.3, color='red', label='기각역')


# x축에 텍스트 추가 (검정통계량과 임계값 추가)
plt.text(t_crit,-0.1,f"{t_crit:.2f}", color='red', ha='center', fontsize=10)
plt.text(t_stat,-0.13,f"{t_stat:.2f}", color='blue', ha='center', fontsize=10)

plt.legend()
plt.title('t-검정 결과 시각화')
plt.xlabel('t값')
plt.ylabel('확률밀도함수')
plt.show()




