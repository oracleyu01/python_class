■■  예제57.산포도_그래프와_상관계수.py



■ 두 변수간의 관계를 분석할 때 사용하는 분석 방법
  
  1. 수치형 : 두변수가 모두 숫자일 때
  
  - 상관관계 분석:  피어슨 상관계수, 스피어만 상관계수
  - 회귀분석 :  단순 회귀 분석, 다중 회귀분석
  
  2. 범주형 : 두변수가 모두 문자일 때 
  
  - 교차표 분석  :  교차표를 생성
  - 카이제곱 검정 : 카이제곱 검정


■ 상관계수 (Correlation Coefficient)

 1. 정의:

  상관계수는 두 변수 간의 관계의 방향(양/음)과 강도(강한/약한)를 나타내는 통계적 지표입니다.

■ 피어슨 상관계수 (Pearson Correlation Coefficient):

 1.용도:

  연속형(수치형) 데이터 간의 선형 관계를 측정합니다.

 2.범위:

 -1부터 +1까지의 값을 가지며,
 +1: 완벽한 양의 선형 관계
 -1: 완벽한 음의 선형 관계
  0: 선형 관계가 없음을 의미하지만, 비선형 관계가 존재할 수 있음

 3.오해:

  "피어슨 상관계수는 -2에서 +2 사이의 값을 가진다"라는 설명은 잘못된 것입니다.

■ 스피어만 상관계수 (Spearman Correlation Coefficient):

 1.용도:

  순서형(ordinal) 데이터나 정규분포를 따르지 않는 데이터에서 순위에 기반한 상관관계를 측정할 때 사용됩니다.

 2.계산 방식:

  실제 값 대신 각 데이터의 순위를 매긴 후 계산합니다. 즉, 실제 값 자체가 아닌 순위를 사용합니다.

 3.오해:

  "스피어만 상관계수는 실제 값을 사용한다"라는 설명은 옳지 않습니다.

■  산점도 (Scatter Plot)

  1. 정의:

       두 변수 간의 관계를 시각적으로 표현하는 그래프입니다.

  2. 특징:

    x축과 y축에 각각 한 변수씩 배치하여 두 변수의 분포와 관계를 확인합니다.

  3. 관계 확인:

    데이터 포인트들이 직선에 가까운 패턴을 보이면 선형관계가 있다고 해석할 수 있습니다.

  4. 적용 대상:

    수치형 데이터의 관계를 시각적으로 표현하는 데 적합합니다.

   5. 주의:

     범주형 데이터는 수치적 크기나 순서가 없으므로 산점도보다는 
     막대그래프나 원형그래프 등 다른 시각화 도구가 더 적합합니다.



문제1. 상관계수의 해석으로 옳은 것은?
     (2023년 제6회 빅데이터분석기사 필기)

1.피어슨 상관계수는 -2에서 +2 사이의 값을 가진다
2.상관계수가 +1은 완벽한 양의 선형 상관관계를 의미한다
3.상관계수가 0이면 두 변수는 관계가 없다
4.스피어만 상관계수는 실제 값을 사용한다

정답: 


문제2. 산점도에 대한 설명으로 옳지 않은 것은?
      (2022년 제4회 빅데이터분석기사 필기)

1.두 변수 간의 관계를 시각적으로 표현한다
2.x축과 y축에 각각의 변수를 나타낸다.
3.직선 형태면 선형관계가 있다고 본다
4.범주형 데이터 분석에 가장 적합하다

정답: 


문제3. 상관분석 방법으로 옳지 않은 것은?
  (2021년 제2회 빅데이터분석기사 필기)

1. 수치형 데이터는 피어슨 상관계수를 사용한다
2. 순서형 데이터는 스피어만 상관계수를 사용한다
3. 명목형 데이터는 카이제곱 검정을 사용한다
4. 모든 데이터는 피어슨 상관계수로 분석한다

정답: 


😊예제1. 중고차 데이터의 가격과 마일리지, 연식이 얼마나 상관관계가 있는지 확인하시오 !

답:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('c:\\data\\usedcars.csv')

# 수치형 변수들의 상관계수 계산
corr = df[['year', 'price', 'mileage']].corr()
print(corr)

# 히트맵으로 상관계수 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()



😊문제1. 의료비와 상관관계가 높은 변수가 무엇인지 확인하시오 !


답: 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('c:\\data\\insurance.csv')




😊예제2.   주행거리와 가격과의 산포도 그래프를  그리시오 !


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('c:\\data\\usedcars.csv')

# 산포도 그래프 그리기
plt.figure(figsize=(10, 6))
sns.scatterplot(x='mileage', y='price', data=df)

# 추세선 추가 (선형 회귀선)
sns.regplot(x='mileage', y='price', data=df, scatter=False, color='red')

# 그리드 추가
plt.grid(True, linestyle='--', alpha=0.7)

# 그래프 표시
plt.show()

# 상관계수 계산 및 출력
correlation = df['mileage'].corr(df['price'])
print(f"주행거리와 가격의 상관계수: {correlation:.4f}")


😊문제2. 중고차의 연식과 가격과의 산포도 그래프를 그리시오 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('c:\\data\\usedcars.csv')



