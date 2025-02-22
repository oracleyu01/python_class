# ■ 예제49.로지스틱_회귀분석_(Logistic_Regression).py

## 1. 기본 개념

- 로지스틱 회귀분석은 종속변수가 범주형(이진 분류)인 경우 사용하는 분석 방법입니다.
- 독립변수들의 선형 결합을 통해 사건 발생 확률을 예측합니다.
- 일반 선형 회귀분석과 달리 종속변수가 0과 1 사이의 확률값을 가지도록 로짓 변환을 사용합니다.

## 2. 핵심 용어

### 2.1 오즈(Odds)
- 실패확률(1-p) 대비 성공확률(p)의 비율
- 수식: Odds = p/(1-p)

관련 그림: https://cafe.daum.net/oracleoracle/Sq8G/66

### 2.2 로짓(Logit)
- 오즈에 자연로그를 취한 값
- 수식: Logit = ln(odds) = ln(p/(1-p))

### 2.3 exp(계수)
- 해당 독립변수가 한 단위 증가할 때 오즈가 증가하는 배수

## 3. 주요 특징

- 검정방법: F검정이 아닌 카이제곱 검정을 사용
- 이진 분류에 적합한 분석 방법으로, 선형 회귀분석과 달리 확률을 예측
- 다른 이진 분류 방법들
  - 서포트 벡터 머신
  - 의사결정나무

## 4. 결과 해석

- 독립변수의 계수를 지수화(exp)하면 오즈비(odds ratio)를 구할 수 있음
- 오즈비는 다른 변수가 고정된 상태에서 해당 변수가 한 단위 증가할 때 성공 확률이 몇 배 증가하는지를 나타냄
- 계수 해석
  - 양수 계수: 성공 확률 증가
  - 음수 계수: 성공 확률 감소

문제1. (2023년 ADSP 35회) 로지스틱 회귀분석에 대한 설명으로 옳지 않은 것은?
① 오즈란 이진 분류에서 실패할 확률 대비 성공할 확률을 의미한다
② 로지스틱 회귀분석의 종속변수는 범주형이다
③ 로지스틱 회귀분석의 검정방법으로 F검정을 사용한다
④ 독립변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측한다


문제2. (2024년 ADSP 42회) 로지스틱 회귀모형에서 exp(x1)의 의미로 가장 적절한 것은?

① 나머지 변수가 주어질 때 x1이 한 단위 증가할 때마다 실패확률이 증가하는 비율
② 나머지 변수가 주어질 때 x1이 한 단위 증가할 때마다 성공의 오즈가 증가하는 배수
③ x1이 한 단위 증가할 때마다 전체 확률이 증가하는 비율
④ x1이 한 단위 증가할 때마다 로그 확률이 증가하는 비율 



문제3. (2023년 ADSP 33회) 다음 중 이진 분류 문제를 해결하기 위한 방법으로 가장 적절하지 않은 것은?

① 로지스틱 회귀분석
② 서포트 벡터 머신
③ 선형 회귀분석
④ 의사결정나무 



■ 로지스틱 회귀와 서포트 백터 머신을 이용한 분류

#▣ 로직스틱 회귀일때 분류 실습

#1. 데이터 생성 
#2. 모델생성
#3. 모델훈련
#4. 분류 시각화
#5. 모델평가 

답:



예제.  iris2.csv 의 데이터를 서포트 백터 머신으로 분류하시오 ! 

#1. 데이터 불러오기
import pandas as pd

iris=pd.read_csv("c:\\data\\iris2.csv")

#2. 결측치 확인
#iris_df.isnull().sum()

#3. 독립변수와 종속변수 분리
x = iris.iloc[:, 0:4]  # 독립변수
y = iris.iloc[:, 4]    # 종속변수

#4. 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)   # 정규화 계산
x2 = scaler.transform(x)  # 계산한 내용으로 데이터 정규화

#5. 훈련과 테스트 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.1, random_state=1)

#6. 모델 생성
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000)  # 최대 반복 횟수 설정

#7. 모델 훈련
logistic_model.fit(x_train, y_train)

#8. 모델 예측
result = logistic_model.predict(x_test)
result

#9. 모델 평가
accuracy = sum(result == y_test) / len(y_test)
print(f"모델 정확도: {accuracy:.4f}")

#10. 모델 성능 상세 평가 (추가)
from sklearn.metrics import classification_report
print("\n분류 리포트:")
print(classification_report(y_test, result))

문제1. 위의 로지스틱 회귀 모델의 성능을 더 올리시오 !


답:



