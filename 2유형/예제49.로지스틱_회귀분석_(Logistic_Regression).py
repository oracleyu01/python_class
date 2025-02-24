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


[위의 내용과 관련된 ADSP 기출 유형] 
로지스틱 회귀분석 결과, 연령(Age)변수의 계수가 0.405로 나왔습니다. exp(0.405) = 1.5일 때, 
다음 물음에 답하시오.

연령이 1살 증가할 때, 오즈는 몇 배 증가하는가?
현재 40세인 사람의 오즈가 1.2라고 할 때, 41세가 되었을 때의 오즈는 얼마인가?

답과 풀이:
exp(계수) = 1.5이므로, 연령이 1살 증가할 때 오즈는 1.5배 증가합니다.
현재 오즈 × exp(계수)
= 1.2 × 1.5
= 1.8

따라서 41세일 때의 오즈는 1.8입니다.

문제:
한 연구에서 흡연이 폐암 발병에 미치는 영향을 로지스틱 회귀분석으로 분석했습니다. 
흡연 여부(비흡연=0, 흡연=1) 변수의 계수가 1.386이었습니다. exp(1.386) = 4일 때, 다음 물음에 답하시오.

비흡연자 대비 흡연자의 폐암 발병 오즈는 몇 배인가?
비흡연자의 폐암 발병 오즈가 0.3일 때, 흡연자의 폐암 발병 오즈는 얼마인가?

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



