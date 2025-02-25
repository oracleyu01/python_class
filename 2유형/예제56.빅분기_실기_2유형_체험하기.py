# ▣ 예제56.빅분기_실기_2유형_체험하기.py

시험환경: https://dataq.goorm.io/exam/3/%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EA%B8%B0%EC%82%AC-%EC%8B%A4%EA%B8%B0-%EC%B2%B4%ED%97%98/quiz/4%3Fembed

# ■ 시험환경 만들기1. sklearn 에서 유방암 환자 데이터 불러오기

import pandas as pd
from sklearn.datasets import load_breast_cancer
brst = load_breast_cancer()
x, y = brst.data, brst.target

col = brst.feature_names                 # 컬럼명 불러오기
X = pd.DataFrame(x , columns=col)        # 학습 데이터
y = pd.DataFrame(y, columns=['cancer'])  # 정답 데이터

# cust_id 컬럼을 추가합니다.
X.insert(0, 'cust_id', range(1, 1 + len(X)))  # X.insert(컬럼자리번호, 컬럼명, 데이터 )
X


# ■ 시험환경 만들기2. 훈련 데이터와 테스트 데이터를 생성합니다.

# 훈련 데이터와 테스트 데이터를 분리합니다.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(X, y,test_size=0.2, random_state=1)

# 만든 데이터를 시험환경에 저장합니다.
x_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
x_test.to_csv("X_test.csv", index=False)


# ■ 시험문제 풀기 시작

#1. 데이터 불러오기
import pandas  as  pd

x_train = pd.read_csv("x_train.csv")
x_test = pd.read_csv("x_test.csv")
y_train = pd.read_csv("y_train.csv")

#2. 데이터 살펴보기 

# 2.1 결측치 확인
# print(x_train.isnull().sum())
# print(x_test.isnull().sum())

# 2.2 문자형 데이터가 있는지 확인 
# print(x_train.info())
# print(x_test.info())

#3. 데이터 인코딩하기(문자 --> 숫자)
# 전부 숫자라서 할 필요 없습니다.

#4. 데이터 스켈링하기

# 훈련 데이터 스켈링 
from sklearn.preprocessing  import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)  # 훈련데이터로 계산
x_train_scaled = scaler.transform(x_train)  # 훈련데이터 변환 

# 테스트 데이터 스켈링 
x_test_scaled = scaler.transform(x_test)  # 테스트 데이터 변환 


#5. 모델 생성
from sklearn.ensemble  import RandomForestClassifier

model = RandomForestClassifier(random_state=1)

#6. 모델 훈련
model.fit(x_train_scaled,y_train)

#7. 모델 예측
# 훈련 데이터 예측
train_pred = model.predict(x_train_scaled) 

# 테스트 데이터 예측
pred = model.predict(x_test_scaled) 

#8. 모델 평가 (훈련 데이터에 대해서만) 
from sklearn.metrics  import  accuracy_score

print(accuracy_score(y_train, train_pred))

#9. 테스트 예측결과 제출 
pd.DataFrame({'pred' : pred }).to_csv("result.csv", index=False)

import  pandas  as  pd

result = pd.read_csv("result.csv")
print(result)

# 42분까지 쉬세요



# 😊 문제: 시험환경에 백화점 데이터의 성별 예측 분류 모델을 생성하고 제출하시오



