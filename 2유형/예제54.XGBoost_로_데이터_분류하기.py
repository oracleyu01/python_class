■■■ 예제54.XGBoost_로_데이터_분류하기.py

■■ 부스팅(Boosting) 기법

설명 그림 : https://cafe.daum.net/oracleoracle/Sq3w/125

■ 기본 개념:

여러 약한 학습기(weak learners)를 순차적으로 학습시켜, 이전 모델이 잘못 예측한 데이터에 가중치를 부여하면서 
점점 성능이 향상되는 강한 모델(strong learner)을 만드는 앙상블 기법입니다.

■ 주요 알고리즘:

 1. AdaBoost:

  초기 부스팅 알고리즘으로, 잘못 분류된 샘플의 가중치를 증가시켜 다음 모델이 집중하도록 합니다.

 2. XGBoost, LightGBM:

  GBM(Gradient Boosting Machine)의 개선판으로, 계산 효율과 성능 면에서 우수하며 
  여러 최적화 기법(예: 정규화, 트리 가지치기, 병렬 처리 등)을 포함합니다.

■ 비교 대상:

 1.랜덤 포레스트(Random Forest):

    부스팅이 아닌 배깅(Bagging) 기반 앙상블 기법으로, 여러 결정 트리를 독립적으로 학습시킨 후 다수결 방식으로 예측합니다.

 2. XGBoost (Extreme Gradient Boosting)

  - 2.1 GBM 개선:

      XGBoost는 기존의 GBM을 개선한 방식으로, 학습 과정에서 정규화(regularization)와 트리 가지치기(pruning) 등  
      다양한 기법을 도입하여 과적합을 줄이고 성능을 향상시킵니다.

  - 2.2 속도 향상:

      GBM보다 빠른 학습 속도를 보이며, 이는 효율적인 메모리 사용과 최적화된 알고리즘 구현 덕분입니다.

  - 2.3 병렬 처리 지원:


     XGBoost는 병렬 처리를 지원하여 다중 코어 환경에서 동시에 여러 트리를 학습할 수 있어, 학습 시간을 단축시킵니다.

  - 2.4  하이퍼파라미터 최적화:

     다양한 하이퍼파라미터(예: 학습률, 최대 깊이, 정규화 계수 등)를 조정해야 하며, 이를 통해 모델의 성능을 최적화할 수 있습니다.

  - 2.5 과적합 방지:

     정규화와 조기 종료(early stopping) 기법 등을 통해 과적합을 효과적으로 제어합니다.

     
■ xgboost 의 이해하기 쉬운예:
====================================
      XGBoost  예측 과정 비교
====================================

[모델 정보: 가격 예측]
------------------------------------
실제 가격         : 5억원
평수             : 32평
층수             : 13층
지하철역까지 거리 : 500m (역세권)
건축연도         : 2018년
학군             : 상위 10%
주차             : 세대당 1.2대

[XGBoost 예측 과정]
------------------------------------

1단계 - 평수 그룹 나누기
---------------------
• 30평 미만  : 3.5억원
• 30-40평    : 4억원     ← 우리 아파트 (32평)
• 40평 이상 : 4.8억원

2단계 - 역세권 반영 (모든 평수 그룹)
---------------------
[30평 미만 그룹]
  - 역세권   : +7000만원  → 3.5억원 + 7000만원 = 4.2억원
  - 비역세권 : +3000만원  → 3.5억원 + 3000만원 = 3.8억원

[30-40평 그룹]
  - 역세권   : +7000만원  → 4억원 + 7000만원 = 4.7억원  ← 우리 아파트 (역세권)
  - 비역세권 : +3000만원  → 4억원 + 3000만원 = 4.3억원

[40평 이상 그룹]
  - 역세권   : +7000만원  → 4.8억원 + 7000만원 = 5.5억원
  - 비역세권 : +3000만원  → 4.8억원 + 3000만원 = 5.1억원

3단계 - 층수 반영 (모든 그룹 내에서)
---------------------
층수에 따른 추가 조정:
  - 10층 이상: +2000만원
  - 10층 미만: +1000만원

[세부 조정]
• 30평 미만 그룹:
    - 역세권, 10층 이상: 4.2억원 + 2000만원 = 4.4억원
    - 역세권, 10층 미만: 4.2억원 + 1000만원 = 4.3억원
    - 비역세권, 10층 이상: 3.8억원 + 2000만원 = 4.0억원
    - 비역세권, 10층 미만: 3.8억원 + 1000만원 = 3.9억원

• 30-40평 그룹:
    - 역세권, 10층 이상: 4.7억원 + 2000만원 = 4.9억원   ← 우리 아파트 (최종 예측)
    - 역세권, 10층 미만: 4.7억원 + 1000만원 = 4.8억원
    - 비역세권, 10층 이상: 4.3억원 + 2000만원 = 4.5억원
    - 비역세권, 10층 미만: 4.3억원 + 1000만원 = 4.4억원

• 40평 이상 그룹:
    - 역세권, 10층 이상: 5.5억원 + 2000만원 = 5.7억원
    - 역세권, 10층 미만: 5.5억원 + 1000만원 = 5.6억원
    - 비역세권, 10층 이상: 5.1억원 + 2000만원 = 5.3억원
    - 비역세권, 10층 미만: 5.1억원 + 1000만원 = 5.2억원



문제1. XGBoost에 대한 설명으로 옳은 것은?
       (2024년 제8회 빅데이터분석기사 필기)

1. GBM을 개선한 방식이며 GBM보다 속도가 빠르다
2. 병렬처리가 지원되지 않는다
3. 과적합이 자주 발생한다
4. 하이퍼파라미터 최적화가 필요없다

정답: 

문제2. 부스팅 기법을 사용하는 알고리즘으로 옳지 않은 것은?
      (2023년 제6회 빅데이터분석기사 필기)

1.AdaBoost
2.XGBoost
3.Random Forest
4.LightGBM

정답: 

문제3. XGBoost의 특징으로 옳은 것은?
     (2022년 제4회 빅데이터분석기사 필기)

1.학습 속도가 느리다
2.과적합되는 경우가 많다
3.병렬 처리를 지원한다
4.하나의 커널함수만 사용한다

정답: 

■ 실습1.단일 의사결정트리 모델일 때의 코드

  독일은행 데이터(credit.csv) 를 가지고 의사 결정트리 모델을 생성하시오!
  채무를 불이행 사람들을 예측하는 머신러닝 모델을 생성 하시오!

#1. 데이터 불러오기
import pandas  as  pd

credit = pd.read_csv("d:\\data\\credit.csv") 
#credit.head()

#2. 데이터 확인하기
#credit.shape
#credit.info()  

#3. 결측치 확인하기 
credit.isnull().sum()

#4. 범주형 데이터를 숫자형으로 인코딩 하기 (★)
#credit.info()

from sklearn.preprocessing import  LabelEncoder

label_encoder = LabelEncoder()

credit['checking_balance'] =label_encoder.fit_transform(credit.loc[ : , 'checking_balance'])
credit['credit_history'] =label_encoder.fit_transform(credit.loc[ : , 'credit_history'])
credit['purpose'] =label_encoder.fit_transform(credit.loc[ : , 'purpose'])
credit['savings_balance'] =label_encoder.fit_transform(credit.loc[ : , 'savings_balance'])
credit['employment_duration'] =label_encoder.fit_transform(credit.loc[ : , 'employment_duration'])
credit['other_credit'] =label_encoder.fit_transform(credit.loc[ : , 'other_credit'])
credit['housing'] =label_encoder.fit_transform(credit.loc[ : , 'housing'])
credit['job'] =label_encoder.fit_transform(credit.loc[ : , 'job'])
credit['phone'] =label_encoder.fit_transform(credit.loc[ : , 'phone'])
credit['default'] =label_encoder.fit_transform(credit.loc[ : , 'default'])

credit.info()

#5. 종속변수와 독립변수 분리하기 
x = credit.drop('default', axis=1)  # 독립변수 
y = credit.loc[ : , 'default']
y.value_counts()  # 1 이 관심범주(채무불이행자), 0 이 비관심범주(채무이행자)

#6. 데이터 스켈링 
from  sklearn.preprocessing  import  MinMaxScaler 
scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

#7. 훈련 데이터와 테스트 데이터 분리하기 
from sklearn.model_selection  import  train_test_split

x_train, x_test, y_train, y_test = train_test_split( x_scaled, y, test_size=0.1,\
                                                              random_state=42) 

print(x_train.shape)  # (800, 16)
print(x_test.shape)   # (200, 16) 
print(y_train.shape)  # (800 , )
print(y_test.shape)   # (200,  )

#8. 모델 생성
from  sklearn.tree  import  DecisionTreeClassifier 

credit_model = DecisionTreeClassifier()


#8. 모델 생성
from  sklearn.tree  import  DecisionTreeClassifier 

credit_model = DecisionTreeClassifier(random_state=42)

#9. 모델 훈련
credit_model.fit(x_train, y_train)  


#10. 모델 예측
# 훈련 데이터 예측 
train_result = credit_model.predict(x_train)

# 테스트 데이터 예측
result = credit_model.predict(x_test) 

#11. 모델 평가
# 훈련 데이터 정확도
print( sum(train_result==y_train) / len(y_train) )  # 1.0

# 테스트 정확도 
print( sum( result == y_test ) / len(y_test) )  # 0.735




■ 실습2. xgboost 로 구현했을때

#1. 데이터 불러오기
import pandas as pd
credit = pd.read_csv("c:\\data\\credit.csv") 

#2. 데이터 확인하기
#credit.shape
#credit.info()  

#3. 결측치 확인하기 
# credit.isnull().sum()

#4. 범주형 데이터를 숫자형으로 인코딩 하기 (★)
#credit.info()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
credit['checking_balance'] =label_encoder.fit_transform(credit.loc[ : , 'checking_balance'])
credit['credit_history'] =label_encoder.fit_transform(credit.loc[ : , 'credit_history'])
credit['purpose'] =label_encoder.fit_transform(credit.loc[ : , 'purpose'])
credit['savings_balance'] =label_encoder.fit_transform(credit.loc[ : , 'savings_balance'])
credit['employment_duration'] =label_encoder.fit_transform(credit.loc[ : , 'employment_duration'])
credit['other_credit'] =label_encoder.fit_transform(credit.loc[ : , 'other_credit'])
credit['housing'] =label_encoder.fit_transform(credit.loc[ : , 'housing'])
credit['job'] =label_encoder.fit_transform(credit.loc[ : , 'job'])
credit['phone'] =label_encoder.fit_transform(credit.loc[ : , 'phone'])
credit['default'] =label_encoder.fit_transform(credit.loc[ : , 'default'])
#credit.info()

#5. 종속변수와 독립변수 분리하기 
x = credit.drop('default', axis=1)  # 독립변수 
y = credit.loc[ : , 'default']
y.value_counts()  # 1 이 관심범주(채무불이행자), 0 이 비관심범주(채무이행자)

#6. 데이터 스켈링 
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

#7. 훈련 데이터와 테스트 데이터 분리하기 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2,\
                                                             random_state=42) 

#8. 모델 생성 및 GridSearch





#9. 모델 훈련
grid_search.fit(x_train, y_train)

# 최적 파라미터 출력
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 최적 모델 저장
best_model = grid_search.best_estimator_

#10. 모델 예측
# 훈련 데이터 예측 
train_result = best_model.predict(x_train)
# 테스트 데이터 예측
result = best_model.predict(x_test) 

#11. 모델 평가
# 훈련 데이터 정확도
print("훈련 데이터 정확도:", sum(train_result==y_train) / len(y_train))
# 테스트 정확도 
print("테스트 데이터 정확도:", sum(result == y_test) / len(y_test))

결과:

Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 200}
Best score: 0.74375
훈련 데이터 정확도: 0.94625
테스트 데이터 정확도: 0.795

문제1. 유방암 데이터의 양성과 악성을 xgboost 로 변경해서 수행하시오 !


#1. 데이터 로드
import pandas  as  pd

wbcd = pd.read_csv("c:\\data\\wisc_bc_data.csv")
wbcd.head()

#2. 데이터 확인
#wbcd.shape
#wbcd.info()
#3. 결측치 확인
#wbcd.isnull().sum()

#4. 정규화 작업
#wbcd.describe()  # 기술 통계정보 

from  sklearn.preprocessing import MinMaxScaler # 0~1 사이의 데이터로 변환
#wbcd.head()
wbcd2 = wbcd.iloc[ : , 2: ] 
wbcd2.head()

scaler = MinMaxScaler()  # 설계도로 scaler 라는 제품(객체)를 생성합니다. 
wbcd2_scaled = scaler.fit_transform(wbcd2)
wbcd2_scaled  # 스켈링된 학습 데이터 

# 정답 데이터 생성 
y = wbcd.loc[  : , 'diagnosis'].to_numpy()
y

# 데이터를 훈련 데이터와 테스트 데이터로 9 :1 로 분리합니다.
from  sklearn.model_selection  import  train_test_split 

x_train, x_test, y_train, y_test = train_test_split( wbcd2_scaled, y, test_size=0.1, random_state=1)
# print(x_train.shape)  # 훈련 데이터
# print(x_test.shape)   # 테스트 데이터
# print(y_train.shape)  # 훈련 데이터의 정답 
# print(y_test.shape)   # 테스트 데이터의 정답    

#5. 모델 훈련
from  sklearn.neighbors   import   KNeighborsClassifier 

model = KNeighborsClassifier( n_neighbors=11 )
model.fit( x_train, y_train )  

#6. 모델 예측
result = model.predict(x_test) 

#7. 모델 평가 
acc = sum( result == y_test ) / len(y_test) * 100
print(acc)

답:


#7. 모델 평가 
acc = sum(result == y_test) / len(y_test) * 100
print("Test Accuracy: %.2f%%" % acc)


