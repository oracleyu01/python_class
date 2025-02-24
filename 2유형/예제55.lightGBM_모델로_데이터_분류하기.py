■ 예제55.lightGBM_모델로_데이터_분류하기.py

■  LightGBM 개요

LightGBM은 마이크로소프트에서 개발한 그라디언트 부스팅(Gradient Boosting) 기반의 알고리즘입니다. 
주요 특징은 다음과 같습니다:

■ 1. Leaf-wise (리프 중심) 성장 방식:

전통적인 트리 알고리즘은 보통 **level-wise (수준별)**로 모든 노드를 동시에 분할하는 반면, 
LightGBM은 **리프(leaf)**를 기준으로 가장 큰 손실 감소를 가져오는 노드를 선택하여 분할합니다.

이 방식은 모델의 학습 속도를 빠르게 하고, 복잡한 패턴을 포착할 수 있지만, 
깊은 트리가 형성되면서 과적합(overfitting)의 위험이 커질 수 있습니다.


■ 2. 효율성:

학습 속도: 히스토그램 기반의 알고리즘을 사용하여 연속형 변수의 값을 구간으로 나누어 계산 비용과 메모리 사용량을 줄입니다.

메모리 사용량: 데이터 처리 과정에서 메모리 효율을 높여 대용량 데이터셋 처리에 유리합니다.

경량성: XGBoost 등 다른 부스팅 계열 모델에 비해 연산량과 메모리 사용 측면에서 더 가벼운 편입니다.

■ 3. 부스팅(Boosting) 계열 알고리즘

부스팅은 여러 약한 학습기(weak learner)를 순차적으로 학습시켜 예측 성능을 높이는 앙상블 기법입니다. 

주요 특징은 다음과 같습니다:

 1.  순차적 학습:
    이전 학습기의 오류(잔차)를 보완하는 방향으로 다음 학습기가 학습됩니다. 
    즉, 각 단계마다 모델이 잘못 예측한 데이터에 더 집중하여 학습합니다.

 2. 예시 알고리즘:

    AdaBoost, Gradient Boosting, XGBoost, LightGBM 등이 있으며, 이들은 모두 부스팅 기법을 사용합니다.

 3. Random Forest와의 차이:

      Random Forest는 배깅(bagging) 방식에 기반하며, 다수의 결정 트리를 병렬적으로 학습시켜 예측을 수행합니다. 
        부스팅과 달리 각 모델이 독립적으로 학습됩니다.


문제1. LightGBM의 특징으로 옳은 것은? 
     (2024년 제8회 빅데이터분석기사 필기)

1. 기존 트리 방식과 달리 leaf 중심으로 분기한다
2. XGBoost보다 실행 속도가 느리다
3. 과적합에 취약하다
4. 하이퍼파라미터 튜닝이 불필요하다

정답: 


문제2. 부스팅 계열 알고리즘에 대한 설명으로 옳은 것은?
      (2023년 제7회 빅데이터분석기사 필기)

1. LightGBM은 XGBoost보다 무거운 모델이다
2. LightGBM은 XGBoost보다 가벼운 모델이다
3. Random Forest는 부스팅 계열이다
4. 순차적 학습이 불가능하다

정답: 


문제3. LightGBM의 장점으로 옳지 않은 것은?
     (2023년 제6회 빅데이터분석기사 필기)

1. 학습 속도가 빠르다
2. 메모리 사용량이 적다
3. 대용량 데이터 처리에 효과적이다
4. 과적합에 강하다

정답: 



■ 실습1. 하나의 의사결정트리로만 분류했을때


#1. 데이터 불러오기
import pandas  as  pd

credit = pd.read_csv("c:\\data\\credit.csv") 
#credit.head()

#2. 데이터 확인하기
#credit.shape
#credit.info()  

#3. 결측치 확인하기 
# credit.isnull().sum()

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

#credit.info()

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

x_train, x_test, y_train, y_test = train_test_split( x_scaled, y, test_size=0.2,\
                                                              random_state=42) 

# print(x_train.shape)  # (800, 16)
# print(x_test.shape)   # (200, 16) 
# print(y_train.shape)  # (800 , )
# print(y_test.shape)   # (200,  )

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

#12. 모델 개선 


■ 실습2. lightgbm으로 변경해서 수행하시오 

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



문제1. 아래의 유방암 데이터의 악성 종양과 양성 종양 분류를 lightgbm 으로 변경해서 수행하시오 !


# ■ 실습1. 유방암 환자 분류 knn 모델 생성 

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



