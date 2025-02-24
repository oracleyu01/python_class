■■■ 예제54.XGBoost_로_데이터_분류하기.py

■■ 부스팅(Boosting) 기법

설명 그림 : https://cafe.daum.net/oracleoracle/Sq3w/125


■ 기본 개념:

여러 약한 학습기(weak learners)를 순차적으로 학습시켜, 이전 모델이 잘못 예측한 데이터에 가중치를 부여하면서 점점 성능이 향상되는 강한 모델(strong learner)을 만드는 앙상블 기법입니다.

■ 주요 알고리즘:

 1. AdaBoost:

초기 부스팅 알고리즘으로, 잘못 분류된 샘플의 가중치를 증가시켜 다음 모델이 집중하도록 합니다.

 2. XGBoost, LightGBM:

GBM(Gradient Boosting Machine)의 개선판으로, 계산 효율과 성능 면에서 우수하며 여러 최적화 기법(예: 정규화, 트리 가지치기, 병렬 처리 등)을 포함합니다.

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




문제1. XGBoost에 대한 설명으로 옳은 것은?
       (2024년 제8회 빅데이터분석기사 필기)

1. GBM을 개선한 방식이며 GBM보다 속도가 빠르다
2. 병렬처리가 지원되지 않는다
3. 과적합이 자주 발생한다
4. 하이퍼파라미터 최적화가 필요없다

정답: 1


문제2. 부스팅 기법을 사용하는 알고리즘으로 옳지 않은 것은?
      (2023년 제6회 빅데이터분석기사 필기)

1.AdaBoost
2.XGBoost
3.Random Forest
4.LightGBM

정답: 3


문제3. XGBoost의 특징으로 옳은 것은?
     (2022년 제4회 빅데이터분석기사 필기)

1.학습 속도가 느리다
2.과적합되는 경우가 많다
3.병렬 처리를 지원한다
4.하나의 커널함수만 사용한다

정답: 3


■ 예제1.단일 의사결정트리 모델일 때의 코드

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




■ 예제2. xgboost 로 구현했을때

#1. 필요한 라이브러리 불러오기
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

#2. 데이터 불러오기

credit = pd.read_csv("d:\\data\\credit.csv")

#3. 결측치 확인

credit.isnull().sum()

#4. 범주형 데이터 인코딩

label_encoder = LabelEncoder()
categorical_columns = ['checking_balance', 'credit_history', 'purpose', 
                              'savings_balance', 'employment_duration', 'other_credit',
                               'housing', 'job', 'phone', 'default']

for column in categorical_columns:
    credit[column] = label_encoder.fit_transform(credit[column])

#5. 특성과 타겟 분리

X = credit.drop('default', axis=1)
y = credit['default']

#6. 데이터 스케일링

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#7. 훈련 데이터와 테스트 데이터 분리하기 

from sklearn.model_selection import train_test_split

# test_size와 random_state 설정을 유지하면서 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, 
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)  # 클래스 비율 유지

print("Training set shape:", x_train.shape, y_train.shape)
print("Test set shape:", x_test.shape, y_test.shape)

#8. XGBoost 모델 생성 및 GridSearchCV 적용
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 탐색할 하이퍼파라미터 설정
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# 기본 모델 생성
xgb_model = XGBClassifier(random_state=42)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

#9. 모델 훈련 (GridSearch 수행)

grid_search.fit(x_train, y_train)

# 최적 파라미터 출력
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# 최적 모델 저장
best_model = grid_search.best_estimator_

#10. 최적 모델로 예측
# 훈련 데이터 예측
train_result = best_model.predict(x_train)

# 테스트 데이터 예측
result = best_model.predict(x_test)

#11. 모델 평가
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 훈련 데이터 정확도
train_accuracy = accuracy_score(y_train, train_result)
print("Training Accuracy:", train_accuracy)

# 테스트 정확도
test_accuracy = accuracy_score(y_test, result)
print("Test Accuracy:", test_accuracy)

# 상세 평가 지표
print("\nClassification Report:")
print(classification_report(y_test, result))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, result))

Training Accuracy: 0.9677777777777777
Test Accuracy: 0.76

문제1. 위의 머신러닝 모델의 성능을 더 올리시오

답:



