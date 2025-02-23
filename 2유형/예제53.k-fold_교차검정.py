■■ 예제53.k-fold_교차검정.py


그림: https://cafe.daum.net/oracleoracle/Sq8G/83

■ 1. 데이터 분할 및 평가 기법 개요

 모델의 일반화 성능(새로운 데이터에 대한 예측력)을 평가하기 위해 데이터를 어떻게 분할하고 활용하는지가 중요합니다.
 대표적인 기법에는 Holdout 기법과 **교차검증(Cross Validation)**이 있으며, 그 중 K-fold 교차검증은 가장 널리 사용되는 방법 중 하나입니다.

■ 2. Holdout 기법

- 정의:

전체 데이터 집합을 한 번만 임의로 분할하여, 일반적으로 **훈련 데이터(Training Set)**와 **검증(또는 테스트) 데이터(Validation/Test Set)**로 나누는 방법입니다.

- 구성 예시:

훈련 데이터: 전체 데이터의 70~80% 정도를 사용하여 모델을 학습합니다.
검증(또는 테스트) 데이터: 나머지 20~30%를 사용하여 학습된 모델의 성능을 평가합니다.

- 특징:

단순성: 한 번의 분할로 진행되어 이해하기 쉽고, 계산 비용이 적습니다.

- 단점:

데이터 분할에 따라 평가 결과가 크게 달라질 수 있으며, 특정 분할에 따른 편향(bias)이 발생할 수 있습니다.
전체 데이터를 모두 활용하지 못해, 데이터의 불균형이나 대표성 문제로 인해 평가의 신뢰도가 낮아질 수 있습니다.

■ 3. 교차검증 (Cross Validation)

- 정의:

데이터를 여러 부분으로 나눈 후, 여러 번의 학습 및 평가를 통해 모델의 성능을 보다 안정적으로 추정하는 방법입니다.

- 일반적인 과정:

데이터를 여러 개의 부분(폴드)으로 나눕니다.
각 반복(iteration)마다 한 부분을 검증용으로, 나머지를 학습용으로 사용하여 모델을 평가합니다.
모든 반복의 평가 결과를 평균하여 최종 성능을 산출합니다.

■ 4. K-Fold 교차검증

- 정의:

전체 데이터 집합을 동일한 크기를 갖는 **K개의 서브셋(폴드)**으로 나누고, K번의 반복을 통해 평가하는 방법입니다.

- 동작 방식:

각 반복(iteration):

검증 집합: 한 개의 폴드
학습 집합: 나머지 K-1개의 폴드
모든 폴드가 한 번씩 검증 데이터로 사용됩니다.

- 장점:

모든 데이터를 학습과 검증에 골고루 활용하므로, Holdout 방식보다 평가의 신뢰성이 높습니다.
데이터 분할에 의한 우연한 편향을 줄일 수 있습니다.

- 단점:

K번의 모델 학습이 필요하므로, 계산 비용이 증가합니다.
K의 값 선택에 따라 평가 결과가 달라질 수 있습니다.


■ 총정리:

------------------------------------------------------------------------------
특징           | Holdout 기법
               |     - 데이터를 한 번만 분할 (예: 70% 학습, 30% 검증)
               |     - 계산 비용: 낮음
               |     - 평가 신뢰도: 분할에 따라 결과가 달라질 수 있음
               |     - 적용 상황: 데이터가 충분하여 단순 분할로도 신뢰할 수 있는 경우
--------------------------------------------------------------------------------
특징           | K-Fold 교차검증
               |     - 데이터를 K개의 폴드로 나누어, 매 반복마다 한 폴드를 검증용으로 사용
               |     - 계산 비용: 상대적으로 높음 (K번의 학습 필요)
               |     - 평가 신뢰도: 여러 번 평가를 평균하여 보다 안정적인 성능 추정 가능
               |     - 적용 상황: 데이터 양이 적거나, 보다 신뢰도 높은 평가가 필요한 경우
--------------------------------------------------------------------------------------




문제1. K-fold 교차검증에 대한 설명으로 옳은 것은?
      (2024년 제8회 빅데이터분석기사 필기)

1. 폴드의 크기가 작을수록 모델 성능이 떨어진다
2. k개로 나누어진 데이터 셋은 각각 한 번씩만 검증용으로 사용한다
3. 학습과 검증을 k/2번 반복해서 수행한다
4. k-2개 데이터 셋은 학습용으로 사용한다

정답: 


문제2. K-fold CV에 대한 설명 중 옳지 않은 것은?
       (2022년 제4회 빅데이터분석기사 필기)

1.검증, 훈련, 테스트 데이터로 이루어져 있다
2.k=3 이상만 가능하다
3.k개의 균일한 서브셋으로 구성된다
4.k-1개의 부분집합을 학습데이터로 사용한다

정답: 


문제3. 전체 데이터 집합을 동일 크기를 갖는 K개의 부분 집합으로 나누고, 훈련 데이터와 평가 데이터로 나누는 기법은?
      (2021년 제2회 빅데이터분석기사 필기)


1. K-Fold
2. 홀드아웃(Holdout)
3. Dropout
4. Cross Validation

정답: 


■ 실습1. 

예제1. 아이리스 데이터를 k-hold 교차검정으로 

# 필요한 패키지 import
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

# 데이터 불러오기
iris = pd.read_csv("iris2.csv")

# 특성(X)과 타겟(y) 분리
X = iris.drop('Species', axis=1)
y = iris['Species']

# 훈련/테스트 데이터 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

print("훈련 데이터 shape:", X_train.shape)
print("테스트 데이터 shape:", X_test.shape)

# 10-fold 교차 검증 설정
kfold = KFold(n_splits=10, shuffle=True, random_state=123)

# 의사결정 트리 모델 생성
dt_model = DecisionTreeClassifier(random_state=123)

# 교차 검증 수행
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=kfold)
print("\n교차 검증 점수:", cv_scores)
print("평균 교차 검증 점수: {:.3f} (+/- {:.3f})".format(
    cv_scores.mean(), cv_scores.std() * 2
))

# 최종 모델 학습 (전체 훈련 데이터 사용)
dt_model.fit(X_train, y_train)

# 테스트 데이터 예측
test_predictions = dt_model.predict(X_test)

# 테스트 세트 성능 평가
print("\n테스트 세트 성능 평가:")
print(confusion_matrix(y_test, test_predictions))
print("\n분류 보고서:")
print(classification_report(y_test, test_predictions))

# 훈련 데이터 예측
train_predictions = dt_model.predict(X_train)

# 훈련 세트 성능 평가
print("\n훈련 세트 성능 평가:")
print(confusion_matrix(y_train, train_predictions))
print("\n분류 보고서:")
print(classification_report(y_train, train_predictions))

# Wine 데이터셋에 대한 SVM 모델 적용
from sklearn.svm import SVC


문제1. wine 의 품질을 분류하는 머신러닝 모델을 k fold 교차검정으로 구현하시오 !

답: 

# 데이터 불러오기
wine = pd.read_csv("wine2.csv")

