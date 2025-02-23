■■■ 예제54.XGBoost_로_데이터_분류하기.py

■■ 부스팅(Boosting) 기법

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


 - GBM 개선:

XGBoost는 기존의 GBM을 개선한 방식으로, 학습 과정에서 정규화(regularization)와 트리 가지치기(pruning) 등 
다양한 기법을 도입하여 과적합을 줄이고 성능을 향상시킵니다.

- 속도:

GBM보다 빠른 학습 속도를 보이며, 이는 효율적인 메모리 사용과 최적화된 알고리즘 구현 덕분입니다.

- 병렬 처리 지원:


XGBoost는 병렬 처리를 지원하여 다중 코어 환경에서 동시에 여러 트리를 학습할 수 있어, 학습 시간을 단축시킵니다.

- 하이퍼파라미터 최적화:

다양한 하이퍼파라미터(예: 학습률, 최대 깊이, 정규화 계수 등)를 조정해야 하며, 이를 통해 모델의 성능을 최적화할 수 있습니다.

- 과적합 방지:

정규화와 조기 종료(early stopping) 기법 등을 통해 과적합을 효과적으로 제어합니다.




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


예제1. 약한 학습기가 몇개가 있어야 90% 이상의 성능을 보일 수 있나 ?


from math import comb
from math import floor

def ret_err(n, err):
    sum = 0
    
    # floor(n/2)부터 n까지 반복
    for i in range(floor(n/2), n + 1):
        sum += comb(n, i) * (err ** i) * ((1 - err) ** (n - i))
    
    return sum

# 1부터 60까지 반복
for j in range(1, 61):
    err = ret_err(j, 0.4)
    print(f"{j} ---> {1-err:.4f}")
    
    # 정확도가 90% 이상이면 중단
    if (1 - err) >= 0.9:
        break


예제2.단일 의사결정트리 모델일 때의 코드

# 필요한 패키지 import
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np

# 데이터 불러오기
iris = pd.read_csv("d:/data/iris2.csv")

# 특성(X)과 타겟(y) 분리
X = iris.drop('Species', axis=1)
y = iris['Species']

# 훈련/테스트 데이터 분할 (90:10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123
)

print("훈련 데이터 shape:", X_train.shape)  # (135, 4)
print("테스트 데이터 shape:", X_test.shape)  # (15, 4)

# 의사결정 트리 모델 생성
dt_model = DecisionTreeClassifier(random_state=123)

# 10-fold 교차 검증 수행
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=10)
print("\n교차 검증 점수:", cv_scores)
print("평균 교차 검증 점수: {:.3f} (+/- {:.3f})".format(
    cv_scores.mean(), cv_scores.std() * 2
))

# 최종 모델 학습
dt_model.fit(X_train, y_train)

# 테스트 데이터 예측
test_predictions = dt_model.predict(X_test)

# 테스트 세트 성능 평가
print("\n테스트 세트 성능 평가:")
print("혼동 행렬:")
print(confusion_matrix(y_test, test_predictions))
print("\n분류 보고서:")
print(classification_report(y_test, test_predictions))
print(f"테스트 데이터 정확도: {accuracy_score(y_test, test_predictions):.3f}")

# 훈련 데이터 예측
train_predictions = dt_model.predict(X_train)

# 훈련 세트 성능 평가
print("\n훈련 세트 성능 평가:")
print("혼동 행렬:")
print(confusion_matrix(y_train, train_predictions))
print("\n분류 보고서:")
print(classification_report(y_train, train_predictions))
print(f"훈련 데이터 정확도: {accuracy_score(y_train, train_predictions):.3f}")


예제3. 배깅으로 구현했을때 

# 필요한 패키지 import
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np

# 데이터 불러오기
iris = pd.read_csv("d:/data/iris2.csv")

# 특성(X)과 타겟(y) 분리
X = iris.drop('Species', axis=1)
y = iris['Species']

# 훈련/테스트 데이터 분할 (90:10)
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.1, random_state=123
)

print("훈련 데이터 shape:", X_train.shape)
print("테스트 데이터 shape:", X_test.shape)

# 배깅 모델 생성 (25개의 의사결정나무)
bagging_model = BaggingClassifier(
   estimator=DecisionTreeClassifier(),
   n_estimators=25,
   random_state=123
)

# 10-fold 교차 검증 수행
cv_scores = cross_val_score(bagging_model, X_train, y_train, cv=10)
print("\n교차 검증 점수:", cv_scores)
print("평균 교차 검증 점수: {:.3f} (+/- {:.3f})".format(
   cv_scores.mean(), cv_scores.std() * 2
))

# 최종 모델 학습
bagging_model.fit(X_train, y_train)

# 테스트 데이터 예측
test_predictions = bagging_model.predict(X_test)

# 테스트 세트 성능 평가
print("\n테스트 세트 성능 평가:")
print("혼동 행렬:")
print(confusion_matrix(y_test, test_predictions))
print("\n분류 보고서:")
print(classification_report(y_test, test_predictions))
print(f"테스트 데이터 정확도: {accuracy_score(y_test, test_predictions):.3f}")

# 훈련 데이터 예측
train_predictions = bagging_model.predict(X_train)

# 훈련 세트 성능 평가
print("\n훈련 세트 성능 평가:")
print("혼동 행렬:")
print(confusion_matrix(y_train, train_predictions))
print("\n분류 보고서:")
print(classification_report(y_train, train_predictions))
print(f"훈련 데이터 정확도: {accuracy_score(y_train, train_predictions):.3f}")

# 모델 정보 출력
print("\n모델 정보:")
print(f"기본 분류기: Decision Tree")
print(f"배깅 분류기 개수: 25")
print(f"교차 검증 평균 정확도: {cv_scores.mean():.3f}")

예제4. 부스팅으로 구현했을때

# 필요한 패키지 import
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
iris = pd.read_csv("d:/data/iris2.csv")

# 특성(X)과 타겟(y) 분리
X = iris.drop('Species', axis=1)
y = iris['Species']

# 훈련/테스트 데이터 분할 (90:10)
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.1, random_state=123
)

print("훈련 데이터 shape:", X_train.shape)
print("테스트 데이터 shape:", X_test.shape)

# GridSearch를 위한 파라미터 그리드 설정
param_grid = {
   'n_estimators': [100, 150, 200, 250],  # 트리의 개수
   'max_depth': [3, 4, 5],                # 트리의 깊이
   'learning_rate': [0.1],                # 학습률
   'min_samples_leaf': [10]               # 말단 노드의 최소 관측치 수
}

# GBM 모델 생성
gbm = GradientBoostingClassifier(random_state=123)

# GridSearchCV로 최적 파라미터 탐색 (10-fold 교차검증)
grid_search = GridSearchCV(
   estimator=gbm,
   param_grid=param_grid,
   cv=10,
   n_jobs=-1,
   verbose=0
)

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적 파라미터 출력
print("\n최적 파라미터:")
print(grid_search.best_params_)

# 최적 모델 가져오기
best_model = grid_search.best_estimator_

# 테스트 데이터 예측
test_predictions = best_model.predict(X_test)

# 테스트 세트 성능 평가
print("\n테스트 세트 성능 평가:")
print("혼동 행렬:")
print(confusion_matrix(y_test, test_predictions))
print("\n분류 보고서:")
print(classification_report(y_test, test_predictions))
print(f"테스트 데이터 정확도: {accuracy_score(y_test, test_predictions):.3f}")

# 훈련 데이터 예측
train_predictions = best_model.predict(X_train)

# 훈련 세트 성능 평가
print("\n훈련 세트 성능 평가:")
print("혼동 행렬:")
print(confusion_matrix(y_train, train_predictions))
print("\n분류 보고서:")
print(classification_report(y_train, train_predictions))
print(f"훈련 데이터 정확도: {accuracy_score(y_train, train_predictions):.3f}")

# 변수 중요도 시각화
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.title('Variable Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

예제5. xgboost 로 구현했을때

# 필요한 패키지 import
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 데이터 불러오기
iris = pd.read_csv("d:/data/iris2.csv")

# 특성(X)과 타겟(y) 분리
X = iris.drop('Species', axis=1)
y = iris['Species']

# 레이블 인코딩 (Species를 숫자로 변환)
y = pd.factorize(y)[0]

# 훈련/테스트 데이터 분할 (90:10)
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.1, random_state=1
)

print("훈련 데이터 shape:", X_train.shape)  # (135, 4)
print("테스트 데이터 shape:", X_test.shape)  # (15, 4)

# XGBoost 데이터셋 생성
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# 하이퍼파라미터 설정
params = {
   'objective': 'multi:softmax',  # 다중 클래스 분류
   'num_class': 3,               # 클래스 수
   'max_depth': 3,              # 트리 최대 깊이
   'eta': 0.3,                  # 학습률
}

# 모델 학습
xgb_model = xgb.train(
   params=params,
   dtrain=dtrain,
   num_boost_round=100,         # 부스팅 반복 횟수
   evals=[(dtest, 'eval'), (dtrain, 'train')],
   early_stopping_rounds=10,    # 10라운드 동안 성능 개선이 없으면 중단
   verbose_eval=1              # 학습 과정 출력
)

# 예측
predictions = xgb_model.predict(dtest)
accuracy = accuracy_score(y_test, predictions)
print(f"\nXGBoost 정확도: {accuracy:.4f}")

# 변수 중요도 출력 (선택사항)
importance = xgb_model.get_score(importance_type='gain')
print("\n변수 중요도:")
for key, value in importance.items():
   print(f"{key}: {value}")

# 특성 중요도 시각화 (선택사항)
import matplotlib.pyplot as plt

xgb.plot_importance(xgb_model)
plt.title('Feature Importance')
plt.show()

문제1. wine2.csv 를 xboost 로 분류하시오 !


