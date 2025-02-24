■■ 예제53.k-hold 교차검정_답

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


# ■ 실습1. 유방암 환자 분류 knn 모델 생성하는 for loop문으로 최적의 하이퍼 파라미터인 k값을 찾으시오

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

for i  in  range(1,20,2):
    model = KNeighborsClassifier( n_neighbors= i )
    model.fit( x_train, y_train )  
    
    #6. 모델 예측
    result = model.predict(x_test) 
    
    #7. 모델 평가 
    acc = sum( result == y_test ) / len(y_test) * 100
    print( 'k가', i, '일때 정확도 ', acc)


■ 실습2. 위의 k 값을 k fold 교참검정으로 찾으시오 !

#1. 데이터 로드
import pandas as pd
wbcd = pd.read_csv("c:\\data\\wisc_bc_data.csv")
wbcd.head()

#2~3. (데이터 확인, 결측치 확인)

#4. 정규화 작업
from sklearn.preprocessing import MinMaxScaler 
wbcd2 = wbcd.iloc[ : , 2: ] 
scaler = MinMaxScaler()
wbcd2_scaled = scaler.fit_transform(wbcd2)

# 정답 데이터 생성 
y = wbcd.loc[ : , 'diagnosis'].to_numpy()

#5. GridSearch로 최적의 k값 찾기






# 모델 훈련
grid_search.fit(wbcd2_scaled, y)

# 최적 파라미터와 점수 출력
print("최적의 k값:", grid_search.best_params_['n_neighbors'])
print("최고 정확도:", grid_search.best_score_)

#6. 최적의 k값으로 최종 모델 생성 및 평가
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wbcd2_scaled, y, test_size=0.1, random_state=1)

final_model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
final_model.fit(x_train, y_train)
result = final_model.predict(x_test)
test_acc = sum(result == y_test) / len(y_test) * 100
print(f"테스트 데이터 정확도: {test_acc:.2f}%")


문제1.  wine.csv 에서 wine 의 품질을 분류하는 최적의 하이퍼 파라미터 k 값을 k fold 교차검정으로 찾으시오 !
       아래의 코드를 활용하세요. 

#1. 데이터 로드
import pandas as pd
wine = pd.read_csv("c:\\data\\wine2.csv")
wine.head()

#2. 데이터 확인
#wine.shape
#wine.info()

#3. 결측치 확인
#wine.isnull().sum()

#4. 정규화 작업
#wine.describe()  # 기술 통계정보 
from sklearn.preprocessing import MinMaxScaler # 0~1 사이의 데이터로 변환
#wine.head()
wine2 = wine.iloc[ : , 1: ]  # Type 열을 제외한 나머지 특성들 선택
wine2.head()
scaler = MinMaxScaler()  # 설계도로 scaler 라는 제품(객체)를 생성합니다. 
wine2_scaled = scaler.fit_transform(wine2)
wine2_scaled  # 스켈링된 학습 데이터 

# 정답 데이터 생성 
y = wine.loc[ : , 'Type'].to_numpy()
y

# 데이터를 훈련 데이터와 테스트 데이터로 9:1로 분리합니다.
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(wine2_scaled, y, test_size=0.1, random_state=1)
# print(x_train.shape)  # 훈련 데이터
# print(x_test.shape)   # 테스트 데이터
# print(y_train.shape)  # 훈련 데이터의 정답 
# print(y_test.shape)   # 테스트 데이터의 정답    

#5. 모델 훈련
from sklearn.neighbors import KNeighborsClassifier 
for i in range(1,20,2):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)  
    
    #6. 모델 예측
    result = model.predict(x_test) 
    
    #7. 모델 평가 
    acc = sum(result == y_test) / len(y_test) * 100
    print('k가', i, '일때 정확도 ', acc)



