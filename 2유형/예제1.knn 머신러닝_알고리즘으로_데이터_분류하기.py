
▩ 예제1.knn 머신러닝_알고리즘으로_데이터_분류하기.py

# 1. knn 정의 ?  

k 개의 이웃한 데이터를 기반으로 대상을 분류/회귀하는 지도학습 알고리즘입니다 
새로운 데이터가 주어졌을때 가장 가까운 이웃 데이터들을 참조하여
결정하는 방식으로 작동합니다.  

#2. 주요특징 ?

 1. 비모수적 방법 :  데이터의 분포에 대한 특별한 가정이 필요없음
 2. 인스턴스 기반 학습 : 별도의 모델 생성 없이 데이터를 그대로 활용 
 3. 거리기반 분류 :  데이터 간의 거리를 측정해서 가장 가까운 이웃을 찾아 분류 

#3. k 값의 영향 ?

 1. k 값이 작을때 :  훈련 데이터만 잘 맞추는 과대적합(overfitting) 위험이 있습니다.
 2. k 값이 클때  :  노이즈에 강하지만 과소적합(underfitting) 발생 가능

#4. knn의 데이터 전처리 ?

  주어진 값의 범위가 크게 다르면 표준화나 정규화를 통한 스켈링 필수입니다. 


문제1. KNN에 대한 설명으로 틀린 것은?  (2015년 8월)

1. 인스턴스 러닝기법이다
2. K값이 클수록 과대적합(Overfitting)문제가 발생한다
3. 가까운 것으로 군집하는 것이다
4. K는 가까운 이웃의 개수를 의미한다

정답:   2


해설: K값이 클수록 과소적합(Underfitting)이 발생하며, K값이 작을수록 과대적합(Overfitting)이 발생합니다.

 

문제2. KNN 알고리즘 수행 시 고려사항으로 가장 적절하지 않은 것은?

1. 데이터 정규화가 필요하다
2. K값이 작을수록 지역적 특성을 잘 반영한다
3. 범주형 변수는 수치화하여 사용한다
4. 학습 데이터가 많을수록 정확도가 떨어진다

정답:  4

해설: KNN은 학습 데이터가 많을수록 더 정확한 예측이 가능합니다. 
다만 계산량이 증가하여 처리 시간이 길어질 수 있습니다. 
학습 데이터가 많을수록 정확도가 떨어진다는 것은 잘못된 설명입니다.

■ 실습

# ☆ 첫번째 실습 (간단한 데이터 분류)

#1. 데이터 생성
from sklearn.datasets  import  make_blobs
x, y = make_blobs(centers=2, random_state=8) 
# centers=2 로 하면 클래스가 2개가 생성
# random_state=8 는 R 에서 set.seed(8) 과 똑같습니다. 

#2. 데이터 시각화
import  mglearn  
import  matplotlib.pyplot as plt
type(x) # numpy 배열( 행렬 계산을 빠르게 하기 위한 모듈) 
mglearn.discrete_scatter( x[ :, 0], x[ :, 1], y )

#3. 모델 생성 
from  sklearn.neighbors  import  KNeighborsClassifier 

model = KNeighborsClassifier(n_neighbors=5)  # 기본 이웃수를 5개 설정 
model

#4. 모델 훈련 
model.fit(x,y)  #  model.fit(독립변수, 종속변수)

#5. 분류 시각화
mglearn.plots.plot_2d_separator(model, x)  # 결정경계선 출력
mglearn.discrete_scatter( x[ :, 0], x[ :, 1], y )  # 기존 데이터를 산포도 그래프로 출력

#6. 모델 평가 
model.score( x, y )

# ☆ 두번째 실습 (k 값에 따른 정확도 차이 실습)

#1. 데이터 생성 (좀더 어렵게 구성) 

from sklearn.datasets  import  make_blobs
x, y = make_blobs(centers=6, random_state=8) 
y = y % 2 

# centers=2 로 하면 클래스가 2개가 생성
# random_state=8 는 R 에서 set.seed(8) 과 똑같습니다. 
# y = y % 2  로 인해서 0, 1, 2, 3 이 0, 1 이 됩니다. 

#2. 데이터 시각화
import  mglearn  
import  matplotlib.pyplot as plt
type(x) # numpy 배열( 행렬 계산을 빠르게 하기 위한 모듈) 
mglearn.discrete_scatter( x[ :, 0], x[ :, 1], y )
plt.xlabel("feature0")
plt.ylabel("feature1")

#3. 모델 생성 
from  sklearn.neighbors  import  KNeighborsClassifier 

model5 = KNeighborsClassifier(n_neighbors=5)  # 기본 이웃수를 5개 설정 
model5

#4. 모델 훈련 
model5.fit(x,y)  #  model5.fit(독립변수, 종속변수)

#5. 분류 시각화
mglearn.plots.plot_2d_separator(model5, x, fill=True, alpha=0.5, cm=plt.cm.Reds)  
mglearn.discrete_scatter( x[ :, 0], x[ :, 1], y )  # 기존 데이터를 산포도 그래프로 출력

#6. 모델 평가 
model5.score( x, y )

문제1.  k값을 줄여서 정확도를 더 올리시오 !


#1. 데이터 생성 (좀더 어렵게 구성) 

from sklearn.datasets  import  make_blobs
x, y = make_blobs(centers=6, random_state=8) 
y = y % 2 

# centers=2 로 하면 클래스가 2개가 생성
# random_state=8 는 R 에서 set.seed(8) 과 똑같습니다. 
# y = y % 2  로 인해서 0, 1, 2, 3 이 0, 1 이 됩니다. 

#2. 데이터 시각화
import  mglearn  
import  matplotlib.pyplot as plt
type(x) # numpy 배열( 행렬 계산을 빠르게 하기 위한 모듈) 
mglearn.discrete_scatter( x[ :, 0], x[ :, 1], y )
plt.xlabel("feature0")
plt.ylabel("feature1")


#3. 모델 생성 
from  sklearn.neighbors  import  KNeighborsClassifier 

model1 = KNeighborsClassifier(n_neighbors=1)  # 기본 이웃수를 5개 설정 
model1

#4. 모델 훈련 
model1.fit(x,y)  #  model5.fit(독립변수, 종속변수)

#5. 분류 시각화
mglearn.plots.plot_2d_separator(model1, x, fill=True, alpha=0.5, cm=plt.cm.Reds)  # 결정경계선 출력
mglearn.discrete_scatter( x[ :, 0], x[ :, 1], y )  # 기존 데이터를 산포도 그래프로 출력

#6. 모델 평가 
model1.score( x, y )

k 값을 1로 했더니 전부 다 분류해냈습니다. 
훈련 데이터는 100% 분류했지만 테스트 데이터는 잘못 분류할 수 있습니다.
이를 과대적합(overfitting) 이라고 합니다. 

■ 실습1. 유방암 환자 분류 knn 모델 생성 

데이터셋: wisc_bc_data.csv  

#1. 데이터 로드
#2. 데이터 확인
#3. 결측치 확인
#4. 정규화 작업
#5. 모델 훈련
#6. 모델 예측
#7. 모델 평가 
#8. 모델 성능 개선

# 구현:

#1. 데이터 로드
import pandas  as  pd

wbcd = pd.read_csv("d:\\data\\wisc_bc_data.csv")
wbcd.head()

#2. 데이터 확인
wbcd.shape  # (569, 32)
wbcd.info()

#3. 결측치 확인
wbcd.isnull().sum()

#4. 정규화 작업
wbcd.describe()  # 기술 통계정보 

from  sklearn.preprocessing import MinMaxScaler # 0~1 사이의 데이터로 변환

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
print(x_train.shape)  # 훈련 데이터 (512, 30)
print(x_test.shape)   # 테스트 데이터 (57, 30)
print(y_train.shape)  # 훈련 데이터의 정답  (512,)
print(y_test.shape)   # 테스트 데이터의 정답     (57,)                                                    

#5. 모델 훈련
from  sklearn.neighbors   import   KNeighborsClassifier 

model = KNeighborsClassifier( n_neighbors=5 )
model.fit( x_train, y_train )  

#6. 모델 예측
result = model.predict(x_test) 
result

#7. 모델 평가 
sum( result == y_test ) / len(y_test) * 100  # 98.24561403508771

# 이원 교차표 출력 
from  sklearn.metrics  import  confusion_matrix 

cm = confusion_matrix( y_test, result )

import  seaborn  as  sns  # 시각화 툴 
import  matplotlib.pyplot  as  plt  # 시각화 툴 

plt.figure( figsize=(4, 2) )  # 그래프 사이즈 지정 

sns.heatmap( cm , annot=True, fmt='d', cmap='Reds',
                  xticklabels=['B', 'M'], yticklabels=['B','M']) 

# annot=True 는 그래프 안에 숫자를 기입
# fmt='d' 는 숫자를 정수형으로 출력
# cmap='Reds' 는 색깔, Greens, viridis, plasma 

plt.xlabel('predict')
plt.ylabel('real')
plt.show()     # 그래프 시각화 

# 이원 교차표 설명: https://cafe.daum.net/oracleoracle/Sq8G/49

#8. 모델 성능 개선

※ 모델 성능 개선 코드를 작성하려면 for loop 문을 알아야합니다.  

파이썬의 반복문 for  loop 문의 이해

예제1.  숫자 7을 출력하시오 !

print(7)

예제2. 숫자 7을 10번 출력하시오 !

for  i  in  range(1,11):  # 1부터 11미만까지 
    print(7)

★ 문법:
for  카운터   in  몇번반복할지를 지정:
    실행문 

예제3. 숫자 1번부터 10번까지 출력하시오 !

for  i  in  range(1,11):
    print(i)

예제4. 숫자 0번부터 10번까지 출력하는데 짝수만 출력하시오 !

for  i  in  range(0,11,2): #0번부터 11미만까지 숫자를 출력하는데 2칸씩 건너뜀
    print(i)

#8. 모델 성능 개선

유방암 분류 기존 코드를 전부 가져옵니다. 

# ■ 실습1. 유방암 환자 분류 knn 모델 생성 

#1. 데이터 로드
import pandas  as  pd

wbcd = pd.read_csv("d:\\data\\wisc_bc_data.csv")
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

model = KNeighborsClassifier( n_neighbors=5 )
model.fit( x_train, y_train )  

#6. 모델 예측
result = model.predict(x_test) 

#7. 모델 평가 
acc = sum( result == y_test ) / len(y_test) * 100
print(acc)

#8. 모델 성능 개선

가져온 코드에 for loop 문을 써서 정확도가 가장 좋은 k 값이 몇인지
알아내게 자동화 하시오 !
# ■ 실습1. 유방암 환자 분류 knn 모델 생성 

#1. 데이터 로드
import pandas  as  pd

wbcd = pd.read_csv("d:\\data\\wisc_bc_data.csv")
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

설명: k 가 11일때가 가장 적절합니다. 

#문제1. 와인 데이터(wine.csv) 의 종류를 분류하는 knn 모델을 생성하시오 !

#1. 데이터 불러오기
import  pandas  as  pd
wine = pd.read_csv("c:\\data\\wine2.csv")
wine.head()

#2. 데이터 확인하기 
#3. 결측치가 있는지 확인하기 
#4. 데이터 스켈링 하기 
#5. 정답 데이터를 numpy 로 변환하기 
#6. 훈련과 테스트를 9:1 로 분리하기 
#7. 모델 훈련
#8. 모델 예측
#9.  모델 평가
#10. 모델 성능 개선 
