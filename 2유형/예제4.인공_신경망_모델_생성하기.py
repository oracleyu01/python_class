
▣ 예제4.인공_신경망_모델_생성하기.py

# 인공신경망(Artificial Neural Network) 이론 정리

## 1. 기본 구조와 개념
- 생물학적 신경망을 모방한 기계학습 모델
- 입력층, 은닉층, 출력층으로 구성
- 각 층은 노드(뉴런)들로 구성되며 가중치로 연결됨
- 단층신경망(입력층 --> 출력층)과 다층신경망(입력층->은닉층들-->출력층)으로 구분

## 2. 학습 메커니즘
### 역전파 알고리즘(Backpropagation)
- 출력값과 목표값의 차이(오차)를 역으로 전파하며 가중치 조정(★)
- 경사하강법을 사용하여 오차를 최소화하는 방향으로 학습
- 다층신경망의 핵심 학습 알고리즘


### 활성화 함수
- 뉴런의 출력을 결정하는 함수
- 주요 활성화 함수:
  - Sigmoid
  - ReLU
  - tanh
- 입력변수의 속성과 관계없이 문제의 특성에 따라 선택

 한번 활성화 함수를 어떤 하나로 지정했으면 계속 그 활성화함수를 사용하는것입니다.
 사람이 나중에 변경할 수 있는데 입력되는 데이터에 따라 자동으로 활성화 함수가
 결정되는건 아닙니다. 

## 3. 과적합(Overfitting) 방지 기법

### 정규화(Regularization)
- 가중치에 제약을 주어 모델의 복잡도를 제한
- L1, L2 정규화 등이 사용됨

예: 훈련 데이터에 독버섯은 냄새가 중요한 변수여서
    신경망이 냄새에 아주 높은 가중치를 부여했는데
    테스트 데이터의 독버섯중에 냄새가 없는 독버섯이
    있어서 오버피팅이 생긴다면 L1,L2 정규화를 써서 냄새에 대한 가중치를 낮추어
   오버피팅을 줄입니다. 


### 드롭아웃(Dropout)
- 학습 과정에서 일부 뉴런을 무작위로 비활성화
- 뉴런 간의 상호의존성을 줄임

### 조기 종료(Early Stopping)
- 검증 오차가 증가하기 시작하면 학습을 중단
- 불필요한 과적합을 방지

예제. 학습을 진행하는데 테스트 데이터의 정확도가
       가장 좋았던 그 지점에서 학습을 멈추겠다.

관련그림: https://cafe.daum.net/oracleoracle/Sq8G/70

## 4. 주요 특성
- 은닉층 노드 수가 많을수록:
  - 더 복잡한 패턴 학습 가능
  - 과대적합 위험 증가

- 가중치가 0이면 선형 모델로 단순화됨
- 이상치나 잡음에 비교적 민감함
- 은닉층의 수와 노드 수는 수동으로 설정 필요(★)
   
 은닉층의 갯수와 뉴런수(노드)는 사람이 조정해야
 하는 하이퍼 파라미터 입니다. 

 가중치는 그냥 파라미터라고 합니다. 
 가중치는 학습되면서 자동으로 갱신됩니다. 

정리:  1. 파라미터 :    자동으로 생성되는 값

            예: 신경망의 가중치 

        2. 하이퍼 파라미터 : 사람이 조정해야하는 값

           예: - 신경망의 뉴런수, 층수
                - 서포트 백터머신의 C 와 gamma
                - knn 의  k 값
                - 나이브베이즈의 laplace 값  

## 5. 장단점
### 장점
- 복잡한 비선형 관계 학습 가능
- 패턴 인식 능력이 우수
- 다양한 문제에 적용 가능

### 단점
- 모델 해석이 어려움(블랙박스)
- 학습에 많은 데이터와 시간 필요
- 하이퍼파라미터 튜닝이 중요

문제1. (2023년 ADSP 34회) 인공신경망에 대한 설명으로 옳지 않은 것은?

① 이상치나 잡음에 민감하지 않다
② 은닉층 노드가 많으면 과대적합이 발생할 수 있다
③ 은닉층의 수와 은닉노드 수는 자동으로 설정된다
④ 가중치가 0이면 선형 모델이 된다

정답:  

문제2. (2024년 빅데이터분석기사 5회) 인공신경망 모형에서 과적합을 방지할 수 있는 방법으로 옳지 않은 것은?

① 정규화
② 드롭아웃
③ 조기 종료
④ 가지치기

정답:  

문제3. (2023년 ADSP 33회) 인공신경망의 특징으로 옳은 것은?

① 입력변수의 속성에 따라 활성화 함수의 선택이 달라진다
② 단층신경망이 다층신경망보다 훈련이 어렵다
③ 은닉층 노드가 적을수록 복잡한 의사결정 경계를 만들 수 있다
④ 역전파 알고리즘을 통해 가중치를 조정한다

정답:  

■  파이썬으로 신경망 만드는 실습 

□ 와인의 품질을 분류하는 인공 신경망 만들기

#1. 데이터 로드
#2. 결측치 확인
#3. 데이터 스켈링
#4. 훈련과 테스트 분리
#5. 모델 생성
#6. 모델 훈련
#7. 모델 예측
#8. 모델 평가
#9. 모델 개선 

# 구현



문제1. 독일 은행 데이터 분류 랜덤포레스트 모델 전체 코드를 그대로 가져와서
      신경망으로 변경해서 분류하시오 !


#1. 데이터 불러오기
import pandas  as  pd

credit = pd.read_csv("d:\\data\\credit.csv") 
#credit.head()


문제2.  위의 신경망 모델의 성능을 더 올리시오 !

nn_model = MLPClassifier( random_state = 42,
                          hidden_layer_sizes=(32,16), 
                          activation='relu',    # 활성화 함수( tanh, logistic )
                          solver='adam',     # 경사하강법
                          max_iter= 1000 )    #  학습 반복횟수 
                                    

 hidden_layer_sizes=(숫자1, 숫자2, 숫자3) <---  숫자를 늘리면서 정확도를 보시면 됩니다

