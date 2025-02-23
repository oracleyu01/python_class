■ 예제50.서포트_백터_머신.py


설명 그림 : https://cafe.daum.net/oracleoracle/Sq8G/82

1. 기본 개념  ──────────────────────────────

목적: 주어진 데이터를 두 개 이상의 클래스로 분리하는 결정 경계(초평면)를 찾는 알고리즘입니다.

핵심 아이디어: 결정 경계와 가장 가까운 데이터 포인트(서포트 벡터) 사이의 거리를 최대화하여 일반화 성능을 높입니다.

2. 선형 분리와 최대 마진 ────────────────────────────── 

선형 분리 가능할 경우: 데이터가 선형적으로 분리 가능하면, SVM은 선형 초평면 (w · x + b = 0)을 이용하여 두 클래스를 분리합니다.

최대 마진 원리:

• 마진: 결정 경계와 서포트 벡터(가장 가까운 데이터 포인트) 사이의 최소 거리를 의미합니다.

3. 하드 마진과 소프트 마진, 그리고 gamma 쉽게 이해하기 ──────────────────────────────

♠ 하드 마진 (Hard Margin):

• "절대 실수하면 안돼!"라는 엄격한 기준을 가진 방식입니다
• 데이터가 완벽하게 구분될 수 있을 때만 사용 가능합니다
• 마치 시험에서 100점만 인정하는 것과 같습니다
• 단점: 현실 세계의 데이터는 완벽하게 구분하기 어려워 실제로 잘 사용하지 않습니다

♠  소프트 마진 (Soft Margin):

• "약간의 실수는 괜찮아"라는 유연한 기준을 가진 방식입니다
• 데이터 구분이 완벽하지 않아도 사용할 수 있습니다
• 마치 시험에서 몇 개의 실수는 허용하는 것과 같습니다
• C값으로 얼마나 실수를 허용할지 조절할 수 있습니다

C값이 크면: 엄격한 기준 (실수를 적게 허용)
C값이 작으면: 유연한 기준 (실수를 많이 허용)

♠  gamma 매개변수 (RBF 커널에서 사용):

• 결정 경계가 얼마나 구불구불해질 수 있는지 결정합니다
• 마치 고무줄의 유연성을 조절하는 것과 같습니다

- gamma가 크면: 매우 구불구불한 경계 (고무줄이 잘 휘어짐)

훈련 데이터에 더 정확히 맞출 수 있음
과적합 위험이 증가

- gamma가 작으면: 부드러운 경계 (뻣뻣한 고무줄)

더 단순한 형태의 경계
일반화 성능이 좋아질 수 있음

♠  실제 사용:
• 소프트 마진 방식을 더 많이 사용합니다 (현실적인 접근)
• C값과 gamma를 함께 조절하여 최적의 모델을 찾습니다

- C: 오분류 허용 정도
- gamma: 결정 경계의 복잡도

• 보통 교차 검증을 통해 최적의 C값과 gamma값을 찾습니다.

4. 커널 트릭 (Kernel Trick) ────────────────────────────── 

문제 상황: 데이터가 선형적으로 분리되지 않는 경우, 원래 입력 공간에서 선형 초평면으로 분리하기 어려움
해결 방법: 커널 함수를 사용하여 데이터를 고차원 특징 공간으로 매핑하면, 고차원에서는 선형 분리가 가능해집니다.

주요 커널 종류:

• 선형 커널: 데이터가 이미 선형 분리 가능한 경우 사용
• 다항식 커널 (Polynomial Kernel): 입력 데이터를 다항식 형태로 매핑
• RBF 커널 (Radial Basis Function Kernel): 가우시안 분포 기반으로 유연한 결정 경계를 형성 (가장 널리 사용됨)
• Sigmoid 커널: 신경망의 활성화 함수와 유사한 형태로 특정 상황에서 사용 가능

5. SVM의 장단점 및 활용 ────────────────────────────── 

장점:
• 높은 차원의 데이터에서도 효과적임
• 최대 마진 원리 덕분에 일반화 성능이 우수함
• 커널 트릭을 통해 비선형 문제도 해결 가능함

단점:
• 대규모 데이터셋에서는 학습 속도가 느리고 메모리 사용량이 많을 수 있음
• 적절한 커널과 하이퍼파라미터(C, 감마 등)의 선택이 모델 성능에 큰 영향을 미침


문제1. SVM(Support Vector Machine)에서 마진(margin)에 대한 설명으로 옳은 것은?
      (2023년 제3회 빅데이터분석기사 필기)

1.결정 경계와 가장 가까운 데이터 포인트 사이의 거리를 최소화한다
2.결정 경계와 가장 가까운 데이터 포인트 사이의 거리를 최대화한다
3.하드 마진은 오분류를 허용한다
4.소프트 마진은 오분류를 절대 허용하지 않는다

정답: 


문제2. SVM의 하이퍼파라미터 C에 대한 설명으로 옳은 것은?
      ( 2022년 제2회 빅데이터분석기사 필기)

1.값이 작을수록 모델이 복잡해진다
2.값이 클수록 모델이 단순해진다
3.기본값은 0이다
4.값이 작을수록 모델이 단순해진다

정답: 


문제3. SVM의 커널(kernel)함수에 대한 설명으로 옳은 것은?
      (2021년 제1회 빅데이터분석기사 필기)

1.선형 분리가 가능한 경우에만 사용한다
2.저차원의 데이터를 고차원으로 매핑한다
3.rbf 커널은 다항식 커널보다 연산속도가 느리다
4. sigmoid 커널은 이진분류에 사용할 수 없다

답: 

예제. 서포트 벡터 머신으로 iris 데이터 분류하기 

#1. 데이터 불러오기
import pandas as pd
iris = pd.read_csv("c:\\data\\iris2.csv")

#2. 결측치 확인
#iris.isnull().sum()

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
x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.1,
                                                  random_state=1)

#6. 모델 생성
from sklearn.svm import SVC
svm_model = SVC(
   kernel='linear',   # rbf 대신 linear 커널 사용 
   C=0.1,            # C값을 작게 설정하여 과소적합 유도
   random_state=1
)

#7. 모델 훈련
svm_model.fit(x_train, y_train)

#8. 모델 예측
result = svm_model.predict(x_test)

#9. 모델 평가
accuracy = sum(result == y_test) / len(y_test)
print(f"모델 정확도: {accuracy:.4f}")

#10. 모델 성능 상세 평가
from sklearn.metrics import classification_report
print("\n분류 리포트:")
print(classification_report(y_test, result))


문제1. 위의 모델에 성능을 올리시오 !




