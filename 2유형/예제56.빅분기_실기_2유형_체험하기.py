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

# 1. 기계 학습 시킬 데이터를 불러옵니다.
import pandas as pd
X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

# 2. 데이터를 정규화 합니다.

# 훈련 데이터 정규화
from  sklearn.preprocessing  import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(x_train)
x_train2 = scaler.transform(x_train)

# 테스트데이터 정규화
from  sklearn.preprocessing  import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(x_test)
x_test2 = scaler.transform(x_test)

# 3. 정답 데이터 만들기

y = y_train['cancer']
y

# 4. 모델 생성

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


# 5. 모델훈련
model.fit(x_train2, y)

# 6. 모델예측
pred = model.predict(x_test2)
pred

# 7. 정답제출

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'label': pred }).to_csv('003000000.csv', index=False)
pd.DataFrame({'cust_id': X_test.cust_id, 'label': pred }).to_csv('003000000.csv', index=False)

# 8. 모델평가
from sklearn.metrics import roc_auc_score

y_hat = model.predict(x_train2)
print( roc_auc_score(y, y_hat))




# 문제:

자동차의 연비 예측을 위해 주어진 데이터셋을 활용하여 모델을 구축하세요.

데이터셋은 각 자동차의 특성(예: 차량_연식, 주행거리, 연료비, 엔진 크기, 차량 무게)을 포함하고 있습니다.

연비는 세 가지 범주로 분류됩니다.

1. 고연비: 매우 경제적인 차량
2. 중간연비: 적당한 경제성을 가진 차량
3. 저연비: 경제적이지 않은 차량

모델의 평가는 f1 score 로 평가됩니다.

# 주어진 데이터 :

훈련 데이터: train.csv
테스트 데이터 : test.csv

# 제출 형식

pred
중간연비
중간연비
중간연비
중간연비
저연비
중간연비
중간연비
저연비
저연비
중간연비

가상의 데이터 만들기:

import pandas as pd
import numpy as np

# 1. 가상 데이터 생성
np.random.seed(42)
n_samples = 1000

# 변수 생성
차량_연식 = np.random.randint(2000, 2021, n_samples)
주행거리 = np.random.randint(5000, 100000, n_samples)  # 상한선 줄임
연료비 = np.random.randint(50, 100, n_samples)  # 경제성 반영
엔진_크기 = np.random.randint(1000, 2000, n_samples)  # 작은 엔진 크기
차량_무게 = np.random.randint(800, 1500, n_samples)  # 가벼운 차량

# 연비를 다중 분류로 수정 ('고연비', '중간연비', '저연비')
연비 = np.where(
    (연료비 < 70) & (주행거리 < 50000) & (엔진_크기 < 1500) & (차량_무게 < 1200),
    '고연비',
    np.where((연료비 < 90) & (주행거리 < 70000), '중간연비', '저연비')
)

# 데이터 프레임 생성
data = {
    '차량_연식': 차량_연식,
    '주행거리': 주행거리,
    '연료비': 연료비,
    '엔진 크기': 엔진_크기,
    '차량 무게': 차량_무게,
    '연비': 연비
}

df = pd.DataFrame(data)

# 2. 학습 데이터 CSV 파일로 저장 (train.csv)
df.to_csv('train.csv', index=False)

# 3. 테스트 데이터 생성 (연비 제외)
n_samples = 500
test_data = {
    '차량_연식': np.random.randint(2000, 2021, n_samples),
    '주행거리': np.random.randint(5000, 100000, n_samples),
    '연료비': np.random.randint(50, 100, n_samples),
    '엔진 크기': np.random.randint(1000, 2000, n_samples),
    '차량 무게': np.random.randint(800, 1500, n_samples)
}

df_test = pd.DataFrame(test_data)

# 4. 테스트 데이터 CSV 파일로 저장 (test.csv)
df_test.to_csv('test.csv', index=False)


답:


