■ 예제51.K-means_군집분석_개념_정리.py



1. K-means란?

- 데이터를 k개의 그룹으로 나누는 대표적인 군집분석 방법입니다
- 비슷한 특성을 가진 데이터들을 같은 그룹으로 묶어주는 기법입니다
- 쉽게 말해 "비슷한 것끼리 모아주는 방법"입니다

2. 기본 원리

설명 그림: https://k-means--p3c2bb8.gamma.site/

- k개의 중심점을 무작위로 정합니다
- 각 데이터를 가장 가까운 중심점에 배정합니다
- 그룹별로 새로운 중심점을 계산합니다
- 중심점이 더 이상 변하지 않을 때까지 2-3을 반복합니다

3. k값 정하는 방법: 엘보우 기법

- k값을 1부터 늘려가면서 그룹 내 분산을 확인합니다
- 그래프가 팔꿈치 모양처럼 꺾이는 지점의 k값을 선택합니다
- 쉽게 말해 "더 나누어도 별 효과 없는 지점"을 찾는 것입니다

4. K-means가 사용되는 경우

데이터를 자연스러운 그룹으로 나누고 싶을 때

예시:

- 고객 세분화: 비슷한 구매 패턴을 가진 고객 그룹화
- 문제의 교복 치수: 비슷한 체형의 학생들을 그룹화
- 이미지 압축: 비슷한 색상을 하나의 값으로 묶기


5. K-means의 특징

장점:

- 이해하기 쉽고 구현이 간단함
- 큰 데이터에서도 빠르게 동작함


단점:

- k값을 미리 정해야 함
- 초기 중심점에 따라 결과가 달라질 수 있음
- 동그란 형태의 군집만 잘 찾음

6. 비지도 학습으로서의 K-means

정답 라벨 없이 데이터만으로 학습하는 방식입니다

다른 비지도 학습 예시:

- 차원 축소
- 이상치 탐지
- 연관 규칙 학습


문제1.K-means의 k를 정하기 위한 방법으로 적절한 것은? (2023년 제7회 빅데이터분석기사 필기)

1.랜덤 포레스트
2.교차 검증
3.엘보우 기법
4.그리드 서치

정답: 


문제2. 비지도 학습 알고리즘 유형으로 알맞은 것은? (2021년 제2회 빅데이터분석기사 필기)

1.회귀분석
2.로지스틱 회귀분석
3.서포트 벡터
4.군집분석

정답: 


문제3. 학생들의 교복의 표준 치수를 정하기 위해 학생들의 팔길이, 키, 가슴둘레를 기준으로 할 때 어떤 방법이 가장 적절한 기법인가?
      (2021년 제2회 빅데이터분석기사 필기)

1.이상치
2.군집
3.분류
4.연관성

정답: 


■ 실습1. 기본 데이터로 실습 

# 1. 기본 데이터로 K-means 실습
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 데이터 생성
data = np.array([
    [3, 4], [1, 5], [7, 9], [5, 4], [6, 8],
    [4, 5], [9, 8], [7, 8], [6, 7], [2, 1]
])

# K-means 모델 생성 및 학습




# 결과 시각화
plt.figure(figsize=(4, 3))
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='*', s=200, c='red', label='Centroids')
plt.title('K-means Clustering (k=2)')
plt.legend()
plt.show()


■ 실습2. 과일/음식 데이터 군집화

data2 = np.array([
    [10, 9],  # apple
    [1, 4],   # bacon
    [10, 1],  # banana
    [7, 10],  # carrot
    [3, 10],  # salary
    [1, 1],   # cheese
    [6, 7]    # tomato
])

# 3개 군집으로 분류


# 시각화 

plt.figure(figsize=(8, 6))
plt.scatter(data2[:, 0], data2[:, 1], c=kmeans2.labels_)
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], 
            marker='*', s=200, c='red', label='Centroids')
plt.title('K-means Clustering (k=3)')
plt.legend()
plt.show()

문제1. 학생 시험 점수 데이터로 k-means 모델을 만드시오 !




# 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(scores[:, 0], scores[:, 1], c=kmeans3.labels_)
plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], 
            marker='*', s=200, c='red', label='Centroids')
plt.xlabel('Math Score')
plt.ylabel('English Score')
plt.title('Student Scores Clustering')
plt.legend()
plt.show()

문제2. 유방암 환자 데이터를 비지도 학습으로 분류하시오 !


