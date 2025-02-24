

■ 예제52.연관규칙_분석(Association_Rule_Analysis).py

1. 연관규칙이란?

데이터 안에서 항목들 간의 관련성을 찾아내는 분석 방법

가장 대표적인 예: 장바구니 분석

"맥주를 산 사람은 과자도 산다"와 같은 규칙 발견


비지도 학습의 한 종류로, 정답 없이 데이터 간의 관계를 찾음

2. 주요 측도(지표)

1) 지지도(Support)

전체 거래 중 항목 A와 B가 동시에 포함된 비율

공식: P(A∩B) = (A와 B가 동시에 있는 거래 수) / (전체 거래 수)

예: 전체 100건의 거래 중 맥주와 과자를 같이 산 경우가 20건이면 지지도는 0.2

2) 신뢰도(Confidence)

A를 구매한 고객이 B도 구매한 비율

공식: P(B|A) = P(A∩B) / P(A)

예: 맥주 구매자 40명 중 30명이 과자도 샀다면 신뢰도는 0.75

3) 향상도(Lift)

두 항목이 우연히 함께 구매되는 것이 아님을 보여주는 지표

공식: P(B|A) / P(B)

- 1보다 크면: 양의 상관관계
- 1이면: 독립
- 1보다 작으면: 음의 상관관계

3. Apriori 알고리즘

연관규칙을 찾는 대표적인 알고리즘

작동 원리:

- 최소 지지도를 만족하는 빈발 항목 집합 찾기
- 빈발 항목 집합으로부터 연관규칙 생성
- 규칙들의 신뢰도와 향상도 계산


4. 연관규칙의 활용

- 상품 진열 전략
- 교차 판매(Cross-selling)
- 추천 시스템
- 고객 장바구니 분석
- 웹사이트 방문 패턴 분석

5. 장점과 단점

장점:

- 이해하기 쉬움
- 해석이 직관적
- 새로운 패턴 발견 가능

단점:

- 계산량이 많을 수 있음
- 너무 많은 규칙이 도출될 수 있음
- 인과관계가 아닌 상관관계만 보여줌


문제1. 연관규칙 측도 중 하나로, A항목이 포함된 거래 중 A항목과 B항목이 동시에 포함된 거래의 비율을 나타내는 지표는?
      (2022년 제5회 빅데이터분석기사 필기)

1.지지도
2.신뢰도
3.향상도
4.연관도

정답: 


문제2. 항목 집합의 지지도를 산출하여 발생 빈도와 최소지지도를 기반으로 거래의 연관성을 밝히는 알고리즘은?
      (2022년 제4회 빅데이터분석기사 필기)

1.K-means
2.Apriori
3.DBSCAN
4.KNN

정답: 

문제3. 비지도 학습 알고리즘의 대표적인 예로 옳은 것은?  (2021년 제2회 빅데이터분석기사 필기)

1.로지스틱 회귀
2.의사결정나무
3.연관규칙
4.랜덤포레스트

정답: 

■ 실습

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 데이터프레임 생성
data1 = pd.DataFrame({
    '빵': [0, 1, 0, 1, 1],
    '계란': [0, 1, 0, 0, 0],
    '우유': [1, 0, 1, 1, 1]
})

# 1. 빈발 항목 집합 찾기 (최소 지지도 0.2)
frequent_itemsets = apriori(data1, min_support=0.2, use_colnames=True)

# 2. 연관규칙 생성 (최소 신뢰도 0.6)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# 3. 규칙을 리프트(lift) 기준으로 정렬
rules = rules.sort_values('lift', ascending=False)

# 4. 결과 출력
print("\n연관규칙 분석 결과:")
print(rules)


문제1.  건물 업종 데이터 분석

# 데이터 읽기
building_data = pd.read_csv("building.csv", encoding='cp949')

