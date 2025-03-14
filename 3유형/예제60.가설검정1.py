■ 예제60.가설검정1.가설검정_용어_이해하기.py


--------------------------------
[가설검정의 기본 용어]

1. 귀무가설 (Null Hypothesis, H0):
   - 정의:
       연구나 실험에서 기존의 주장, 이론 또는 아무런 효과나 차이가 없음을 나타내는 가설.
       예: "두 집단 간에 차이가 없다", "처리 효과가 없다".
   
2. 대립가설 (Alternative Hypothesis, H1):
   - 정의:
       귀무가설과 반대되는 가설로, 연구자가 입증하고자 하는 가설.
       예: "두 집단 간에 차이가 있다", "처리 효과가 있다".

--------------------------------
[유의수준 (Significance Level, α)]

- 정의:
    가설 검정에서 귀무가설이 참일 때, 이를 잘못 기각할 확률의 한계값.
- 일반적으로 0.05(5%)나 0.01(1%) 등을 사용.
- 해석:
    p-value가 유의수준보다 작으면 귀무가설을 기각함.
    (문제에서 “유의수준은 귀무가설이 참일 때 기각할 확률”이라는 설명이 옳은 이유)

--------------------------------
[오류의 종류]

🎨그림: https://github.com/oracleyu01/python_class/blob/main/3유형/그림/1종오류.png
🎨그림: https://github.com/oracleyu01/python_class/blob/main/3유형/그림/2종오류.png
                       
1. 제1종 오류 (Type I Error):
   - 정의:
       실제로 귀무가설이 참인데, 이를 잘못 기각하는 오류.
   - 유의수준(α)은 제1종 오류를 범할 확률을 나타냄.

2. 제2종 오류 (Type II Error):
   - 정의:
       실제로 대립가설이 참인데, 귀무가설을 기각하지 못하는 오류.
   - β로 표기하며, (1-β)는 검정의 파워(Power)를 의미함.

--------------------------------
[p-value의 의미]

- 정의:
    귀무가설이 참이라는 전제 하에, 관측된 검정통계량보다 더 극단적인 값이 나타날 확률.
- 해석:
    p-value가 작을수록 (예: 0.05 미만) 귀무가설을 기각할 충분한 근거가 됨.
    즉, p-value는 귀무가설 하에서 현재의 데이터를 관찰할 확률을 나타냄.

--------------------------------
[가설검정 절차 요약]

🎨그림: https://github.com/oracleyu01/python_class/blob/main/3유형/그림/검정통계량.png
                       
1. 가설 설정: 귀무가설(H0)과 대립가설(H1) 설정.
2. 유의수준 결정: 일반적으로 0.05 또는 0.01 등.
3. 검정통계량 계산: 데이터로부터 검정통계량 산출.
4. p-value 산출: 검정통계량에 해당하는 p-value 계산.
5. 결정: p-value < 유의수준이면 귀무가설 기각, 그렇지 않으면 채택.

--------------------------------
요약:
가설검정은 데이터를 바탕으로 귀무가설과 대립가설 중 어느 쪽을 채택할지 결정하는 통계적 절차입니다.
유의수준, p-value, 제1종 및 제2종 오류 등 각 용어의 의미를 정확히 이해하는 것이 중요합니다.



문제1. 가설검정에서 사용되는 용어에 대한 설명으로 옳은 것은?
       (2023년 제7회 빅데이터분석기사 필기)

1. 유의수준(α)은 귀무가설이 참일 때 이를 기각할 확률이다
2. p-value가 유의수준보다 크면 귀무가설을 기각한다
3. 제1종 오류는 귀무가설이 거짓인데 채택하는 오류이다
4. 제2종 오류는 귀무가설이 참인데 기각하는 오류이다


정답: 


문제2. 가설검정에서 귀무가설과 대립가설에 대한 설명으로 옳은 것은?
        (2022년 제4회 빅데이터분석기사 필기)

1.귀무가설은 연구자가 입증하고 싶은 가설이다
2.대립가설은 기존의 주장이나 이론을 나타낸다
3.귀무가설은 영가설이라고도 하며 기존의 주장이다
4.대립가설이 기각되면 귀무가설도 기각된다

정답: 


문제3. 통계적 가설검정에서 p-value의 의미로 옳은 것은?
      (2021년 제2회 빅데이터분석기사 필기)

1.귀무가설이 참일 때 검정통계량이 관측된 값보다 더 극단적인 값이 나올 확률
2.대립가설이 참일 때 검정통계량이 관측된 값과 같을 확률
3.표본이 정규분포를 따를 확률
4.모집단에서 표본이 추출될 확률

정답: 




