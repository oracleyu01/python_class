
▩ 예제28.사분위수_데이터_만들기.py

예제1. 사원 테이블의 월급의 1사분위수 값과 3사분위수 값을 출력하시오 !

 |------------|------------|--------------|-------------|
 0           25%         50%            75%          100%
              ↑                           ↑
          1사분위수(Q1)             3사분위수(Q3)

IQR (Interquartile Range) =  Q3 -- Q1

SQL> select  percentile_cont(0.25)  within  group ( order  by sal ) as  Q1,
                 percentile_cont(0.75)  within  group ( order  by sal ) as  Q3
           from  emp;

Pandas>  




예제2. 아래의 SQL을 판다스로 수행하시오 !

SQL> with  tab1 as  (  select  percentile_cont(0.25)  within  group ( order  by sal ) as  Q1,
                                      percentile_cont(0.75)  within  group ( order  by sal ) as  Q3
                           from  emp )
       select  abs( Q3 - Q1 )
         from  tab1;

답:

# 사원 테이블 월급의 Q1 과 Q3 를 각각 구합니다. 
q1 = emp['sal'].quantile(0.25)
q3 = emp['sal'].quantile(0.75)

# IQR(Q3-Q1) 을 계산하는데 절대값을 출력합니다.
print( abs(q3 - q1) ) 



문제1. (빅분기 시험 4회 시험 출제 문제)  employee_salary_data.csv 에서 
      salary 컬럼의 3사분위수와 1사분위수의 차이를 절대값으로 구하고 
       소수점을 버린수 정수로 출력하시오 !

데이터 있는곳 : https://cafe.daum.net/oracleoracle/Sp62/714

import  pandas  as  pd

df = pd.read_csv("d:\\data\\employee_salary_data.csv")

답:



