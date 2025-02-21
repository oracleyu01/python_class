
▩ 예제22.건수_출력하기_(count).py

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  job, count(*)
          from  emp
          group  by  job;

답:  




문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  job, count(*)
          from emp
          group  by  job
          having  count(*) >= 3; 

답:  




문제2.  허위매물여부, 허위매물여부별 건수를 출력하시오 !

SQL> select  허위매물여부, count(*)
          from  train
          group  by  허위매물여부;

답:


※ 데이콘에서 상위권에 등급하는 팁 !    관심범주와 비관심주의 비율이 동일할수록 기계학습 결과가 좋습니다. 

