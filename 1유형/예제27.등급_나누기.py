
▩ 예제27.등급_나누기.py

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  ename, sal, ntile(4)  over ( order by  sal desc ) as 등급 
          from  emp;

Pandas>  




문제. 부동산 허위매물 데이터(train.csv) 에서 보증금을 4개의 등급으로 나눠서 
        아래의 SQL 처럼 판다스로 출력하시오 !

SQL>  select   ID, 보증금, ntile(4) over ( order  by 보증금 desc)  as 등급 
          from train;

Pandas>  



※ 위의 등급과 같은 파생 컬럼을 생성하게 되면 기계학습할 때 기계가 학습을 더 잘하게 됩니다.
