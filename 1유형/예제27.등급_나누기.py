
▩ 예제27.등급_나누기.py

예제. 아래의 결과를 판다스로 구현하시오 !

ename      sal    등급  
----------------------
KING       5000   1   
FORD       3000   1   
SCOTT      3000   1   
JONES      2975   1   
BLAKE      2850   2   
CLARK      2450   2   
ALLEN      1600   2   
TURNER     1500   3   
MILLER     1300   3   
MARTIN     1250   4   
WARD       1250   4   
ADAMS      1100   4   
JAMES      950    4   
SMITH      800    4   

Pandas>  




문제. 부동산 허위매물 데이터(train.csv) 에서 보증금을 4개의 등급으로 나눠서 
        아래의 SQL 처럼 판다스로 출력하시오 !

SQL>  select   ID, 보증금, ntile(4) over ( order  by 보증금 desc)  as 등급 
          from train;

Pandas>  



※ 위의 등급과 같은 파생 컬럼을 생성하게 되면 기계학습할 때 기계가 학습을 더 잘하게 됩니다.
