
▩ 예제26.순위를_출력하는_dense_rank_함수_배우기.py

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  ename, sal, dense_rank()  over  ( order  by sal  desc)  순위
          from emp;  

Pandas>



문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  job, ename, sal, dense_rank()   over ( partitoin  by  job
                                                                 order  by  sal  desc )  순위
          from  emp;

답:




