
▩ 예제25.순위를_출력하는_rank_함수_배우기.py

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  ename, sal, rank()  over  ( order  by sal  desc )  as  순위
            from  emp;

Pandas> 



문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, sal, job, rank()  over  ( order  by sal  desc ) 순위
         from  emp
        where  job='SALESMAN';

답:



문제2. 아래의 SQL을 판다스로 구현하시오 !

SQL>  seleect  deptno, ename, sal, rank()  over ( partition  by  deptno 
                                                              order  by  sal  desc ) 순위
           from  emp;

답:



