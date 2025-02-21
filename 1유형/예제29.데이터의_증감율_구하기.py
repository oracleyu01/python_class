
▩ 예제29.데이터의_증감율_구하기.py 

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, sal, lag(sal,1)  over  ( order  by sal  asc)   as  lag_sal,
                               lead(sal,1)  over  ( order  by sal asc)  as  lead_sal
          from  emp;

Pandas>




문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL> with  tab2  as (  select  ename, sal, 
                                 lag(sal,1)  over  ( order  by sal  asc)   as  lag_sal,
                                   lead(sal,1)  over  ( order  by sal asc)  as  lead_sal
                              from  emp   )
       select   ename, sal - lag_sal
          from  tab2;

Pandas>









