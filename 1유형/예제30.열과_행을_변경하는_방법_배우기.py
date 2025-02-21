
▩ 예제30.열과_행을_변경하는_방법_배우기.py

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  *
           from   (  select  deptno, sal  from  emp )
           pivot  ( sum(sal)  for  deptno  in  ( 10, 20, 30 )  ) ;


Pandas>  


예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  *
           from   (  select  job, deptno,  sal  from  emp )
           pivot  ( sum(sal)  for  deptno  in  ( 10, 20, 30 )  ) ;

답: 


문제1. 위에서 출력되고 있는 결과에서 결측치 부분을 0 으로 출력하시오 !

답:


문제2. 아래의 SQL을 판다스로 구현하시오 !

SQL> select *
           from   (  select  deptno,job, sal  from  emp )
           pivot (  sum(sal)  for  job  in ( 'SALESMAN'   as "SALESMAN",
                                                   'ANALYST'   as "ANALYST",
                                                   'CLERK'  as  "CLERK",
                                                  'MANAGER'  as  "MANAGER",
                                                  'PRESIDENT'  as "PRESIDENT" )  ); 

답:



문제3. 아래의 SQL을 판다스로 구현하시오 !

SQL> select *
           from   (  select  to_char(hiredate,'RRRR') as h_year ,job, sal  from  emp )
           pivot (  sum(sal)  for  job  in ( 'SALESMAN'   as "SALESMAN",
                                                   'ANALYST'   as "ANALYST",
                                                   'CLERK'  as  "CLERK",
                                                  'MANAGER'  as  "MANAGER",
                                                  'PRESIDENT'  as "PRESIDENT" )  ); 


답:





