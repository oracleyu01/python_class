
▩ 예제40.오라클의_서브쿼리를_판다스로_구현하기3_(multiple_column_subquery).py

예제1. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, sal, job
         from  emp
         where  deptno   in  (  select  deptno 
                                        from  emp 
                                       where  comm  is  not  null  )
          and   job   in  ( select  job
                                 from  emp
                                 where  comm  is  not  null );
답:




문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL>  select   ename, sal, job
             from  emp
             where  deptno  = ( select  deptno 
                                          from  dept
                                          where  loc='DALLAS' )
             and   sal   in  (  select  sal
                                     from  emp 
                                    where  job in ('ANALYST', 'CLERK')  );

답:




