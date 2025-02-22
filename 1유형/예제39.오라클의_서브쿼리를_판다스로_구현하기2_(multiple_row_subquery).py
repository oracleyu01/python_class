
▩ 예제39.오라클의_서브쿼리를_판다스로_구현하기2_(multiple_row_subquery).py

예제. 아래의 SQL을 판다스로 구현하시오 ! 

    직업이 SALESMAN 인 사원들과 같은 월급을 받는 사원들의 이름과 월급을
    출력하시오 ! 

SQL> select  ename, sal
           from  emp
           where   sal   in  ( select  sal 
                                    from  emp
                                    where  job='SALESMAN' );

답:



문제1. 아래의 SQL을 판다스로 구현하시오 !
          관리자인 사원들의 이름과 월급 출력 하는 SQL

SQL>  select  ename, sal
           from  emp
           where  empno  in  ( select  mgr
                                       from  emp );

답:

