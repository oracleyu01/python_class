
▩ 예제38.오라클의_서브쿼리를_판다스로_구현하기1_(single_row_subquery).py

* 서브쿼리의 종류 3가지 ?

1. 단일행 서브쿼리 (single row subquery)
2. 복수행 서브쿼리 (multiple row  subquery)
3. 복수 컬럼 서브쿼리 (multiple column subquery)

예제. 아래의 SQL을 판다스로 구현하시오 !
       JONES 보다 더 많은 월급을 받는 사원들의 이름과 월급을 출력하시오!

SQL> select  ename, sal
          from  emp
          where  sal > ( select  sal 
                                from  emp 
                                where  ename='JONES');

답:




문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, sal
           from  emp
           where  sal =  (  select   sal 
                                  from  emp 
                                  where  ename='SCOTT' )
           and  ename != 'SCOTT';

답:




