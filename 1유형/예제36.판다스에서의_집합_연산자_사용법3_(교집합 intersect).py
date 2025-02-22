
▩ 예제36.판다스에서의_집합_연산자_사용법3_(교집합 intersect).py

        오라클               vs                판다스
1.     union   all                              pd.concat
2.     union                                    pd.concat + drop_duplicates()
3.    intersect                                 isin 을 사용한 코드
4.    minus                                    isin 을 사용한 코드

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select ename, sal, deptno
            from  emp
            where deptno  in  (10, 20 )
         intersect
        select  ename, sal, deptno
           from emp
           where deptno = 10;

답:




문제1. 아래의 SQL을 판다스로 구현하시오 !
       dept 테이블에서 부서번호를 출력하는데 emp 테이블에 있는 부서번호만 출력하시오!

select  deptno
  from  dept 
 intersect
select  deptno
  from  emp; 

답:  





