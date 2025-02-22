
▩ 예제37.판다스에서의_집합_연산자_사용법4_(차집합_minus).py

        오라클               vs                판다스
1.     union   all                              pd.concat
2.     union                                    pd.concat + drop_duplicates()
3.    intersect                                 isin 을 사용한 코드
4.    minus                                    isin 을 사용한 코드

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  ename, sal, deptno
           from  emp
           where  deptno  in  ( 10, 20 )
          minus 
         select  ename, sal, deptno
           from emp
           where deptno =10;

답: 




설명:  x1.ename.isin(x2.ename)==False 이면 ?  차집합이 출력 
       x1.ename.isin(x2.ename)==True 이면 ?  교집합이 출력 

문제1.  아래의 SQL을 판다스로 구현하시오 
        dept 테이블에는 존재하는 부서번호인데 emp 테이블에는 존재하지 않는 부서번호를 출력하시오 !

SQL>  select  deptno
           from dept
          minus
         select  deptno
           from emp;

답: 



