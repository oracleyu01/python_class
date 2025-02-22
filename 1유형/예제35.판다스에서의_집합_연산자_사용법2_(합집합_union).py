
▩ 예제35.판다스에서의_집합_연산자_사용법2_(합집합_union).py

        오라클               vs                판다스
1.    union   all                              pd.concat
2.    union                                    pd.concat + drop_duplicates()
3.    intersect                                isin 을 사용한 코드
4.    minus                                    isin 을 사용한 코드

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select ename, sal, deptno
         from  emp
        where deptno  in  (10, 20 )
         union 
        select  ename, sal, deptno
           from emp
           where deptno = 10;

설명: union 은 union all 과 다르게 중복행을 제거해서 출력합니다.

답:




문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL>  select   ename, sal, deptno
           from  emp
           where  sal  between  1000  and  3000
          union  
          select  ename, sal, deptno
           from emp
          where deptno  in (10, 20);

답:



