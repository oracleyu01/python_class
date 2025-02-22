
▩ 예제34.판다스에서의_집합_연산자_사용법1_(합집합).py

        오라클               vs                판다스
1.     union   all                              pd.concat
2.     union                                    pd.concat + drop_duplicates()
3.    intersect                                 isin 을 사용한 코드
4.    minus                                    isin 을 사용한 코드

예제.  아래의 SQL을 판다스로 구현하시오 !

SQL> select   ename, sal, deptno 
           from  emp 
           where   deptno  in  ( 10, 20 )
        union  all
        select  ename, sal, deptno
          from  emp
          where  deptno = 10;

답:




설명: axis=0 은 위아래로 연결(기본값)
     axis=1  은 양옆으로 연결 

문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL> select   deptno, job
            from  emp
            where  deptno = 10 
       union  all 
        select  deptno, loc
          from  dept; 

답:




