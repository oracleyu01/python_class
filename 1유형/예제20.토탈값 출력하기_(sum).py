
▩ 예제20.토탈값 출력하기_(sum).py

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select   job as 직업, sum(sal)  as  토탈월급
            from  emp
            where  job  != 'SALESMAN'
            group  by  job; 

답:




문제1.   아래의 SQL을 판다스로 구현하시오 !

SQL>  select  deptno,  sum(sal)
            from  emp
            where  deptno  != 20
           group  by  deptno
           order  by  sum(sal)  desc;

답:



