
▩ 예제23.컬럼_두개로_group_by_하기.py 

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  deptno, job, sum(sal)
           from  emp
           group by  deptno, job ; 

Pandas>  



문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  to_char(hiredate, 'RRRR'), deptno, sum(sal)
           from  emp
           group  by  to_char(hiredate,'RRRR'), deptno; 

Pandas> 




