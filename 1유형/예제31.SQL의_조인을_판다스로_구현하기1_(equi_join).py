
▩ 예제31.SQL의_조인을_판다스로_구현하기1_(equi_join).py

     오라클                 vs                 판다스 

    equi join
 non equi join                                merge 함수
   outer  join
   self   join   

예제.  dept.csv 를 데이터 프레임으로 구성 하시오 !

import pandas  as  pd

emp = pd.read_csv("d:\\data\\emp.csv")
dept = pd.read_csv("d:\\data\\dept.csv")

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select   e.ename,  d.loc
           from  emp  e,  dept   d
          where  e.deptno = d.deptno ;

답:
import pandas  as  pd

result = pd.merge(emp, dept, on='deptno')
result.loc[ : , ['ename', 'loc'] ]

문제1. 아래의 SQL 을 판다스로 구현하시오 !

SQL>  select   e.ename,  d.loc
           from  emp  e,  dept   d
          where  e.deptno = d.deptno  and  e.job='ANALYST'; 

답:
import pandas  as  pd

result = pd.merge(emp, dept, on='deptno')
result.loc[ result['job']=='ANALYST' , ['ename', 'loc'] ]

