
▩ 예제33.SQL의_조인을_판다스로_구현하기1_(self_join).py


     오라클                 vs                 판다스 

    equi join
 non equi join                                merge 함수
   outer  join
   self   join   

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  사원.ename, 관리자.ename
           from  emp   사원,  emp  관리자 
           where  사원.mgr = 관리자.empno; 

답: 




문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  사원.ename, 관리자.ename
           from  emp   사원,  emp  관리자 
           where  사원.mgr = 관리자.empno  and  사원.sal> 관리자.sal;

답: 




