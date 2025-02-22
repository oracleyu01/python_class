
▩ 예제32.SQL의_조인을_판다스로_구현하기1_(outer_join).py


     오라클                 vs                 판다스 

    equi join
 non equi join                                merge 함수
   outer  join
   self   join   

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select   e.ename,  d.loc
           from  emp  e,  dept   d
          where  e.deptno = d.deptno (+) ;

답:  




how='inner' 는 오라클의 equi join 과 똑같습니다.
how='right' 는 dept 테이블쪽의 데이터가 모두 나오게해라 !
how='left' 는 emp테이블쪽의 데이터가 모두 나오게해라 !
how='outer' 는 오라클의 full outer 조인과 똑같습니다. 

문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL>  select   e.ename,  d.loc
           from  emp  e,  dept   d
          where  e.deptno (+) = d.deptno ;

답:  




문제2. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  d.loc, sum(e.sal)
          from  emp  e,  dept   d
          where  e.deptno = d.deptno 
          group  by  d.loc; 

답: 



문제3. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  d.loc, sum(e.sal)
          from  emp  e,  dept   d
          where  e.deptno (+) = d.deptno 
          group  by  d.loc; 

답: 


