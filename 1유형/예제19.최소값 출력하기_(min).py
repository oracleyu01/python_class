
▩ 예제19.최소값 출력하기_(min).py

예제. 아래의 SQL을  판다스로 구현하시오 !

SQL> select  job,  min(sal)
          from  emp
          group   by   job; 

답: 




설명:  groupby 와  reset_index() 는 서로  짝꿍입니다.
       groupby 만 쓰게 되면 시리즈(컬럼) 형태로 출력되는데
       reset_index() 를 쓰게되면 데이터 프레임 형태로 출력됩니다.

문제1.  아래의 SQL을 판다스로 구현하세요.

SQL>  select   deptno,  min(sal)
          from  emp
          group  by  deptno  
          order  by   deptno  asc ;

답:  




