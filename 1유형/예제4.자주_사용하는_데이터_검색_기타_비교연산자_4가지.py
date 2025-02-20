
▩ 예제4.  자주 사용하는 데이터 검색 기타 비교연산자 4가지 

           오라클           vs        판다스  

1. between  .. and                    emp['sal'].between(1000, 3000)
2. in                                 emp['job'].isin( [ 10, 20 ] )
3. is  null                           emp['comm'].isnull()
4. like                               emp['ename'].apply( lambda 함수 )

문제1. 월급이 1000 에서 3000 사이인 사원들의 이름과 월급을 출력하시오!

SQL>  select  ename, sal
          from emp
          where  sal between  1000  and  3000;

Pandas> 

문제2. 월급이 1000 에서 3000 사이가 아닌 사원들의 이름과 월급을 출력하시오.

힌트:  오라클 not 이고 R 은 !(느낌표) 이고 파이썬은 물결(~) 입니다. 

Pandas> 

문제3. 직업이 SALESMAN, ANALYST 인 사원들의 이름과 월급과 직업을
          출력하시오 !

SQL> select  ename, sal, job
         from  emp
         where  job in ('SALESMAN', 'ANALYST');

Pandas> 

문제4. 직업이 SALESMAN, ANALYST 가 아닌 사원들의 이름과 직업을 출력 ?

SQL> select  ename, sal, job
         from  emp
         where  job not in ('SALESMAN', 'ANALYST');

Pandas> 
  
문제5. 커미션이 null 인 사원들의 이름과 커미션을 출력하시오 !

SQL> select  ename, comm
          from  emp
         where  comm  is  null;

Pandas>

문제6. 커미션이 null 이 아닌 사원들의 이름과 커미션을 출력하시오 !

SQL> select  ename, comm
          from  emp
         where  comm  is not null;

Pandas> 
  
문제7. 위의 결과에서 1400.0 이 아니라 1400 으로 출력하고 싶다면 ?

Pandas> 
  
문제8. 이름의 첫번째 철자가 A 로 시작하는 사원들의 이름을 출력하시오 !

SQL> select  ename
          from  emp
          where   ename  like  'A%';

Pandas>  

설명: 
 1. apply 가 함수명입니다.
    emp['ename']  <-- emp 데이터 프레임의 ename 이라는 시리즈를 선택한다는 뜻

 2. emp['ename'].apply(함수명) <-- ename 데이터가 함수에 적용이 됩니다. 

 3. lambda 가 이름없는 한줄짜리 함수를 작성하겠다는 뜻입니다 

 4. lambda  입력값 :  출력값  

 5. lambda  x : x[0] =='A'  <-- x에 입력되는 어떤 문자의 첫번째 철자가 A 면 True 

Pandas>  emp.loc[ emp['ename'].apply( lambda  x : x[0] =='A') , ['ename'] ]

문제9.  아래의 SQL을 판다스로 출력하시오 !

SQL>  select   ename,  sal
          from  emp
         where   ename  like  '_M%';

Pandas>  

문제10.  아래의 SQL을 판다스로 출력하시오 !

SQL>  select  ename, sal
          from  emp
         where   ename  like  '%T'; 

Pandas>  

문제11. 아래의 SQL을 판다스로 출력하시오 !

SQL> select  ename, sal
        from  emp
        where  ename  like  '%M%';

Pandas>  

문제12. 아래의 SQL을 판다스로 출력하시오 ! 

SQL> select  ename, sal
        from  emp
        where  ename not  like  '%M%';

Pandas>   
