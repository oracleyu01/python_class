
▩ 예제3. 특정 검색 조건에 해당하는 데이터 검색하기 

문법:  데이터 프레임명[ ['컬럼명1', '컬럼명2']] [ 검색조건 ]

예제:  월급이 3000 이상인 사원들의 이름과 월급을 출력하시오 !

SQL> select  ename, sal
         from  emp
        where  sal  >= 3000;

Pandas> emp[ ['ename', 'sal'] ] [ emp['sal'] >= 3000 ]

※ 비교 연산자 총정리 

  오라클    vs   파이썬 
      >              >
      >=            >=
     <               <
      <=           <=
      =             ==
      !=             !=

문제1. 직업이 SALSMAN 인 사원들의 이름과 월급과 직업을 출력하시오 !

emp[ ['ename, 'sal', 'job'] ] [ emp['job']=='SALESMAN']

※ 판다스의 데이터 검색하는 방법 3가지 ?

1. 첫번째 방법 :  emp[ 컬럼 리스트 ] [ 검색조건 ]

emp[ ['ename, 'sal', 'job'] ] [ emp['job']=='SALESMAN']

2. 두번째 방법 :  emp.loc[ 검색조건, 컬럼 리스트 ]

emp.loc[ emp['job']=='SALESMAN', ['ename', 'sal'] ] 

3. 세번째 방법: emp.iloc[ 검색조건, 컬럼 번호 ]

emp.iloc[ (emp['job']=='SALESMAN').values, [1, 5] ]

※ 파이썬은 1부터 시작하지 않고 0부터 시작합니다. 

문제2. 부서번호가 20번인 사원들의 이름과 월급과 부서번호를 출력하시오!

SQL> select  ename, sal, deptno
          from emp 
          where  deptno = 20;

Pandas> emp.loc[ emp.deptno == 20, ['ename', 'sal', 'deptno'] ]

