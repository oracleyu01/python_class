
▩ 예제1. 판다스 모듈이란 ?

 판다스(pandas) 모듈 ?

1. csv 파일이나 엑셀 파일의 데이터를 검색하기 편하도록 구현한 
  파이썬 모듈과 함수를 모아놓은 코드들의 집합 
2. 데이터 시각화 함수들이 내장되어 있습니다
3. 데이터 분석함수들이 내장되어 있습니다.
4. 머신러닝 함수들이 내장되어 있습니다. 

예제. emp 데이터 프레임에서  이름과 월급을 출력하시오 !

문법:  emp[ 컬럼명 ] [검색조건]

import  pandas  as  pd   # 판다스 모듈을 불러옵니다. 

emp = pd.read_csv("d:\\data\\emp.csv")
emp[ ['ename', 'sal'] ]

▩ 예제2. 특정 컬럼 출력하기 

문법:  데이터 프레임명 [  ['컬럼명1', '컬럼명2'] ]

SQL> select  ename, sal
          from  emp;

Pandas>  emp[ ['ename', 'sal']  ]

문제1. 사원이름, 월급, 직업, 부서번호를 출력하시오 !

Pandas> emp[ ['ename', 'sal', 'job','deptno'] ]

