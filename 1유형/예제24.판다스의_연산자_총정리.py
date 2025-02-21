
▩ 예제24.판다스의_연산자_총정리.py 

* 연산자의 종류 3가지 ?

 1. 산술 연산자
                   연산        오라클        판다스 
                   덧셈            +             +
                   뺄셈             -             -
                   곱셈             *             *
                  나눗셈            /            / 
                  나머지          mod          % 
                  거듭제곱      power(a,b)   a**b 

 2. 비교 연산자
                 연산              오라클          판다스 
                  같음              =                ==
                같지않음           !=               !=
                  크다             >                >
               크거나 같다          >=              >=
                  작다             <                <
               작거나 같다          <=               <=
                null 비교          is null            isnull()
               범위내 비교      between a and b    between(a,b)
               목록 존재        in  ( a,b,c)            isin([a,b,c])

 3. 논리 연산자 
                연산              오라클             판다스 
               그리고               and                 &
                또는                or                  |
               부정                 not                 ~

예제. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, sal,  deptno 
         from  emp
         where  sal >= 1200   and  deptno in (10,20) ;

답: 



문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL> select   ename, sal, job, comm
           from   emp
           where   sal  between  1000  and  3000  
              and  comm   is  not  null ;

답:



문제2. employee_data.csv 파일에서 date_hired 가 2020년 1월 1일을 포함한 
      그 이후이고 country 가 'United States' 인 행의 수를 출력하시오 !
      (빅분기 4회 시험 작업형1번 기출문제) 

데이터: https://cafe.daum.net/oracleoracle/Sp62/714

답:  



※ & (and) 를 쓸때는 양쪽에 소괄호로 조건을 둘러줘야합니다. 

