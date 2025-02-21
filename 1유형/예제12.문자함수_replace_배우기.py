
▩ 예제12.문자함수_replace_배우기

문법:  파이썬의 내장함수인 replace 는 문자열에서 특정 철자를 다른 철자로
       변경하는 함수 입니다. 



문제1. 사원 테이블의 구조를 확인하시오 !



#   Column     Non-Null Count  Dtype         
---  ------     --------------  -----         
 0   empno      14 non-null     int64         
 1   ename      14 non-null     object        
 2   job        14 non-null     object        
 3   mgr        13 non-null     float64       
 4   hiredate   14 non-null     datetime64[ns]
 5   sal        14 non-null     int64         <---    숫자형 컬럼  
 6   comm       4 non-null      float64       
 7   deptno     14 non-null     int64         
 8   ename_len  14 non-null     int64    

문제2.  emp 에 sal 을 문자형으로 변환하시오 !




 sal        14 non-null     object       <--- 문자형으로 변환되었습니다.

문제3. 아래의 SQL을 파이썬으로 변경하세요.

SQL>  select  ename, replace( sal, 0, '*')
           from  emp; 

답:



