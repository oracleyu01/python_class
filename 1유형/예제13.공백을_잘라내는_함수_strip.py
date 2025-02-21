
▩ 예제13.공백을_잘라내는_함수_strip

문법:  strip 함수를 사용해서 공백을 잘라낼 수 있습니다.

함수                  설명 
lstrip()             문자열에서 존재하는 왼쪽 공백을 제거 
rstrip()             문자열에서 존재하는 오른쪽 공백을 제거 
strip()              문자열에서 존재하는 양쪽 공백을 제거 

예제.  

text1 = '     A  story  is  2025      '  
print(text1)
print( text1.lstrip()  )
print( text1.rstrip()  )
print( text1.strip()   )

문제1.  아래의 SQL을 판다스로 구현하시오 !

insert  into  emp(empno, ename, sal )
  values( 9911, '  JACK  ',  3000);

답:




설명: 일부러 이름에 JACK 을 양쪽 공백을 넣어서 입력했습니다.

문제2.  이름이 JACK 인 사원의 이름과 월급을 출력하시오 !

SQL> select  ename, sal
          from  emp 
          where  trim(ename)='JACK';

Pandas> 


자주 사용하는 문자함수를 총정리: 

1.upper()
2.lower()
3. slice()
4. replace()
5. len()
6. find()
7. strip()

판다스에서 위의 함수를 사용하려면 str 과 함께 사용해야합니다. 
