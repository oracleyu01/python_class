
▩ 예제15.숫자_함수1_(round)

     오라클                vs                파이썬   
1.   round                                    round
2.   trunc                                    math.trunc
3.   mod                                       %

예제.        1  6  .  5  5  4 


print( round( 16.554 ) )  # 소수점 첫번째 자리에서 반올림해서 17을 출력
print( round( 16.554, 0 ) )  # 소수점 첫번째 자리에서 반올림해서 17.0을 출력
print( round( 16.554, 1 ) )  # 소수점 두번째 자리에서 반올림해서 16.6을 출력
print( round( 16.554, 2 ) )  # 소수점 세번째 자리에서 반올림해서 16.55을 출력

print( round( 16.554, -1 ) ) # 일의 자리에서 반올림해서 20.0 이 출력됨 

※ 파이썬에서 반올림할 때 중요하게 알아야할 내용 

print ( round( 142.5, 0 )  )  

   142       142.5       143  

예상과 다르게 결과가 143이 아니라 142 로 출력되었습니다. 

파이썬과 R 은 짝수를 좋아합니다. 

문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL>   select  ename, sal * 1.245  as  bonus 
             from  emp;

답:




문제2. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select  ename, round(sal*1.245)  as  bonus
            from   emp;

답:


설명:  .astype(int) 를 뒤에 붙이면 정수형으로 변환됩니다.


