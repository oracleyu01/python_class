▣  예제80. while loop문

 while loop 문은 조건이 참인 동안 계속해서 반복 실행하는 루프문입니다.
 조건이 거짓이 되면 루프문이 종료됩니다. 
 while 루프는 반복 횟수가 정해져 있지 않거나 특정 조건이 만족될때까지
 반복을 수행해야 할 때 유용합니다.

예제1.  숫자 1부터 10까지 출력하는 반복문을 작성하시오 !

      for   loop  문                  vs               while    loop 

  for  i   in   range(1, 11):                          x = 1
      print( i )                                       while   x < 11:
                                                           print(x)
                                                           x = x + 1 

예제2.  1 ~ 10번까지 출력하는 while loop 문을 수행하시오 !

x = 1
while  x < 11:
    print(x)
    x = x + 1 

예제3.  위의 예제에서 x = x + 1 을 빼고 수행하면 어떻게 되겠는가 ?

작업관리자를 먼저 엽니다. 

x = 1
while  x < 11:
    print(x)

while  loop 문을 작성할 때 무한 루프가 돌아가지않도록 주의 해서 코딩하셔야합니다.

예제4. while loop문의 장점이 무엇인가요 ?

무한 루프가 단점이자 장점입니다.  

while True:
    print( """ <초간단 mbti  테스트 하기>
    
     문제가 생겼을 때 당신의 대처 방법은?
              """  )
    
    q1  =  input("Q1. 말이 많아진다면 E, 생각이 많아진다면 I, 선택해봐유?   ")
    q2 =  input("Q2. 그냥 그런가보다면 S, 어떻게 그럴수 있지 N , 선택해봐유?  ")
    q3 =  input("Q3. 이해는 안되는데 공감은 된다면 F, 이해가 되야 공감을 하든지 말든지 하면 T, 선택해봐유? " )
    q4 = input("Q4. 나는 한다하면 하면 J, 뭐부터 해야하는겨 하면 P,  선택해봐유? ")
    
    print("                ")
    result = q1+q2+q3+q4 
    print(  '당신의 mbti 는 '  +  result   + ' 입니다'   )

예제5. 위의 질문중에  중지시키겠습니까? 라는 질문을 넣고 yes 라고 하면
      위의 프로그램이 중지되게 코드를 수정하시오 !
     
   앞의 코드들 .... 

    print(  '당신의 mbti 는 '  +  result   + ' 입니다'   )

    stop = input('중지 시키겠습니까? (yes or no): ')
    if stop=='yes':
        break

