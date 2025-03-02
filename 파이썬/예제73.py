
▣  예제73. if 개념 배우기 ① (if~else)

 어떤 조건을 참과 거짓으로 판단할 때 if 문을 사용합니다.
 참과 거짓을 구분하여 코드를 실행하다 보면 if ~ else 문을 사용합니다.
 코드를 작성하다 보면 조건에 따라 수행하는 일이 달리 해야하는
 경우가 있습니다. 조건이 참인지 거짓인지 검사를 하고 참인 경우에는
 이 일을 하고 거짓인 경우에는 저일을 해라 !

■ if 문법

if  조건:              # 만약 조건1 이 True 라면
    실행코드1       # 실행코드 1을 실행해라 
else:                  # 만약 조건1 이 False 라면
    실행코드2       # 실행코드 2를 실행해라!

예제1. 주사위를 10번 던지시오 !



예제2. 10을 2로 나눈 나머지값을 출력하시오 !

10 % 2

예제3. 주사위를 10번 던지는데 주사위의 눈이 짝수 일때만 출력하시오 !

import random
dice = [ 1, 2, 3, 4, 5, 6 ]

for  i  in  range(10):            #  아래의 실행문을 10번 반복하는데
    a = random.choice(dice)  #  주사위를 던져서 주사위의 눈이
    if  a % 2 == 0:              #  짝수면
        print(a)                    #  출력해라 

예제4. 주사위를 20번 던져서 홀수 일때만 출력하시오 !

import random
dice = [ 1, 2, 3, 4, 5, 6 ]

for  i  in  range(20):            
    a = random.choice(dice)  
    if  a % 2 == 1:             
        print(a)      

예제5. if ~ else 문을 사용해서 주사위의 눈을 출력할 때
        이 눈이 짝수인지 홀수 인지가 같이 출력되게 하시오 !

import random
dice = [ 1, 2, 3, 4, 5, 6 ]

for  i  in  range(20):            
    a = random.choice(dice)  
    if  a % 2 == 0:             
        print(a, '는 짝수 입니다')      
    else:
        print(a, '는 홀수 입니다')
        
문제1. 동전을 10번 던져서 아래와 같이 출력되게하시오 !

import random
coin = ['앞면', '뒷면']

for i in range(10):
    a = random.choice(coin)
    print( a, ' 입니다')
