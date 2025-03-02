
▣ 예제72. 들여쓰기 배우기

 파이썬에서는 실행 코드 부분을 묶어주는 괄호가 없습니다.
 괄호 대신 들여쓰기를 사용합니다.
 R 에서 for loop 문을 사용하거나 if 문을 사용할때 괄호를 사용해야
 했습니다.  그런데 파이썬은 들여쓰기로 블럭(실행영역)을 구분합니다.

예: 주사위를 10번 던져봅니다. 

import  random
dice = [ 1, 2, 3, 4, 5, 6 ]

for  i  in  range(10):  # i 인덱스 카운트에 0~9까지 담으면서 아래의 실행문을
    print( i , '실행문')   # 10번 실행하겠다. 


import  random
dice = [ 1, 2, 3, 4, 5, 6 ]

for  i  in  range(10):
    print( i, '번째 주사위의 눈은 ' , random.choice(dice) )  

설명: 파이썬은 위와 같이 loop 문 사용할때 괄호를 안쓰고 들여쓰기로
       실행문을 구분해주고 있습니다. 

문제1.  동전을 10번 던져서 다음과 같이 출력되게 하시오 !

답:

import  random
coin=['앞면', '뒷면']

for  i  in  range(10):
    print( i+1, '번째 동전의 결과는 ' , random.choice(coin) )
