
▣  예제76. for loop 문 + 카운트 하기

 for 루프문과 카운트 하기는 반복문을 사용하여 특정 조건을 만족하는
 항목의 수를 세는 작업입니다. 이를 통해 리스트, 문자열 또는 다른 시퀀스에서
 특정 조건을 만족하는 항목이 얼마나 있는지 확인을 할 수 있습니다.

예제1. 리스트에서 짝수의 갯수를 세기 

num = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
count = 0 

for  i   in   num:
    if  i % 2 == 0:
        count =  count + 1 

print ('짝수의 갯수는' , count )

문제1. 동전을 10번 던져서 앞면 또는 뒷면을 출력하시오 !

import  random

coin = ['앞면', '뒷면']

for  i   in  range(1, 11):
    result = random.choice(coin)
    print( result )
    
문제2.  동전을 10번 던져서 앞면이 나오는 횟수를 출력하시오 !

import  random

coin = ['앞면', '뒷면']
count = 0   

for  i   in  range(1, 11):
    result = random.choice(coin)
    if  result =='앞면':
        count = count + 1

print( count )

