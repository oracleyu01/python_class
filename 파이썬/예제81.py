

▣  예제81. 연산자 축약 이해하기

  예제:  count = count + 1  

   이거를 이렇게 표현합니다. 
    
          count += 1   

  코드를 간결하고 가독성이 높게 하려고 축약 연산자를 사용합니다. 

예제1.  += 연산자

a = 10 
a += 5   # a = a + 5
print(a)  # 15 

예제2.  -= 연산자

b = 10
b -= 3    # b = b - 3
print(b)   # 7

예제3.  *= 연산자

c = 10
c *= 2
print(c)

예제4.  /= 연산자

d = 10
d /= 2
print(d)

문제1.   동전을 10000 번 던져서 앞면이 나오는 횟수를 출력하시오 !

import  random

coin = ['앞면', '뒷면']
cnt = 0

for  i  in  range(1, 10001):
    result = random.choice(coin)
    if  result =='앞면':
        cnt += 1
print(cnt)


문제2.  동전을 던졌을 때 앞면이 나오는 확률을 출럭하시오 ! 
          ( 동전을 10만번 던지세요)

import  random

coin = ['앞면', '뒷면']
cnt = 0

for  i  in  range(1, 100001):
    result = random.choice(coin)
    if  result =='앞면':
        cnt += 1
print(cnt/100000)

