
▣ 예제75. for loop 문 

 for  루프문은 특정 시퀀스(리스트, 튜플, 문자열) 의 각 항목에 대해
 반복적인 작업을 수행할 때 사용됩니다.
 for 루프문을 이용하면 항목을 순차적으로 가져와서 블럭(block) 내에서
 실행합니다.  반복이 끝나면 루프가 종료됩니다.

예제1. 문자 반복

fruits = ['apple', 'banana', 'cherry']
for  i   in  fruits:
    print( '과일: ' +  i  )

예제2.  문자열 반복 

word = 'python'

for  i  in  word:
    print( '철자:' + i ) 

예제3. 숫자 범위 반복

for  i  in  range(5):  # 0 ~ 4까지의 숫자를 반복
     print( '숫자: ' + str(i)  )

문제1. 숫자 1번부터 10번까지 출력하시오 ~

for   i   in   range(1, 11):  # 1번부터 11 미만까지 
    print( i )

문제2. 위의 결과를 다시 출력하는데 가로로 출력하시오 !

for   i   in   range(1, 11):  # 1번부터 11 미만까지 
    print( i, end="  " )  # 출력사이에 공백을 두고 줄 바꿈을 하지 않도록 하겠다.

문제3. 어제 배운 if 문을 이용해서 위의 결과를 다시 출력하는데 짝수만 출력하시오.

for   i   in   range(1, 11):  # 1번부터 11 미만까지
    if  i % 2 == 0 :
        print( i, end="  " )
