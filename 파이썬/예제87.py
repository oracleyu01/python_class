
▣ 예제87. 한줄짜리 이름없는 함수 만들기(lambda)

https://cafe.daum.net/oracleoracle/Sp62/103

추가문제1.  아래의 코드를 람다를 이용한 코드로 변경하시오 ! 

numbers = [ 1, 2, 3, 4, 5 ]

def  square(x):
    return   x ** 2

result = list( map( square,  numbers ) )
print( result )

답:

추가문제2.   아래의 리스트를 받아서 아래의 결과로 출력되게하시오 !

numbers = [ 1, 2, 3, 4, 5 ]

결과예시:
print(result)

['홀수', '짝수', '홀수', '짝수', '홀수' ]

답:
numbers = [ 1, 2, 3, 4, 5 ]

def  check_even(num):
    if  num % 2 == 0:
        return '짝수'
    else:
        return '홀수'

result = list(map( check_even, numbers) )
print(result)

추가문제3. 위의 코드를 람다를 이용한 코드로 변경하시오 !
               ( comprehension 을 활용을 해서 수행을 해야합니다.)

numbers = [ 1, 2, 3, 4, 5 ]

result = list( map( lambda  num : '짝수'  if  num % 2 == 0 else  '홀수', numbers) )

print( result )

추가문제4.  아래의 numbers 리스트의 요소들을 아래의 결과로 출력하시오 !
               ( 람다 사용하지 않고 수행)

numbers = [ -3, 0, 7, -1, 5, 0, -8 ]

결과:
print(result)

['음수', '0', '양수', '음수', '양수', '0', '음수' ]

추가문제5. 위의 코드를 람다를 이용한 코드로 작성하시오 !

numbers = [ -3, 0, 7, -1, 5, 0, -8 ]

결과:
print(result)

['음수', '0', '양수', '음수', '양수', '0', '음수' ]

추가문제6. 아래의 numbers 리스트의 요소를 아래의 결과로 출력하는데
           람다를 이용한 한줄 코드로 작성하시오 !

numbers = [ 4,  5, 0, 12, 3, 0,  7  ]

print(result)

['짝수', '홀수', '0', '짝수', '홀수', '0', '홀수' ]

답:
# 원본 리스트
numbers = [4, 5, 0, 12, 3, 0, 7]

result = list(map(lambda num: '0' if num == 0 else ('짝수' if num % 2 == 0 else '홀수'), numbers))

# 결과 출력
print(result)  # 출력: ['짝수', '홀수', '0', '짝수', '홀수', '0', '홀수']

