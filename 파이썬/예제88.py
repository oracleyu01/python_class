
▣ 예제88. 한줄로 코딩하기 comprehension

https://cafe.daum.net/oracleoracle/Sp62/131


리스트 컴프리핸션 추가문제1.  아래의 리스트에서 리스트 컴프리핸션을 이용해서
숫자 5보다 큰 숫자만 선택해서 리스트를 구성하시오 !

numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

print( result )

[ 6, 7, 8, 9, 10 ]

컴프리헨션 이용하지 않았을때:

numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

result = []
for  i  in  numbers:
    if i > 5:
        result.append(i)
print(result)


딕셔너리 컴프리헨션 추가문제1.  아래의 딕셔너리의 요소들 중에서 뉴진스 노래들만
딕셔너리로 구성하시오 ! 

original_dict = {'아이유' : '너랑나', '뉴진스' : 'super shy', '마이클 잭슨' : 'beat it',
                     '뉴진스' : 'OMG',  '비틀즈' : 'let it be', '뉴진스' : 'how sweet' } 

결과 :  {'뉴진스' : 'supper shy', '뉴진스' : 'OMG', '뉴진스': 'how sweet' }

답:  딕셔너리에는 키가 유일한 키 하나만 구성되어야하므로 구현 안됩니다. 

* 딕셔너리 컴프리헨션 추가문제2. 아래의 딕셔너리에서 값이 10이상인 키와 쌍을 
  필터링하여 새로운 딕셔너리로 구성하시오 !

a = { '사과' : 8 , '바나나' : 15, '체리' : 5, '배' : 12, '망고' : 20 }

결과:  { '바나나' : 15, '배' : 12, '망고' : 20 }

답:
a = { '사과' : 8 , '바나나' : 15, '체리' : 5, '배' : 12, '망고' : 20 }
result = { key : value for key, value in a.items() if value >= 10 }
print(result)


* 셋 컴프리핸선 추가 문제1.(점심시간 문제) 
 아래의 리스트의 요소들의 철자의 길이중에 짝수인것만 셋 자료형에 결과로 담아 출력하시오

words = ["apple", "banana", "cherry", "date", "apple", "banana", "elderberry"]

결과 :  { 4, 6, 10 }

답:
   다 된 라인은 검사 받고 식사하러 가시면 됩니다. 

* 제너레이터를 사용한 함수 실행하기 

def generate_odd_squares():
    for x in range(1, 101):
        if x % 2 != 0:
            yield x ** 2   # 값을 반환할때 호출한 코드에 값을 반환을 하고
                              # 다음 return 을 만날때 일시 정지 합니다. 

# 제너레이터 실행 예시
odd_squares = generate_odd_squares()
for square in odd_squares:  # 제너레이트 함수를 호출해서 yield 를 통해 생성되는 값을
    print(square)  #   차례데로 반환합니다. 

* 제너레이터 추가문제.  제너레이터 함수가 일반 함수에 비해서 메모리가 절약이되는지
  비교하는 실험을 하시오 !

1. 일반함수

def  general_fun():
    return  [ x**2  for  x  in  range(1, 101)  if  x % 2 != 0 ]

result1 = general_fun()
print(result1)

2. 제너레이트 함수 

def  generate_fun():
    for  x  in  range(1, 101):
        if  x % 2 != 0:
            yield  x**2

result2 = generate_fun()
print( list(result2)) 

3. 메모리 사용 비교

import  sys

# 리스트 메모리 사용
result1 = general_fun()
list_memory = sys.getsizeof(result1)     # getsizeof 는 메모리 사용량을 볼 수 있는 함수
print( list_memory)   # 472

#제너레이트 메모리 사용
result2 = generate_fun()
generator_memory = sys.getsizeof(result2)
print(generator_memory)   # 200

대량의 데이터를 처리할 때는 제너레이트 함수를 사용하는게 더 효율적입니다. 

■ 복습문제1. 주어진 리스트에서 각 문자열이 소문자 'a' 를 포함하고 있는지
    확인하고 포함하고 있는 문자열을 별도의 리스트에 담아 출력하시오 !
    (컴프리헨션 사용지 말고 출력)

words=['apple', 'banana', 'cherry', 'date', 'aricot', 'kiwi' ]

결과:  [ 'apple', 'banana', 'date', 'aricot' ]

답: 
words=['apple', 'banana', 'cherry', 'date', 'aricot', 'kiwi' ]

result = []
for  i  in  words:
    if 'a' in  i:
        result.append(i)

print(result)

■ 복습문제2.  이번에는 comprehension 을 이용해서 리스트로 출력하시오 !

words=['apple', 'banana', 'cherry', 'date', 'aricot', 'kiwi' ]

결과:  [ 'apple', 'banana', 'date', 'aricot' ]

답:
words=['apple', 'banana', 'cherry', 'date', 'aricot', 'kiwi' ]

result = [ i  for  i  in  words  if  'a'  in  i ]
print(result)

■ 복습 문제

복습문제7.아래의 리스트를 받아서 정렬해서 출력하는 sort_list 함수를 생성하세요

data = [5, 3, 8, 1, 4, 7, 2, 6]

# 함수 호출 및 결과 출력
sorted_list = sort_list(data)
print(sorted_list)

[1, 2, 3, 4, 5, 6, 7, 8]

답:

