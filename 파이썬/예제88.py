
▣ 예제88. 한줄로 코딩하기 comprehension

https://cafe.daum.net/oracleoracle/Sp62/131


리스트 컴프리핸션 추가문제1.  아래의 리스트에서 리스트 컴프리핸션을 이용해서
숫자 5보다 큰 숫자만 선택해서 리스트를 구성하시오 !

numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

print( result )

[ 6, 7, 8, 9, 10 ]

컴프리헨션 이용하지 않았을때:



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


* 셋 컴프리핸선 추가 문제1.
 아래의 리스트의 요소들의 철자의 길이중에 짝수인것만 셋 자료형에 결과로 담아 출력하시오

words = ["apple", "banana", "cherry", "date", "apple", "banana", "elderberry"]

결과 :  { 4, 6, 10 }

답:
 

■ 복습문제1. 주어진 리스트에서 각 문자열이 소문자 'a' 를 포함하고 있는지
    확인하고 포함하고 있는 문자열을 별도의 리스트에 담아 출력하시오 !
    (컴프리헨션 사용지 말고 출력)

words=['apple', 'banana', 'cherry', 'date', 'aricot', 'kiwi' ]

결과:  [ 'apple', 'banana', 'date', 'aricot' ]

답: 

■ 복습문제2.  이번에는 comprehension 을 이용해서 리스트로 출력하시오 !

words=['apple', 'banana', 'cherry', 'date', 'aricot', 'kiwi' ]

결과:  [ 'apple', 'banana', 'date', 'aricot' ]

답:


■ 복습 문제

복습문제7.아래의 리스트를 받아서 정렬해서 출력하는 sort_list 함수를 생성하세요

data = [5, 3, 8, 1, 4, 7, 2, 6]

# 함수 호출 및 결과 출력
sorted_list = sort_list(data)
print(sorted_list)

[1, 2, 3, 4, 5, 6, 7, 8]

답:

