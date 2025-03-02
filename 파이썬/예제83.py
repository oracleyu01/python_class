▣ 예제83. 리스트 함수 총정리 

▣ 예제83_1. 리스트 함수 이해하기1 (슬라이싱)

 리스트에서 특정 요소들을 추출할때 슬라이싱을 사용합니다.

예제1. 다음 a 리스트에서 '바위' 를 잘라내서 출력하시오 !

a = ['낙' ,'숫', '물', '이', '바', '위', '를', '뚫', '는', '다' ]

결과:  ['바', '위']

답:  

문제1.  아래의 a 리스트에서 '뚫는다' 를 잘라내시오

a = ['낙' ,'숫', '물', '이', '바', '위', '를', '뚫', '는', '다' ]

결과: [ '뚫', '는', '다' ]

답:

▣ 예제83_2. 리스트 함수 이해하기2 (append)

리스트 변수에 맨 끝의 요소로 값을 추가할 때는 append 함수를 사용합니다.

a = [ 1000, 2000, 3000, 4000 ]



예제1. 아래의 a 리스트에서 짝수 데이터만 b 라는 리스트에 담아내고 출력하시오!

a = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]




문제2. 아래의 a 리스트의 데이터를 정제해서 b 리스트에 담고 b 리스트를 
         출력하시오 !

a = [ '1.끝내주게 숨쉬기.' , '2.간지나게 자기!', '3.작살나게 밥먹기?' ]

print(b)

 ['끝내주게 숨쉬기',  '간지나게 자기', '작살나게 밥먹기']  

답: 




문제3. 아래의 리스트에서 '정상품' 만 b 리스트에 담아서 출력하시오 !

a = ['정상품', '정상품', '불량품', '정상품', '불량품', '정상품', '정상품', '불량품']

print(b)

['정상품', '정상품', '정상품', '정상품', '정상품'] 

답: 




▣ 예제83_3. 리스트 함수 이해하기3 (insert, extend)

* 리스트에 요소값 추가하는 3가지 함수 ?

 1. append  :  리스트 마지막에 요소명을 추가
 2. insert    :   리스트에 특정 위치에 요소를 추가 
 3. extend  :    리스트에 여러개의 요소들을 한번에 추가

예제1. 아래의  dice 리스트의 마지막 요소로 6을 추가하시오 !

dice = [ 1, 2, 3, 4, 5 ]



예제2.  아래의 dice 리스트의  숫자 4를 순서에 맞춰서 입력하시오 !

dice = [ 1, 2, 3, 5, 6 ]



문제1.  아래의 동전 리스트에 앞면을 추가하시오 !

coin = [ '뒷면' ]

print( coin )

 ['앞면', '뒷면' ]

답: 

예제3. 다음과 같이 a 리스트와 b 리스트가 있습니다. 
      a 리스트의 요소로 b 리스트의 요소들을 한번에 추가하세요.

a = [ 1000, 2000, 3000, 4000 ]
b = [ 5000, 6000, 7000, 8000 ]

결과:  [ 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000 ]

답:  

문제2.  다음 animals 리스트의 요소로 mammals 요소들을 추가시키시오 !

animals =['dog', 'cat']
mammals =['tiger', 'elephant'] 

답: 


문제3.  animals 리스트의 요소로 mammals 요소들을 추가하는데 앞쪽에 추가하시오

animals =['dog', 'cat']
mammals =['tiger', 'elephant'] 

결과: 
print(animals)
[ 'tiger', 'elephant', 'dog', 'cat' ]

답:    



▣ 예제83_4. 리스트 함수 이해하기4 (리스트 요소 정렬)

 1. 리스트.sort()     :  리스트의 요소들을 실제로 정렬시킴
 2. sorted(리스트명) : 리스트의 요소들은 그대로 두고 리스트를 정렬된 상태로 출력
 3. 리스트.reverse() :  리스트의 요소들을 실제로 역순으로 정렬 시킴
 4. reversed(리스트) :  리스트의 요소들은 그대로 두고 리스트의 요소들을 
                             역순으로 정렬된 상태로 출력

예제1.  다음 a 리스트의 요소들을 정렬해서 출력하시오 !

a = [ 2000, 1000, 4000, 5000, 3000 ]

결과: [ 1000, 2000, 3000, 4000, 5000]

답:



예제2.  다음 a 리스트의 요소들을 역순으로 정렬해서 출력하시오 !

a = [ 2000, 1000, 4000, 5000, 3000 ]

답:



예제3. 다음 a 리스트의 요소들을 정렬된 상태로 출력하시오 !

a = [ 2000, 1000, 4000, 5000, 3000 ]

print( sorted(a) )
print( a )

문제1.  아래의 fruits 리스트의 데이터를 정렬해서 fruits2 라는 리스트를 구성하시오!

fruits = ['carrot', 'banana', 'lemon', 'apple', 'guaba']

print( fruits2 )

결과: [ 'apple', 'banana', 'carrot', 'guaba', 'lemon'] 

답:



예제4.  아래의 a 리스트의 요소들을  역순으로 정렬해서 출력하시오 !

a = [ 2000, 1000, 4000, 5000, 3000 ]

a.reverse()
print(a)

문제2.  다음의 결과를 역순으로 정렬해서 출력하시오 !

b = [ '하', '중', '상' ]

print(b)  # ['상', '중', '하' ] 

답:



예제5. 아래의 리스트의 요소들은 그냥 그대로 두고 역순으로 정렬된 상태로 출력하시오

b = [ '하', '중', '상' ]



문제3. 다음 best_reaction 리스트의 요소들을 그냥 그대로 두고 
        best_reaction2 라는 요소들을 역순으로 정렬해서 구성하시오

best_reaction = [ '진짜?', '대박!', '헐~']

print( bese_reaction2 )

결과: [ '헐~', '대박!', '진짜?'] 


▣ 예제83_5. 리스트 함수 이해하기5 (리스트의 요소 찾기)

* 리스트에서 요소를 검색하는 함수는 2가지가 있습니다.

 1. 리스트.count('요소명') : 리스트안의 요소명의 갯수를 출력
 2. 리스트.index('요소명') : 리스트안의 요소명의 인덱스 번호 출력

예제1. 다음 box 리스트에서 불량품이 몇개가 있는지 출력하시오 !

box = ['정상품', '정상품', '정상품', '불량품', '정상품', '불량품']

결과 : 2



예제2. 다음 box 리스트에서 불량품의 자리번호를 출력하시오 !

box = ['정상품', '정상품', '정상품', '불량품', '정상품', '불량품']



※ 두번째 불량품 자리번호를 출력하려면 어떻게 해야하는가 ?  enumerate 함수 사용

예제3.  enumerate 함수 이용하기 

box = ['정상품', '정상품', '정상품', '불량품', '정상품', '불량품']



문제1. 아래의 box 리스트에서 불량품의 자리번호 2개를 출력하시오 !
        if 문을 이용해서 수행하시오~

box = ['정상품', '정상품', '정상품', '불량품', '정상품', '불량품']



결과: 4, 6

답:


문제2. 아래의 코드를 수행해서 d 리스트를 생성하시오 !

winter = open('c:\\data\\winter.txt')
data = winter.read()
data2 = data.split()
d = [ ]

for i in data2:
    st = i.strip(' 1234567890.,?!-/~*^"@#$%&')[0:4]
    if st.lower() == 'elsa':
        d.append(i)

print(d)
print(len(d))

위의 결과는 SQL로는 절대 할 수 없는 데이터 분석입니다. 
SQL의 regexp_count 함수가 있긴한데 위의 파이썬 코드 처럼 다양한 elsa 단어들을
볼 수 있습니다. 

* 데이터의 종류 3가지 ?
 
 1. 정형화된 데이터  :    테이블   <------------- SQL 로 분석 가능 
 2. 비정형화된 데이터 :  텍스트, 영상, 사진  <----   파이썬 , R 로 분석가능
 3. 반정형화된 데이터  :   xml, html 과 같이 키와 값으로 이루어진 데이터

문제3.  d 리스트 안에  elsa가  몇개가 있는지 카운트 하시오 !

winter = open('c:\\data\\winter.txt')
data = winter.read()
data2 = data.split()
d = [ ]




문제4.  d 리스트 안에  elsa’s   가 몇개가 있는지 출력하시오 !



※ 리스트의 count 함수로는 count 할 수 없는 요소를 카운트 하는 방법

문제5. 아래의 리스트에서 이름에 영희가 포함된 학생은 몇명인지 출력하시오!

name = ['김인호', '최영희', '안상수', '윤성식','김영희']

name.count('영희')  # 0   

답:



▣ 예제83_6. 리스트 함수 이해가기6 (리스트의 요소 지우기)

 1. 리스트.remove('요소명')  : 리스트의 요소를 요소명으로 삭제합니다.
 2. del 리스트[인덱스 번호] : 리스트의 요소를 자리번호로 삭제합니다. 
 3. 리스트.clear()    :  리스트의 모든 요소들을 다 삭제합니다.

문제1. 다음 box 리스트에서 요소명이 불량품을 제거하시오! 

box =['정상품', '정상품', '정상품', '정상품', '불량품', '정상품', '불량품']

답:


문제2. 아래의 box 리스트에서 맨뒤에 있는 '불량품' 을 지우시오 !

box =['정상품', '정상품', '정상품', '정상품', '불량품', '정상품', '불량품']

답:


문제3. (난이도 중) 아래의 box 리스트에서 요소명 불량품을 모두 지우시오!

box =['정상품', '정상품', '정상품', '정상품', '불량품', '정상품', '불량품']

힌트:  del 과 enumerate 와 for loop문을 활용하면 됩니다. 

print(box) 

['정상품', '정상품', '정상품', '정상품',  '정상품']

답:   

▣ 예제83_7. 리스트 함수 이해하기7 ( len 과 sum)

 1. len(리스트)  :  리스트의 요소들의 갯수를 출력
 2. sum(리스트) :  리스트의 요소들의 합을 출력

예제1. 다음 box 리스트의 요소들의 갯수를 출력하시오 !

box =['정상품', '정상품', '정상품', '정상품', '불량품', '정상품', '불량품']



예제2. 아래의 리스트의 요소들의 합을 출력하시오 !

box2 = [ 1000, 2000, 3000, 4000, 5000 ]



문제1. 다음의 몸무게 데이터의 평균값을 출력하시오!

weight=[ 72, 81, 90, 78, 84, 65 ]



문제2. 위의 결과를 numpy 를 이용해서 출력하시오 ! 

weight=[ 72, 81, 90, 78, 84, 65 ]

답:



▣ 예제83_8. 리스트 함수 이해가기8 ( map 과 filter )

 1. map( 함수명, 리스트 ) : 리스트의 요소들을 함수에 적용
 2. filter( 함수명, 리스트 ) : 리스트의 요소들을 함수에 적용 

예제1.  다음 weight 리스트의 요소들을 가지고 다음의 결과를 출력하시오!

weight=[ 72, 81, 90, 78, 84, 65 ]

결과 : ['정상', '비만', '비만', '정상', '비만', '정상' ]

답:


결과: ['정상', '비만', '비만', '정상', '비만', '정상']

문제1.   아래의 score  리스트를 만들고 다음의 결과가 출력되게 하시오 !

score=[ 78, 92, 23, 54, 67, 88 ]

결과: [ '중', '상', '하', '중', '중', '상' ]

80점이상이면 '상'
50점이상 ~ 80미만이면 '중'
50미만이면 '하' 

답:



예제2.  (filter 함수 예제)  다음 weight 리스트에서 요소가 80 이상인것만 
          별도의 리스트 result 에 담아 출력하시오 !

weight=[ 72, 81, 90, 78, 84, 65 ]

결과:  [ 81, 90, 84 ] 

답: 



위의 결과를 보면 None 이 출력되고 있습니다.  그래서 None 이 출력되지 
않게하려면 filter 함수를 이용하면 됩니다. 




※ 이게 map 과 filter 의 차이 입니다. 

문제1.  다음의 온도 리스트에서 특정 범위의 온도만 추출하여 새로운 리스트로
          반환하는 코드를 작성하시오.

          온도가 28도 이상 33도 미만인 경우에만 추출해야합니다.

temp =[ 22, 28, 34, 29, 30, 25, 31, 33, 27, 26, 35, 32, 24 ]

결과: [ 28, 29, 30, 31, 33, 32 ]

답:



▣ 예제83_9. 리스트 함수 이해하기9 ( zip 과 enumerate )

1. zip(리스트1, 리스트2) : 리스트1과 리스트2의 요소들의 순서에 따라 짝지어주는 함수

2. enumerate(리스트) : 리스트의 요소들을 인덱스 번호와 함께 짝지어주는 함수 

예제1. 다음의 2개의 리스트의 요소를 짝지어 출력하시오 !

weight= [ 71, 81, 90, 78, 84, 65 ]
result = ['정상', '과체중', '비만', '정상', '과체중', '정상' ]


문제1. 아래의 리스트 3개를 짝지어 출력하시오 !

name = ['김인호', '안상수', '이상식', '오연수', '강인식', '고성인'] 
weight= [ 71, 81, 90, 78, 84, 65 ]
result = ['정상', '과체중', '비만', '정상', '과체중', '정상' ]


문제2. 위에서 출력되고 있는 결과를  all_data 라는 비어있는 리스트를 만들고
         append 시키시오 !

name = ['김인호', '안상수', '이상식', '오연수', '강인식', '고성인'] 
weight= [ 71, 81, 90, 78, 84, 65 ]
result = ['정상', '과체중', '비만', '정상', '과체중', '정상' ]



문제3. 위에 all_data 리스트에 있는 결과를 예쁘게 표형태로 출력하시오 !

from  tabulate  import  tabulate


예제2. enumerate  함수를 이용하여 아래의 name 리스트를 번호를 붙여서 출력하시오

name = ['김인호', '안상수', '이상식', '오연수', '강인식', '고성인'] 



문제4. 아래의 리스트를 이용해서 다음과 같이 결과를 출력하시오 !

name = ['김인호', '안상수', '이상식', '오연수', '강인식', '고성인'] 
weight= [ 71, 81, 90, 78, 84, 65 ]
result = ['정상', '과체중', '비만', '정상', '과체중', '정상' ]

답:


