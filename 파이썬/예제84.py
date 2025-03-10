▣ 예제84. 딕셔너리 자료형

▣ 예제84_1. 딕셔너리 자료형 이해하기1﻿

 
  키(key) 와 값(value) 으로 구성된 자료 구조를 딕셔너리라고 합니다. 

  딕셔너리의 필요성 ?

   1.  빠른 데이터 검색 : 키를 가지고 값을 빨리 찾을 수 있습니다.
    
       리스트 자료형보다 데이터를 체계적으로 잘 정리해서 저장을 합니다. 
       정리가 잘되어있으니까 데이터 검색을 빠르게 할 수 있습니다. 

   2. 딕셔너리 자료형으로 테이블이 가능

      오라클의 테이블이 파이썬과 R 에서는 data frame 이라고 합니다. 
      판다스 데이터프레임으로 만들어놓기만 하면 너무 쉽게 검색되고
      너무 쉽게 시각화 할 수 있습니다. 


a = { '마이클 잭슨' : '하루 6시간씩 꾸준히 춤 연습을 했어요.',
      '김연아' : '공중 세바퀴 회전 1번 실패하면 65번씩 연습했어요.',
      '박태환' : '하루에 15,000km 이상 수영해야 세계적인 선수들과 겨룰 수 있어요' }


예제1. 김연아 명언만 출력하시오 !



예제2. a 딕셔너리에서 key 값들만 출력하시오 !



예제3. a 딕셔너리에서 values 들만 출력하시오 !



예제4. a 딕셔너리에서 key 와 값을 같이 출력하시오 !



예제5. a 딕셔너리에 새로운 데이터(키와 값) 을 추가하시오 ! 

a = { '마이클 잭슨' : '하루 6시간씩 꾸준히 춤 연습을 했어요.',
         '김연아' : '공중 세바퀴 회전 1번 실패하면 65번씩 연습했어요.',
       '박태환' : '하루에 15,000km 이상 수영해야 세계적인 선수들과 겨룰 수 있어요' }



문제1.  a 딕셔너리에 양궁선수 안산 선수의 명언을 입력하시오 !

 키: 안산
 값: 쫄지말고 대충쏴 ~

문제2. 다음은 우리 반 학생들의 과목별 점수를 저장한 딕셔너리입니다:

student_scores = {
   '김철수': {'국어': 85, '수학': 90, '영어': 78},
   '이영희': {'국어': 92, '수학': 88, '영어': 95},
   '박지성': {'국어': 75, '수학': 82, '영어': 90},
   '최유리': {'국어': 88, '수학': 95, '영어': 70},
   '정민호': {'국어': 94, '수학': 80, '영어': 85}
}

문제:
각 학생의 평균 점수를 계산하여 '학생이름: 평균점수' 형태의 새로운 딕셔너리를 만드세요.

결과: 
{'김철수': 84.3, '이영희': 91.7, '박지성': 82.3, '최유리': 84.3, '정민호': 86.3}



▣ 예제84_2. 딕셔너리 자료형 이해하기2

 딕셔너리의 값이 여러개인 경우, 리스트 형식으로 구성하면 됩니다.

예제1.  파이썬 개발자 금융권 단가를 딕셔너리로 구성하시오 !

salary = { '초급' : [ '500만원~600만원', '0~3년'],
            '중급' :  ['600만원~700만원', '3~6년'],
            '고급' :  ['700만원~900만원', '6~9년'] 
            }

salary

예제2.  위의 salary 리스트에 특급을 추가하시오 


문제1. 다음의 딕셔너리에 새로운 직업을 키로 하고 아래의 리스트를 값으로 해서
        추가하시오 !

job = { '사라질 직업' : ['텔레마케터', '법률보조원'] }

키: 새로운 직업
값: ['자동화 구현 개발자', '딥러닝 개발자'] 


예제3. 위의 job 리스트에 새로운 직업키에 직업을 추가하기



예제4. 위의 job 리스트에서 사라질 직업에 운전기사를 추가하시오 !



예제5. 다시 운전기사를 삭제하시오 !



문제1. 식당에서 메뉴판을 관리하는 프로그램을 만들려고 합니다.
       다음과 같이 카테고리별로 메뉴와 가격이 저장된 딕셔너리가 있습니다:

menu = {
   '한식': [
       {'이름': '비빔밥', '가격': 8000, '인기여부': True},
       {'이름': '된장찌개', '가격': 7000, '인기여부': True},
       {'이름': '불고기', '가격': 12000, '인기여부': True}
   ],
   '중식': [
       {'이름': '짜장면', '가격': 6000, '인기여부': True},
       {'이름': '짬뽕', '가격': 7000, '인기여부': False},
       {'이름': '탕수육', '가격': 15000, '인기여부': True}
   ],
   '일식': [
       {'이름': '초밥', '가격': 20000, '인기여부': True},
       {'이름': '라멘', '가격': 9000, '인기여부': False}
   ]
}

다음 작업을 수행하는 코드를 작성하세요:

1. '양식' 카테고리를 추가하고, 다음 메뉴들을 등록하세요:
  - 파스타: 13000원, 인기메뉴
  - 피자: 18000원, 인기메뉴 아님
  - 스테이크: 25000원, 인기메뉴

2. '중식' 카테고리에 '마라탕' 메뉴를 추가하세요 (가격: 12000원, 인기메뉴)

3. 모든 메뉴 중 가격이 10000원 이상인 메뉴의 이름과 가격을 출력하세요.

이름	가격	카테고리
------------------------------
불고기	12000원	한식
탕수육	15000원	중식
마라탕	12000원	중식
초밥	20000원	일식
파스타	13000원	양식
피자	18000원	양식
스테이크	25000원	양식

▣ 예제84_3. 딕셔너리 자료형 이해하기3

 여러개의 리스트의 데이터를 한번에 딕셔너리로 구성하기
 여러 비정형화된 데이터를 손쉽게 테이블 형태로 구성하기 위한 준비작업(데이터 전처리)

예제1. 아래의 두개의 리스트를 가지고 딕셔너리를 구성하시오!

artist = ['비틀즈', '비틀즈', '아이유', '아이유', '마이클 잭슨', '마이클 잭슨' ] 
music = ['yesterday', 'imagine', '너랑나', '마슈멜로우', 'beat it', 'smoth ciriminal' ]


문제1. 아래의 감독 리스트와 영화 리스트를 가지고 moive_dict 를 생성하시오

# 감독 리스트
directors = ['크리스토퍼 놀란', '크리스토퍼 놀란', '스티븐 스필버그', '스티븐 스필버그', '마틴 스콜세지', '마틴 스콜세지']

# 영화 리스트
movies = ['인셉션', '다크 나이트', '쥬라기 공원', 'ET', '아이리시맨', '위대한 개츠비']

답:





