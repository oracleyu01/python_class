▣ 예제85. 파이썬 함수

함수로 만들지 않고 숫자를 물어보게해서 짝수 인지 홀수 인지가 출력되게 코드를 작성하시오 !

숫자를 입력하세요 ~   2

 짝수입니다. 

숫자를 입력하세요 ~  7

 홀수 입니다.

답:



경험이 많은 파이썬 사용자들은 자기가 자주 쓰는 함수들을 모듈로 만들어서
저장하고 불러 옵니다.  시간 절약하려고 모듈로 만듭니다.

※ 모듈 생성 방법  순서

1. 메모장을 엽니다.
2. 다음의 스크립트를 저장합니다.
3. yys.py 로 저장합니다.
4. yys 모듈을 불러오고 사용합니다.

함수 생성 문제1.  숫자를 입력하면 해당 숫자가  양수인지, 음수인지, 0인지를 
                 출력하는 함수를 생성하시오 !

check_number(7)

양수 입니다.

check_number(-2)

음수 입니다.

check_number(0)

 0 입니다.

함수 생성 문제2.  위의 check_number 함수를 yys.py 에 두번째 코드로 넣으시오!

yys.py 에 반영했으면 커널을 restart 해주셔야 반영됩니다. 



함수 생성 문제3.  import  yys 를 실행했을때 yys.py 에 있는 함수 목록이 출력되게
                 하시오 !

yys.py  스크립트 맨아래에 아래의 코드를 추가합니다. 

#1. 짝수 홀수 판정 함수

def check_even_odd(number):
    if number % 2 == 0:
        return  "짝수 입니다."
    else:
        return  "홀수 입니다."

#2. 양수인지 음수인지 판정하는 함수






