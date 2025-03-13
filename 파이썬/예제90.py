
▣ 예제89.  파이썬 패키지 이해하기 

  우리가 음악파일을 저장할 때도 장르별로 폴더를 만들어서 별도로
  저장을 하듯이 파이썬 모듈도 음악처럼 갯수가 많아지면 폴더(모듈 꾸러미)
  별로 별도로 관리를 해야 관리자 편해지는데 이 폴더가 바로 '패키지' 이다.


■ 파이썬 패키지를 만드는 단계

1. 아래의 디렉토리에 my_loc 라는 폴더를 생성한다. 

c:\\Users\\ITWILL  

c:\\Users\\ITWILL\my_loc

2. my_loc 폴더 안에  yu_auto.py 를  옮겨 놓는다. 

yu_auto.py 의 내용:

def  add_number(n1, n2):
    result = n1 + n2 
    return  result

def  minus_number(n1, n2):
    result = n1 - n2
    return  result

def  gob_number( n1, n2 ):
    result = n1 * n2
    return  result


3.  이 평범한 폴더가 패키지로 인정을 받으려면 반드시 갖고 있어야하는
     파일이 있습니다.  그 파일이 __init__.py 라는 파일입니다. 

c:\\Users\\ITWILL\my_loc
                                |
                                |
                                 1. __init__.py
                                 2. yu_auto.py 

4.  새로운 창에서 아래와 같이 스크립트를 수행합니다. 

from  my_loc  import  yu_auto 

print ( yu_auto.add_number(1,2) )  
                                          




