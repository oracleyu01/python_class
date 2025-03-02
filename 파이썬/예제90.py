
▣ 예제89.  파이썬 패키지 이해하기 

  우리가 음악파일을 저장할 때도 장르별로 폴더를 만들어서 별도로
  저장을 하듯이 파이썬 모듈도 음악처럼 갯수가 많아지면 폴더(모듈 꾸러미)
  별로 별도로 관리를 해야 관리자 편해지는데 이 폴더가 바로 '패키지' 이다.

  패키지 (폴더)         vs  모듈 (폴더 안의 my_cal.py 같은 파이썬 스크립트)

■ 파이썬 패키지를 만드는 단계

1. 아래의 디렉토리에 my_loc 라는 폴더를 생성한다. 

c:\\Users\\stu  

c:\\Users\\stu\my_loc

2. my_loc 폴더 안에  my_cal.py 를  옮겨 놓는다. 
   (my_cal.py 를 my_loc 폴더에 복사하고 기존에 있던 my_cal.py는 지우시오)


3.  이 평범한 폴더가 패키지로 인정을 받으려면 반드시 갖고 있어야하는
     파일이 있습니다.  그 파일이 __init__.py 라는 파일입니다. 

c:\\Users\\stu\my_loc
                                |
                                |
                                 1. __init__.py
                                 2. my_cal.py 

4.  새로운 창에서 아래와 같이 스크립트를 수행합니다. 

from  my_loc  import   my_cal   # from 패키지 import  모듈

print ( my_cal.add_number(1,2) )   # my_cal 모듈 안에 있는 add_number 함수
                                            # 를 실행해라 ~

어제 위와 비슷한 스크립트를 보았는데

from  scipy.stats  import   norm # scipy 패키지안에 stats 라는 패키지에
                                          # norm 이라는 모듈을 임폴트해라 ~

print ( norm.pdf(x, 평균, 표준편차) ) # norm 모듈에 pdf (확률밀도함수)를
                                              # 실행해라 ~

■ 45. 파이썬 모듈 임포트 이해하기 ① (import)

  이미 만들어져 있는 어떤 함수를 우리가 작성하는 코드에서 
  자유롭게 활용할 수 있으려면 해당 함수가 포함된 모듈을 임폴트해야합니다.
  임폴트하는 방법은 다음과 같습니다.

  import  모듈이름
  from 패키지 import  모듈이름
  import  패키지이름.모듈이름

예제: 아래와 같이 우리가 만든 모듈이 아니라 다른 사람이 만든 모듈을
        임폴트를 해서 썼는데 이 모듈은 어디에 있는것일까 ?

import   pandas
import   numpy 

위와 같이 패키지 이름을 안주고 모듈만 import 했는데 잘 실행이 되었습니다.
위와같은 모듈은 어떤 모듈입니까 ?

1. 파이썬 내장 모듈
2. sys.path 에 정의되어 있는 모듈

* import 를 만나면 파이썬 모듈을 찾는 순서
1. 파이썬 내장 모듈에 있는지 확인
2. sys.path 에 정의되어 있는 디렉토리를 뒤져봅니다.

* 파이썬 내장 모듈이 무엇이 있는지 확인하는 방법
import  sys
print ( sys.builtin_module_names)

('_abc', '_ast', '_bisect', '_blake2', '_codecs', '_codecs_cn', '_codecs_hk', 
'_codecs_iso2022', '_codecs_jp', '_codecs_kr', '_codecs_tw', 
'_collections', '_contextvars', '_csv', '_datetime', '_functools',
 '_heapq', '_imp', '_io', '_json', '_locale', '_lsprof', '_md5',
 '_multibytecodec', '_opcode', '_operator', '_pickle', 
'_random', '_sha1', '_sha256', '_sha3', '_sha512', 
'_signal', '_sre', '_stat', '_string', '_struct', 
'_symtable', '_thread', '_tracemalloc', 
'_warnings', '_weakref', '_winapi', 'array', 
'atexit', 'audioop', 'binascii', 'builtins', '
cmath', 'errno', 'faulthandler', 'gc',
 'itertools', 'marshal', 'math', 'mmap', 
'msvcrt', 'nt', 'parser', 'sys', 'time', 'winreg', 'xxsubtype', 'zipimport', 'zlib')

* sys.path 에 정의된 디렉토리가 무엇인지 확인하는 방법

import  sys
for  i  in   sys.path:
    print ( i ) 

C:\Users\stu
C:\Users\stu\anaconda3\python37.zip
C:\Users\stu\anaconda3\DLLs
C:\Users\stu\anaconda3\lib
C:\Users\stu\anaconda3

C:\Users\stu\AppData\Roaming\Python\Python37\site-packages
C:\Users\stu\anaconda3\lib\site-packages
C:\Users\stu\anaconda3\lib\site-packages\win32
C:\Users\stu\anaconda3\lib\site-packages\win32\lib
C:\Users\stu\anaconda3\lib\site-packages\Pythonwin
C:\Users\stu\anaconda3\lib\site-packages\IPython\extensions
C:\Users\stu\.ipython

* site-packages 란 무엇인가 ?

 site-packages 란 파이썬의 기본 라이브러리 패키지 외에 추가적인 패키지를
 설치하는 디렉토리 입니다. 
 site-packages 디렉토리에 여러가지 소프트웨어가 사용할 공통 모듈을
 넣어두면 물리적인 장소에 구애받지 않고 모듈에 접근하여 반입할 수 있습니다

아래의 명령어가 수행되려면 아래의 명령어를 수행하는 스크립트가 
c:\\Users\\stu 밑에 있어야합니다. 왜냐하면 my_loc 폴더가
바로 c:\\Users\\stu 밑에 있기 때문입니다.  

from  my_loc import  my_cal

그런데 c:\\Users\\stu 가 아니더라도 다른 디렉토리에서라도 
from  my_loc import  my_cal 명령어를 자유롭게 실행하려면 
my_loc 폴더가 C:\Users\stu\anaconda3\lib\site-packages 
밑에 있으면 됩니다. 

문제151. my_loc 폴더를 site-packages 폴더 밑에 두세요 !!

C:\Users\stu\anaconda3\lib\site-packages 

from my_loc3 import my_cal 

print ( my_cal.add_number(1,2) ) 
print ( my_cal.gob_number(1,2) )
print ( my_cal.devide( 10, 2 )  )    #  5.0 

