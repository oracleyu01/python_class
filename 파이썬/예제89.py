
▩ 예제89. 파이썬 모듈 이해하기

  import   모듈명   <----  나의 코딩생활을 편하게 해줄 스크립트들을 모아넣습니다.

  모듈명.함수

  파이썬에서는 각각의 소스 파일을 일컬어 모듈이라고 합니다.
  이미 만들어져 있고 안정성이 검증된 함수들을 성격에 맞게 하나의 파일로 묶어 놓은것을
  모듈이라고 합니다.
  외부의 모듈에 있는 함수를 사용하려면 이 모듈을 먼저 우리 코드로 가져와서 자유롭게
  사용할 수 있도록 해야하는데 이런일을 파이썬에서는 모듈을 import 한다라고 합니다.

import  pandas  as  pd  # 판다스라는 모듈을 임폴트해서 우리 코드에서 자유롭게 쓰겠다.

emp = pd.read_csv("c:\\data\\emp.csv")

위의 코드는 외부의 모듈을 호출하는 코드입니다. 

우리가 직접 모듈을 만들고 싶다면 아래와 같이 하면 됩니다.

def  add_number(n1, n2):
    result = n1 + n2 
    return  result

def  minus_number(n1, n2):
    result = n1 - n2
    return  result

def  gob_number( n1, n2 ):
    result = n1 * n2
    return  result

위의 3개의 함수를 모듈화 시키기 위해서 메모장을 새로열고 위의 스크립트를 붙여넣고
yu_auto.py  라는 이름으로 저장하세요 ~

!dir  를 수행하면 주피터 노트북의 홈 디렉토리 위치를 확인할 수 있습니다.

C:\Users\YYS 디렉터리

