
▩ 예제42.범주형_데이터를_숫자로_변환하는_2가지_방법.py

이론설명:  https://cafe.daum.net/oracleoracle/Sq8G/31

# 예제. label encode 사용 예시 

# 1. 데이터 생성 
import pandas  as  pd

df = pd.DataFrame( {'색상' : ['빨강', '파랑', '노랑'] } )
df

# 2. 범주형 데이터를 숫자형으로 변환(라벨 인코더 사용)
from sklearn.preprocessing  import  LabelEncoder



예제. get_dummies 를 이용해서 색상을 숫자로 변환하시오 !

# 1. 데이터 생성 
import pandas  as  pd

df = pd.DataFrame( {'색상' : ['빨강', '파랑', '노랑'] } )
df

# 2. 범주형 데이터를 숫자형으로 변환(get_dummies 사용)



문제1.  emp 데이터 프레임에서 직업을 get_dummies 를 이용해서 숫자로 변환하시오 !

답:



문제2.  emp 데이터 프레임의 job 에 서열이 있다고 가정하고 label encoder 로 
        숫자로 변환하시오 !

답:

# 1. 인코딩에 필요한 모듈을 불러옵니다. 
from  sklearn.preprocessing  import  LabelEncoder 

#2. LabelEncoder 설계도로 제품(객체)를 만듭니다.


#3. label_encoder 를 이용해서 job 을 숫자로 변환합니다. 



#4. job 을 drop 시킨 나머지 컬럼으로  emp_encode 라는 데이터 프레임을 생성합니다. 


