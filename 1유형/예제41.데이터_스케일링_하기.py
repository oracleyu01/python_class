
▩ 예제41.데이터_스케일링_하기.py

* 데이터 스켈링 방법 2가지 ? 

 데이터 스케일(scale) :    서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
                           아무리 좋은 데이터를 가지고 있어도 그 데이터로 기계학습을 하거나
                           통계분석을 하려면 데이터 스케일링(scaling) 을 해야합니다.

 1. 최대 최소 정규화 : 데이터를 0~1사이의 값으로 변환하는것 

 예:  from  sklearn.preprocessing  import  MinMaxScaler 

 2. 표준화 :  데이터를 0 을 중심으로 양쪽으로 분포 시키는것  

 예:   from  sklearn.preprocessing  import  StandardScaler 
        
예제.  iris 데이터를  표준화 수행 작업

#예제.  iris 데이터를  표준화 수행 작업

#1.  iris2.csv 를 불러옵니다. 
import pandas as pd

iris = pd.read_csv("d:\\data\\iris2.csv")
iris.head()

#2. 기술통계정보를 확인합니다. 


#3. 표준화를 수행합니다. 



# axis=1 이면 양옆으로 붙인다. axis=0 이면 위아래로 붙인다. 

#4. 표준화가 잘 되었는지 확인합니다. 


문제. iris 데이터를 정규화 하시오 !

# 답:

#1.  iris2.csv 를 불러옵니다. 
import pandas as pd

iris = pd.read_csv("d:\\data\\iris2.csv")
iris.head()

#2. 기술통계정보를 확인합니다. 


#3. 정규화를 수행합니다. 



# axis=1 이면 양옆으로 붙인다. axis=0 이면 위아래로 붙인다. 


#4. 정규화가 잘 되었는지 확인합니다. 

