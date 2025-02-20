▩ 예제5.  데이터 정렬하기  

문법:  df[ ['컬럼명1', '컬럼명2']].sort_values(by='정렬할 컬럼명', ascending=True )

예제1.  아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, sal
          from  emp 
          order   by   sal   desc ;

Pandas>  emp[ ['ename', 'sal' ] ].sort_values(by='sal', ascending=False) 

또는 

Pandas> emp.loc[  :  , ['ename', 'sal' ] ].sort_values(by='sal', ascending=False)

 콜론(:) 은 모든 행을 다보겠다는 의미입니다. 

문제1.  직업이 SALESMAN 이 아닌 사원들의 이름과 월급을 출력하는데
          월급이 높은 사원부터 출력하시오 !

SQL> select  ename, sal
         from  emp
         where  job  != 'SALESMAN'
         order  by  sal  desc ;

Pandas> emp.loc[ emp.job !='SALESMAN', ['ename', 'sal' ] ].sort_values(by='sal', ascending=False)


문제2. 데이콘 데이터인 부동산 허위매물 데이터의 훈련 데이터를 불러와서 판다스 
        데이터 프레임으로 생성하시오 !

데이터: https://cafe.daum.net/oracleoracle/Sq8G/2

답:

import pandas as  pd

train=pd.read_csv("D:\\data\\open_budongsan\\train.csv")
train.head()

문제3. train 데이터프레임에서 허위매물인 데이터의 허위매물여부와 보증금을 검색
        하는데 보증금이 높은것부터 출력하시오 !

import pandas as  pd

train=pd.read_csv("D:\\data\\open_budongsan\\train.csv")

df = train.loc[ train.허위매물여부==1, ['허위매물여부', '보증금'] ]

df.sort_values( by='보증금', ascending=False)

