▩ 예제6. 판다스에서 날짜 데이터 다루기 

 기본적으로 엑셀이나 csv 파일을 판다스로 구성하면 날짜 데이터의 경우는
 날짜형이 아니라 문자형으로 구성됩니다. 
 그래서 반드시 날짜 데이터는 날짜형으로 변환하는 작업을 해주어야합니다. 

예제.  emp 데이터 프레임의 구조를 확인하시오 !

emp.info()

#   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   empno     14 non-null     int64  
 1   ename     14 non-null     object 
 2   job       14 non-null     object 
 3   mgr       13 non-null     float64
 4   hiredate  14 non-null     object  <----  문자형 
 5   sal       14 non-null     int64  
 6   comm      4 non-null      float64
 7   deptno    14 non-null     int64  

예제. emp 데이터 프레임의 hiredate 를 날짜형으로 변환하시오 !

import  pandas  as  pd 

emp['hiredate'] = pd.to_datetime(emp['hiredate'])

emp.info()

#   Column    Non-Null Count  Dtype         
---  ------    --------------  -----         
 0   empno     14 non-null     int64         
 1   ename     14 non-null     object        
 2   job       14 non-null     object        
 3   mgr       13 non-null     float64       
 4   hiredate  14 non-null     datetime64[ns] <--- 날짜형으로 변환되었습니다.
 5   sal       14 non-null     int64         
 6   comm      4 non-null      float64       
 7   deptno    14 non-null     int64    

문제1.  아래의 SQL을 판다스로 구현하시오 !

SQL>  select  ename, hiredate
           from  emp
           where  hiredate='81-11-17';

pandas> 

문제2. 아래의 SQL을 판다스로 구현하시오 !

SQL> select  ename, hiredate
          from  emp
          where  hiredate  between  '81-01-01'  and  '81-12-31'
          order  by  hiredate  desc; 

답:



문제3.  train.csv 에서 매물확인방식이 현장확인인것 필터링해서 매물확인방식과 
        게재일을 출력하는데 게재일이 높은것부터 출력하시오 !

답:













