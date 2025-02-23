
▩ 예제43.날짜형_데이터를_숫자로_변환하기.py


#예제. 사원 테이블에서 hiredate 를 숫자로 변환하기 

#1. object 를 datetime 으로 변환합니다.
#emp_encode.info()
emp_encode['hiredate'] = pd.to_datetime( emp['hiredate'])
#emp_encode.info()

#2. 년도, 월, 일로 분리합니다.
emp_encode['hiredate_year'] = emp_encode['hiredate'].dt.year
emp_encode['hiredate_month'] = emp_encode['hiredate'].dt.month
emp_encode['hiredate_day'] = emp_encode['hiredate'].dt.day
emp_encode

#3. hiredate 컬럼을 drop 한 나머지 컬럼으로 데이터 프레임을 생성합니다. 
emp_final = emp_encode.drop('hiredate',axis=1)
emp_final

문제1. 1981년도에 입사한 사원들의 이름과 입사일을 출력하시오 

