
▩ 예제11.데이터를_찾는_find_함수_배우기.py

문법:  SMITH 라는 단어에서 M 이라는 철자는 몇번째 있는가 ?

a = 'smith' 
a.find('m')

a.find('k')   # 없는 철자를 찾으려고 시도하면 -1 이 나옵니다. 

문제1.  emp.csv 에서 ename 에서 T 가 몇번째 철자인지 출력하시오 ! 

emp = pd.read_csv("c:\\data\\emp.csv")
emp


