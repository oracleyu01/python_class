
▣ 예제87. 한줄짜리 이름없는 함수 만들기(lambda)

■ 람다 함수(`lambda`)의 필요성  

람다 함수는이름 없이 한 줄로 정의할 수 있는 익명 함수로, 다음과 같은 상황에서 유용하게 사용됨.

1. 간단한 연산을 수행할 때 코드 가독성을 높임

- 굳이 함수로 만들기에는 너무 짧고 단순한 연산을 빠르게 처리할 수 있음.


2. `map()`, `filter()`, `sorted()` 등과 함께 사용하여 코드 간결화
- 리스트의 각 요소를 변환할 때, 불필요한 함수 정의 없이 사용 가능.


- 특정 조건을 만족하는 요소만 필터링할 때 활용.



3. 정렬 기준을 지정할 때 유용함
- `sorted()`에서 정렬 기준을 직접 지정할 수 있음.


4. 일회성 함수로 사용하기 적합
- 굳이 별도로 정의할 필요 없이 즉석에서 정의하여 사용 가능.


5. 함수형 프로그래밍 스타일에서 필수적
- `map`, `filter`, `reduce` 같은 함수형 프로그래밍 패턴에서 빠르고 깔끔한 코드 작성 가능.


■ 결론:

람다 함수는짧고 간단한 연산을 빠르게 처리할 때,일회성 함수가 필요할 때, 
기존 함수(`map`, `filter`, `sorted` 등)와 함께 사용할 때 유용함. 하지만 너무 복잡한 로직을 람다로 만들면 가독성이 떨어질 수 있으므로, 
적절한 상황에서만 사용하는 것이 중요함.


■ 데이터 분석가들에게 람다 함수(lambda)가 유용한 경우  

데이터 분석에서는 데이터를 변환하거나 필터링하는 작업이 많고, 
이를 간결하게 처리하는 데 람다 함수가 매우 유용함. 특히 Pandas, NumPy, 
데이터 정렬 및 필터링, 함수형 프로그래밍 활용 등에서 강력한 도구로 사용됨.


■ 1. Pandas에서 데이터 변환 (apply()와 함께 사용)

데이터프레임의 특정 컬럼 값을 변환할 때 유용함.

import pandas as pd

df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})





■ 2. 여러 조건을 적용할 때 (apply()와 lambda)

조건에 따라 새로운 컬럼을 만들 때 유용함.





■ 3. map()을 활용한 데이터 변환
- map() 함수는 Series 객체에 적용 가능하며, 값 매핑 시 활용 가능.





 ■ 4. filter()와 함께 특정 조건의 데이터만 추출
- 특정 조건을 만족하는 데이터를 필터링할 때 사용.


data = [10, 15, 20, 25, 30]






 ■ 5. 정렬 기준 (sorted() 활용)
- 데이터 리스트를 특정 기준으로 정렬할 때 유용함.


data = [('Alice', 25), ('Bob', 30), ('Charlie', 20)]



■ 6. groupby()와 함께 집계 연산 수행
Pandas의 groupby()와 함께 활용하면 데이터 그룹별 변환을 쉽게 할 수 있음.


df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A'],
    'value': [10, 20, 30, 40, 50]
})





■ 7. reduce()를 활용한 누적 계산
- functools.reduce()를 활용하여 데이터를 누적 계산할 때 사용 가능.




 ■ 람다 함수가 유용한 경우 정리:

✅ Pandas의 apply()로 데이터 변환  
✅ 조건부 값 변경 (if-else 활용)  
✅ 데이터 필터링 (filter())  
✅ 데이터 정렬 (sorted())  
✅ groupby()와 함께 데이터 집계 처리  
✅ 누적 계산 (reduce())  

⚠️ 주의할 점  

람다 함수는 간단한 경우에는 유용하지만, 
코드가 길어지면 일반 def 함수로 변환하는 것이 가독성에 좋음!  
특히, 여러 개의 조건문이 포함될 경우 일반 함수가 더 적절할 수 있음.
