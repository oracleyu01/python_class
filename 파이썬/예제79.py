▣ 예제79. 이중 for  loop문

 이중 for 루푸문은 하나의 for 루프문 안에 또 다른 for 루프문을 중첩하여
 사용는것입니다. 

예제1. 구구단 2단을 출력하시오 !

for   i   in  range( 1, 10 ):
    print( ' 2  x  ' , i , '=',  2*i )  

예제2. 구구단 2단 밑에 3단도 출력하시오!

for   i   in  range( 1, 10 ):
    print( ' 2  x  ' , i , '=',  2*i )  

for   i   in  range( 1, 10 ):
    print( ' 3  x  ' , i , '=',  3*i )  

구구단 전체를 출력해야한다면 위의 for loop문을 9번을 써야합니다. 
그렇게 하지 말고 이중 for loop문으로 하면 짧고 간결하게 출력할 수 있습니다.

예제3.  구구단 전체를 출력하시오 !

for   dan   in   range(2, 10):
    for   num  in   range(1, 10):
        print( dan, '  x  ',  num, '=',  dan*num )

설명:  dan 이 2일때 num 을 1 ~ 9 까지 반복
        dan  이 3일때 num 을 1 ~9 까지 반복 
                            :
       dan  이 9일때 num 을 1 ~9 까지 반복 

문제1. 서포트 벡터 머신에서 하이퍼 파라미터 2개인 C 와 gamma 를
         서로 다르게해서 최적의 하이퍼 파라미터를 찾으려 합니다.
         C 는 1부터 5까지 하고 gamma 는 0.01 ~ 0.05 까지해서
        모든 조합이 출력될 수 있도록 이중 루프문을 작성하시오!

결과 예시:
C: 1, gamma: 0.01
C: 1, gamma: 0.02
C: 1, gamma: 0.03
C: 1, gamma: 0.04
C: 1, gamma: 0.05
      :
C: 5, gamma: 0.05

답:
C_values = range(1,6)
gamma_values=[0.01, 0.02, 0.03, 0.04, 0.05 ]

for  C  in  C_values:
    for gamma  in  gamma_values:
        print ( f"C: {C}, gamma: {gamma}")

설명:  f-string 을 사용하면 문자열 포멧팅을 할 수 있습니다. 
        중괄호 {}  안에 변수를 넣어서 해당 변수의 값을 문자열에 입력할 수 있습니다.
