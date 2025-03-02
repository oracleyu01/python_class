▣ 예제78. for loop 문 + break 문

 for  루프문과 break 문은 반복문을 사용할 때 특정 조건을 만나면
 반복을 중단하고 루프문을 종료하는 기능을 제공합니다. 
 반복을 하다고 break 를 만나면 반복문을 완전히 종료 시켜버립니다.

예제1.  1부터 10까지의 숫자를 출력하는 for loop문에서 숫자 5를 만나면
          바로 for loop문을 종료시키시오

for  i  in  range(1, 11):
    if  i == 5 :
        print( '숫자 5를 만났습니다. 반복문을 완전히 종료합니다.')
        break
    print(i)

문제1. 다음의 상품들을 박스 처리하는데 불량품이 발견되면 그냥 모든 공정을
         중단시켜버리겠금 break문을 사용한 for loop 문을 작성하시오!

box  = [ '정상품', '정상품', '불량품', '정상품', '정상품', '정상품']

결과:

정상품 을 박스 처리 합니다.
정상품 을 박스 처리 합니다.
불량품이 발견되었습니다. 처리를 중단합니다.

답:

box  = [ '정상품', '정상품', '불량품', '정상품', '정상품', '정상품']

for  i   in  box:
    if  i =='불량품':
        print('불량품이 발견되었습니다. 처리를 중단합니다.')
        break
    print( i , '을 박스 처리합니다')

