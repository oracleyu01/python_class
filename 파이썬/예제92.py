
▩ 예제92. 클래스 이해하기(class)

import  auto as a

a.my_data()

우리가 앞에서 위와 같이 my_data 라는 함수를 만들어서 우리가 해야할 작업을
쉽고 편하게 할 수 있게 했듯이 클래스도 프로그램 코딩을 편하게 하려고 만들어놓은
파이썬의 기능입니다. 

클래스의 가장 큰 장점은 ?    상속 입니다.

코드를 상속 받게 되면 상속받은 코드를 내가 직접 작성하지 않아도 됩니다. 
그냥 가져다 쓰면 됩니다. 

예제. 내가 영화 할인 카드 프로그램을 개발해야하는데 그럴러면 카드의 기본기능을
       수행해야하는 코드를 다 작성해야합니다.  그런데 카드의 기본 기능은 너무 중요한
        코드라서 팀장님이 그 코드를 만들었으면 나는 그냥 팀장님이 만든 카드 기본기능
       코드를 상속받기만하면 되고 내가 구현야할 영화 할인에만 집중해서 코드를 작성하
       면 됩니다.

  팀장님                                     팀원들

 카드 기본 기능                          A 팀원은 영화 할인에만 집중
                                              B 팀원은 주유 할인에만 집중
                                              C 팀원은 스타벅스 할인에만 집중

팀장님이 만든 카드 기본 기능 코드를 팀원들이 사용하려면 클래스로 만들어서
상속받으면 됩니다. 

클래스란 ?    설계도와 같은 것입니다.

 예: 총 설계도     ------------------>  총(제품)
     (클래스)                                  (객체)

문제.  다음과 같이 숫자를 입력하고 실행을 하면 다음의 결과가 출력되는 함수를
            생성하시오 !

shoot(3)

탕!
탕!
탕!

def  shoot(num):
    for  i  in  range(1, num+1):
        print('탕!')

shoot(3)

문제. 총을 장전하는 기능인 charge 라는 함수를 다음과 같이 생성하시오 !

charge(5)

5발 장전 되었습니다.

def  charge(num):
    print( num, '발이 장전 되었습니다')

설명:  shoot 과 charge 함수는 총의 기능에 해당하는 함수 입니다. 
        그런데 지금 별개로 따로 따로 만들어서 별개로 작동하고 있습니다.
        그런데 만약 총알을 장전한 만큼 총을 shoot 하게 하려면 이 함수들을
        하나의 클래스에 넣어야합니다.

* 총 클래스(설계도) 생성

class  Gun():                                                  # 총 설계도 만듭니다.
    def  charge( self,  num ):                              # 총알을 충전하는 함수입니다.
        self.bullet = num                                    # 총알을 num 숫자만큼 장전합니다.
        print( self.bullet, '발이 장전 되었습니다')

    def  shoot( self, num ):                                # 총을 쏘는 함수입니다.
        for  i  in  range( 1, num+1):                      # 입력된 숫자만큼 반복합니다.
            if  self.bullet > 0:                                # 총알이 있다면
                print( '탕!' )                                   # 한발 쏩니다.
                self.bullet = self.bullet - 1                # 그리고 한발 차감합니다.
            elif  self.bullet == 0 :                          # 총알이 없다면
                print('총알이 없습니다.')                   # 없다는 메세지를 출력하고 
                break                                           # loop 문을 종료 시킵니다.


왜 클래스 이름의 첫글자를 대문자로 했냐면 코드의 가독성을 높이기 위해 쓴 표기법인데
이것을 낙타 표기법이라고 합니다.  낙타등처럼 생겼다고 해서 낙타 표기법이라고 합니다.

예: GunCharge

예제. 총 설계도(클래스) 를 가지고 총 제품을 만드시오 !

gun1 = Gun()

예제. 총을 10발 장전 합니다.

gun1.charge(10)

예제. 총을 한발 쏩니다. 

gun1.shoot(1)

문제. 총 설계도로 gun2 라는 이름의 총을 만드시오 !

gun2 = Gun()

문제. gun2 에는 총알을 5발 장전하세요 !

gun2.charge(5)

문제. gun2 의 총알 5발을 전부 다 사용해서 총을 쏘세요 ~

gun2.shoot(5)

※ 중요 !!  gun1 과 gun2 는 같은 클래스로 만들었지만 서로 다른 총입니다.

gun1.shoot(2)

탕!
탕

gun1 은 아직 총알이 남아있어서 총알을 발사 할 수 있습니다. 

※ 클래스 생성할 때 왜 self 를 사용했는가 ?

charge( self, num1):  <--- 이렇게 클래스(설계도)를 생성했습니다. 

gun1.charge(20)

gun1 이라는 것을 확실하게 해주기 위해서 self 를 쓴것입니다. 

다른 총에 충전하는게 아니라  self(자기자신) 의 총에 충전해야하기 때문에 self 를 
함수 생성할 때 입력 매개변수로 둔것 입니다. 

문제. 위의 총 설계도를 수정해서 앞으로 생산하는 총의 shoot 함수를 쓸때는
      다음과 같이 총을 쏘고 몇발 남았는지가 출력되게하시오 !

gun3 = Gun()

gun3.charge(10)
gun3.shoot(2)

탕!
탕!
8발 남았습니다. 

답:

class  Gun():                                                   # 총 설계도 만듭니다.
    def  charge( self,  num ):                              # 총알을 충전하는 함수입니다.
        self.bullet = num                                    # 총알을 num 숫자만큼 장전합니다.
        print( self.bullet, '발이 장전 되었습니다')

    def  shoot( self, num ):                                # 총을 쏘는 함수입니다.
        for  i  in  range( 1, num+1):                      # 입력된 숫자만큼 반복합니다.
            if  self.bullet > 0:                                # 총알이 있다면
                print( '탕!' )                                   # 한발 쏩니다.
                self.bullet = self.bullet - 1                # 그리고 한발 차감합니다.
            elif  self.bullet == 0 :                          # 총알이 없다면
                print('총알이 없습니다.')                   # 없다는 메세지를 출력하고 
                break                                           # loop 문을 종료 시킵니다.
        print(  self.bullet, '발 남았습니다.') 

※ 클래스 생성시 self 를 사용해야하는 경우

 1. 변수 이름 앞에 
 2. 함수 생성시 괄호안에 첫번째 입력 매개변수로 

왜 ! self 를 써야하는가 ?   설계도로 여러개의 제품을 만들 수가 있는데
                                   특정 제품에서 사용하는 변수이고 함수이다라는것을
                                   명확하게 하기 위해서 입니다.

문제.  위에 Gun() 클래스를 가지고 Card 라는 클래스를 생성하시오 !

 변수 :  bullet ----> money
 함수1 :  charge ---> charge
 함수2 :  shoot ----> consume 


class  Card():                                                   # 카드 설계도 만듭니다.
    def  charge( self,  num ):                              # 돈을 충전하는 함수입니다.
        self.money = num                                    # 돈을 num 숫자만큼 장전합니다.
        print( self.money, '원이 충전 되었습니다')

    def  consume( self, num ):                          # card 를 쓰는 함수입니다.
        if  self.money > 0:                                # card 에 돈이 있다면
            print( num, '원이 사용되었습니다.' )     # 돈을 씁니다.
            self.money =  self.money - num         # 그리고 쓴 돈만큼 차감합니다.
        elif  self.money == 0 :                          # 돈이 없다면
            print('잔액이 없습니다.')                   # 없다는 메세지를 출력합니다.
                                                       
        print(  self.money, '원 남았습니다.') 

문제.  위의 카드 설계도로 카드를 발급하세요 !  (카드 이름은 card1 입니다.)
      그리고 10000 원 충전하고 1000원 사용해보세요 !

card1 = Card()

card1.charge(10000)

card1.consume(1000)

문제.  card1 = Card()  이렇게 카드를 발급했을때 "카드가 발급 되었습니다." 라는
       메세지가 출력되게 설계도를 고치시오 !

class  Card():                      # 카드 설계도 만듭니다.
    def  __init__(self):            # 설계도를 가지고  객체를 생성할때 바로 작동되는 함수
        self.money = 0          #  money 라는 변수에 값을 0 을 넣었습니다. 
        print('카드가 발급 되었습니다. ', self.money, '원이 충전되어 있습니다.' ) 

    def  charge( self,  num ):                              # 돈을 충전하는 함수입니다.
        self.money = num                                    # 돈을 num 숫자만큼 장전합니다.
        print( self.money, '원이 충전 되었습니다')

    def  consume( self, num ):                          # card 를 쓰는 함수입니다.
        if  self.money > 0:                                # card 에 돈이 있다면
            print( num, '원이 사용되었습니다.' )     # 돈을 씁니다.
            self.money =  self.money - num         # 그리고 쓴 돈만큼 차감합니다.
        elif  self.money == 0 :                          # 돈이 없다면
            print('잔액이 없습니다.')                   # 없다는 메세지를 출력합니다.
                                                       
        print(  self.money, '원 남았습니다.') 

