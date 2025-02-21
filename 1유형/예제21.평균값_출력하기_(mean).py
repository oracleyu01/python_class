
▩ 예제21.평균값_출력하기_(mean).py

예제. 아래의 SQL을 판다스로 출력하시오 !

SQL> select  to_char(hiredate, 'RRRR') as 입사년도, avg(sal) as 평균월급
          from  emp
          group  by  to_char(hiredate,'RRRR'); 

답:
#1. hiredate 를 object 에서 datetime 으로 데이터 유형을 변환 


#2. hiredate 에서 연도만 추출하는 방법 


#3. 연도 컬럼 생성하기


#4. 입사한 년도, 입사한 년도별 평균월급을 출력하기 

 
문제1. 아래의 SQL을 판다스로 구현하시오 !

SQL>  select   job as 직업,  avg(sal)  as 평균월급
           from   emp 
           group  by   job
           order  by   평균월급  desc;

답:



문제2.  보증금 데이터를 히스토그램 그래프로 그려서 분포를 확인하시오 

답:

문제3. 보증금을 5개의 그룹으로 나누는 컬럼을 생성하시오 !

데이터의 분포를 고려해서 균등하게 5개로 나눕니다.

답:


문제4. 보증금_등급과 전용면적별 평균값을 출력하시오 ! 

SQL> select  보증금_등급, avg(전용면적)
           from   train
           group  by 보증금_등급 ;

답:


