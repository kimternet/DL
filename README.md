
## 1. 신경망 기초수학( 함수, 극한, 미분 등 파이썬으로 구현)
1. 1차함수
2. 2차함수
3. 3차함수(다항함수)
   - 함수의 최소값/ 최대값
   - 특정 구간 내에서 최소값 구하기
4. 지수함수/로그함수
5. 지수함수
   - 밑이 e인 지수함수 표현
   - 로그함수
   - 역함수 관계
   - 함수 조작
6. 극한
   - 삼각함수의 극한
   - 지수함수, 로그함수의 극한
   - 자연로그(e)의 밑
7. 미분
   - 미분과 기울기
   - 함수 위의 점(a,b)에서의 접선의 방정식
   - 미분 공식
   - 편미분
   - 기울기(gradient)
   - 기울기의 의미를 그래프로 확인
## 2. 신경망 데이터 표현
1. 텐서(Tensor)
2. 스칼라(0차원 텐서)
3. 벡터(1차원 텐서)
   -벡터의 합
   -벡터의 곱
4. 스칼라와 벡터의 곱
5. 2차원 텐서(행렬)
   - 행렬 원소곱
   - 행렬 점(dot)곱(내적, product)
   - 역행렬
   - 전치행렬
6. 3차원 텐서
   - 3차원 텐서 활용 예시(이미지)
7. 브로드캐스팅(broadcasting)
   - 4, 5차원 텐서(동영상)
## 3. 신경망 구조
1. 신경망 구조
     - 퍼셉트론
     - 뉴런의 수학적 표현
     - 완전 연결 계층(Fully-Connected Layer) 수학적 표현
     - 논리회로(AND, OR, NAND, XOR 게이트)
     - 다층 퍼셉트론(Multi Layer Perceptron, MLP)
       -다층 퍼셉트론의 구성
       -XOR 게이트
     - 활성화 함수(Activation Function)
       - Step Function(계단함수)
       - sigmoid Function(시그모이드)
       - 시그모이드 함수와 계단 함수 비교
       - ReLU(Rectified Linear Unit)
       - 하이퍼볼릭탄젠트 함수(Hyperbolic tangent function, tanh)
       - Identity Function(항등 함수)
       - Softmax
         - 소프트맥스 함수 주의점
       - max(a)를 빼는 이유
         - 활성화 함수를 비선형 함수(non-linear function)로 사용하는 이유
         - 그 외의 활성화 함수
         - 활성화 함수(Activate Function)참고
    - 3층 신경망 구성하기
      - 활성화 함수 정의
      - 레이어 정의
      - 신경망 추론 실행
## 4.모델 학습과 손실함수
1. 모델 학습과 손실함수
   - 모델의 학습
     - 지도 학습 vs 비지도 학습
   - 학습 매개변수(Hyper Parameter)
   - 손실함수(Loss Function, Cost Function)
     - 학습의 수학적 의미
     - 원-핫 인코딩(one-hot encoding)
     - 평균절대오차(Mean Absolute Error, MAE)
     - 평균제곱오차(Mean Squared Error, MSE)
     - 손실함수로서의 MAE와 MSE 비교
     - 교차 엔트로피 오차(Cross Entropy Error, CEE)
        - 이진 분류에서의 교차 크로스 엔트로피(Binary Cross Entropy, BCE)
## 5. 경사 하강법
1. 볼록함수(Convex Function)
2. 비볼록함수(Non-Convex-Function)
3. 경사하강법
   - 미분과 기울기
   - 경사하강법의 과정
   - 경사하강법 구현
   - 경사하강법 시각화
   - 비볼록 함수(Non-Convex Function)에서의 경사하강법
   - 비볼록함수 경사하강법 시각화
4. 전역 최적값 vs 지역 최적값
   - 전역 최솟값 vs 지역 최솟값 시각화
   - 경사하강법 구현(2)
5. 학습률(learning rate)
   - 학습률별 경사하강법
6. 안장점(Saddle Point)

