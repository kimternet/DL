
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

## 6. 신경망 학습
1. 단순한 신경망 구현: Logic Gate
   - 필요한 모듈 import
   - 하이퍼파라미터(HyperParameter)
   - 유틸 함수들(Utill Functions)
   - 신경망
   - AND Gate
      - 모델 생성 및 학습
      - 테스트
   - OR Gate
      - 모델 생성 및 학습
      - 테스트
   - NAND Gate
      - 모델 생성 및 학습
      - 테스트
      - 2층 신경망으로 XOR 게이트 구현(1)
      - 테스트
      - 2층 신경망으로 XOR게이트 구현(2)
      - 하이퍼파라미터(HyperParameter)
      - 모델 생성 및 학습
      - 테스트
2. 다중 클래스 분류:MNIST Dataset
   - 배치 처리
     - 신경망 구현:MNIST
     - 필요한 모듈 임포트
     - 데이터 로드
     - 데이터 확인
     - 데이터 전처리(Data Preprocessing)
     - 하이퍼파라미터(HyperParameter)
     - 사용되는 함수들(Util Functions)
     - 2층 신경망으로 구현
     - 모델 생성 및 학습
   - 모델 결과
## 7. 오차역전파(Backpropagation)
1. 오차역전파 알고리즘
2. 오차역전파 학습의 특징
3. 신경망 학습에 있어서 미분가능의 중요성
4. 합성함수의 미분(연쇄법칙, chain rule)
   - 합성함수 미분(chain rule) 예제
5. 덧셈, 곱셈 계층의 역전파
6. 활성화 함수(Activation)에서의 역전파
   - 시그모이드(Sigmoid)함수
   - ReLU 함수
7. 행렬 연산에 대한 역전파
   - 순전파(forward)
   - 역전파(1)
   - 역전파(2)
   - 배치용 행렬 내적 계층
8. MNIST 분류 with 역전파
   - Modules Import
   - 데이터 로드
   - 데이터 전처리
   - Hyper Parameters
   - Util Functions
   - Util Classes
     - ReLU
     - Sigmoid
     - Layer
   - Softmax
   - 모델 생성 및 학습
9. 모델 성능 개선해보기
    - SGD추가
    - 정규화 및 모델 용량, 하이퍼파라미터 튜닝
    - Loss 시각화해서 보기
## 8. 딥러닝 학습 기술
1. 최적화 방법: 매개변수 갱신
   - 확률적 경사하강법(Stochastic Gradient Descent, SGD)
   - SGD의 단점
   - 모멘텀(Momentum)
   - AdaGrad(Adaptive Gradient)
   - RMSProp(Root Mean Square Propagation)
   - Adam(Adaptive moment estimation)
2. 최적화 방법 비교(예, Linear Regression)
3. AI 두 번째 위기(가중치 손실, Gradient Vanishing)
4. 가중치 초기화
   - 초기값 : 0(zeros)
   - 초기값: 균일분포(Uniform)
   - 초기값: 정규분포(nomalization)
   - 아주 작은 정규분포값으로 가중치 초기화
   - 초기값: Xavier(Glorot)
   - 초기값: Xavier(Glorot)-tanh
5. 비선형 함수에서의 가중치 초기화
   - 초기값: 0(zeros)
   - 초기값: 정규분포(Nomalization)
     - 표준편차: 0.01일 때
   - 초기값: Xavier(Glorot)
   - 초기값: He
6. 배치 정규화(Batch Normalization)
7. 과대적합(Overfitting)/과소적합(Underfitting)
   - 과대적합 (Overfitting, 오버피팅)
   - 과소적합 (Underfitting, 언더피팅)
8. 규제화(Regularization) - 가중치 감소
   - L2 규제
   - L1 규제
9. 드롭아웃(Dropout)
10. 하이퍼파라미터(Hyper Parameter)
    - 학습률(Learning Rate)
    - 학습 횟수(Epochs)
    - 미니배치 크기(Mini Batch Size)
    - 검증데이터(Validation Data)
11. MNIST 분류
    - Modules Import
    - 데이터 로드, 전처리
    - Hyper Parameters
    - Util Function
    - Util Classes
      - ReLU
      - Sigmoid
      - Layer
      - Batch Normalization
      - Dropout
      - Softmax
12. Model
    - 모델 생성 및 학습1
      - 시각화
    - 모델 생성 및 학습2
      - 시각화
    - 모델 생성 및 학습3
      - 시각화
13. 3가지 모델 비교
## 9. CNN
1. 합성곱 신경망(Convolutional Neural Networks, CNNs)
   - 완전연결계층과의 차이
   - 컨볼루션 신경망 구조 예시
   - 합성곱 연산
   - 패딩(padding)과 스트라이드(stride)
     - 패딩
     - 스트라이드
   - 출력 데이터의 크기
   - 풀링(Pooling)
     - 맥스 풀링(Max Pooling)
     - 평균 풀링(Avg Pooling)
   - 합성곱 연산의 의미
   - 2차원 이미지에 대한 필터 연산 예시
     - modules import
     - util functions
     - 이미지 확인
     - 필터연산 적용
     - 이미지 필터를 적용한 최종 결과
   - 3차원 데이터의 합성곱 연산
     - 연산과정
     - modules import
     - util functions
     - 이미지 확인
     - 필터연산 적용
     - 필터연산을 적용한 최종 결과
     - 전체 과정 한번에 보기
2. 합성곱 신경망 구현
   - 합성곱 층(Convolution Layer)
     - 컨볼루션 레이어 테스트
     - 동일한 이미지 여러 장 테스트(배치 처리)
     - 동일한 이미지 배치처리(color)
   - 풀링 층(Pooling Layer)
     - 풀링 레이어 테스트
     - 동일한 이미지 배치처리
3. 대표적인 CNN 모델 소개
   - LeNet -5
   - AlexNet
   - VGG- 16
4. CNN 학습 구현 -MNIST
   - modules import
   - Util Functions
   - Util Classes
   - 데이터 로드
   - Build Model
   - Hyper Parameters
5. 모델 생성 및 학습
   - 학습이 잘 안된 이유
 
